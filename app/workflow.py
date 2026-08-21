"""LangGraph 显式工作流：DeepSeek 意图识别 -> 路由 -> RAG/资产查询 -> 生成回复。"""

from __future__ import annotations

import json
import re
from typing import Any, cast

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import END, StateGraph

from app.assets import AssetRepository
from app.models import AssetQueryInput, FinancialWorkflowState, IntentClassification
from app.prompts import (
    FINANCIAL_CUSTOMER_SERVICE_SYSTEM_PROMPT,
    INTENT_CLASSIFICATION_SYSTEM_PROMPT,
)


# LLM 失败时的兜底规则（不再作为主路径）
_ASSET_ID_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"资产编号[:：]?\s*([A-Za-z0-9_-]{4,32})"),
    re.compile(r"资产编码[:：]?\s*([A-Za-z0-9_-]{4,32})"),
    re.compile(r"采购单号[:：]?\s*([A-Za-z0-9_-]{4,32})"),
    re.compile(r"案例号[:：]?\s*([A-Za-z0-9_-]{4,32})"),
    re.compile(r"\b([A-Z]{2,}[A-Z0-9]{6,})\b"),
]

_ASSET_INTENT_KEYWORDS = [
    "资产编号",
    "资产编码",
    "采购单号",
    "资产查询",
    "查询资产",
    "核对资产",
    "对账",
    "资产信息",
    "资产核对",
]

_MIN_ASSET_CONFIDENCE = 0.55


def _extract_asset_id(text: str) -> str | None:
    for pattern in _ASSET_ID_PATTERNS:
        match = pattern.search(text or "")
        if match:
            return cast(str, match.group(1)).strip()
    return None


def _rule_based_classification(question: str) -> IntentClassification:
    """LLM 失败时的规则兜底：资产关键词走 asset_query，否则 faq_rag。"""
    if any(keyword in question for keyword in _ASSET_INTENT_KEYWORDS):
        return IntentClassification(
            intent="asset_query",
            asset_id=_extract_asset_id(question),
            confidence=0.6,
            reason="规则兜底：命中资产查询关键词",
        )
    return IntentClassification(
        intent="faq_rag",
        asset_id=None,
        confidence=0.6,
        reason="规则兜底：未匹配资产查询工作流",
    )


def _parse_classification_payload(payload: Any) -> IntentClassification:
    if isinstance(payload, IntentClassification):
        return payload
    if isinstance(payload, dict):
        return IntentClassification.model_validate(payload)
    if hasattr(payload, "model_dump"):
        return IntentClassification.model_validate(payload.model_dump())
    if hasattr(payload, "content"):
        content = payload.content
        if isinstance(content, IntentClassification):
            return content
        if isinstance(content, dict):
            return IntentClassification.model_validate(content)
        text = str(content or "").strip()
        if text.startswith("```"):
            text = re.sub(r"^```(?:json)?\s*", "", text)
            text = re.sub(r"\s*```$", "", text)
        return IntentClassification.model_validate(json.loads(text))
    raise TypeError(f"无法解析意图分类结果：{type(payload)!r}")


def _classify_with_llm(llm: Any, question: str) -> IntentClassification:
    messages = [
        SystemMessage(content=INTENT_CLASSIFICATION_SYSTEM_PROMPT),
        HumanMessage(content=question),
    ]
    try:
        structured = llm.with_structured_output(IntentClassification)
        return _parse_classification_payload(structured.invoke(messages))
    except Exception:
        # 部分模型不支持 structured_output，退化为要求 JSON 文本
        try:
            raw = llm.invoke(
                [
                    *messages,
                    HumanMessage(
                        content=(
                            "请只输出 JSON 对象，字段为 intent, asset_id, confidence, reason。"
                        )
                    ),
                ]
            )
            return _parse_classification_payload(raw)
        except Exception:
            return _rule_based_classification(question)


def _normalize_classification(
    classification: IntentClassification,
    question: str,
) -> IntentClassification:
    intent = classification.intent
    asset_id = (classification.asset_id or "").strip() or None
    confidence = float(classification.confidence or 0.0)

    if intent == "asset_query" and confidence < _MIN_ASSET_CONFIDENCE:
        return IntentClassification(
            intent="faq_rag",
            asset_id=None,
            confidence=confidence,
            reason=f"置信度不足，改为知识库兜底：{classification.reason}",
        )

    if intent == "asset_query" and not asset_id:
        asset_id = _extract_asset_id(question)

    return IntentClassification(
        intent=intent,
        asset_id=asset_id,
        confidence=confidence,
        reason=classification.reason,
    )


def _message_text(message: Any) -> str:
    content = getattr(message, "content", message)
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = [
            block.get("text", "")
            for block in content
            if isinstance(block, dict) and block.get("type") == "text"
        ]
        return "".join(parts)
    return str(content)


def _build_respond_user_prompt(state: FinancialWorkflowState) -> str:
    question = state.get("last_question", "")
    route = state.get("route")

    if route == "faq_rag":
        evidence = state.get("rag_evidence") or "（知识库未返回可用证据）"
        return (
            f"用户问题：{question}\n\n"
            f"知识库检索证据：\n{evidence}\n\n"
            "请基于上述证据回答。证据不足时明确说明，不要编造。"
        )

    if route == "asset_queryWithId":
        asset_id = state.get("asset_id") or ""
        result = state.get("asset_query_result") or {}
        return (
            f"用户问题：{question}\n\n"
            f"资产编码：{asset_id}\n"
            f"资产查询结构化结果（JSON）：\n{json.dumps(result, ensure_ascii=False, indent=2)}\n\n"
            "请用自然语言向用户说明查询结果。"
            "只能使用结果中的字段，不要编造未返回的数据。"
        )

    return f"用户问题：{question}\n\n请给出简短、克制的回复。"


def build_financial_workflow(
    *,
    search_tool: Any,
    asset_repo: AssetRepository,
    llm: Any,
) -> Any:
    """创建可编译的 LangGraph 工作流。"""

    def understand(state: FinancialWorkflowState) -> dict[str, Any]:
        messages = state["messages"]
        last_user = messages[-1]["content"] if messages else ""
        question = str(last_user or "")

        classification = _normalize_classification(
            _classify_with_llm(llm, question),
            question,
        )

        if classification.intent == "asset_query":
            route = (
                "asset_queryWithId"
                if classification.asset_id
                else "asset_queryMissingId"
            )
            return {
                "intent": "asset_query",
                "route": route,
                "asset_id": classification.asset_id,
                "last_question": question,
                "intent_confidence": classification.confidence,
                "intent_reason": classification.reason,
            }

        return {
            "intent": "faq_rag",
            "route": "faq_rag",
            "asset_id": None,
            "last_question": question,
            "intent_confidence": classification.confidence,
            "intent_reason": classification.reason,
        }

    def route_router(state: FinancialWorkflowState) -> str:
        route = state.get("route")
        if route in {"faq_rag", "asset_queryWithId"}:
            return "tools"
        return "clarify"

    def tools(state: FinancialWorkflowState) -> dict[str, Any]:
        route = state.get("route")
        question = state.get("last_question", "")

        if route == "faq_rag":
            evidence = search_tool.invoke(question)
            return {"rag_evidence": evidence}

        if route == "asset_queryWithId":
            asset_id = state.get("asset_id")
            if not asset_id:
                return {"asset_query_result": None}
            query = AssetQueryInput(asset_id=asset_id)
            result = asset_repo.query_asset_by_id(query)
            return {"asset_query_result": result.to_dict()}

        return {}

    def respond(state: FinancialWorkflowState) -> dict[str, Any]:
        messages = state["messages"]
        reply_messages = [
            SystemMessage(content=FINANCIAL_CUSTOMER_SERVICE_SYSTEM_PROMPT),
            HumanMessage(content=_build_respond_user_prompt(state)),
        ]
        try:
            ai_message = llm.invoke(reply_messages)
            content = _message_text(ai_message).strip()
        except Exception as exc:
            content = f"生成回复失败：{exc}"

        if not content:
            content = "抱歉，我暂时无法生成回复，请稍后重试。"

        assistant: dict[str, Any] = {"role": "assistant", "content": content}
        return {"messages": [*messages, assistant]}

    def clarify(state: FinancialWorkflowState) -> dict[str, Any]:
        messages = state["messages"]
        content = (
            "我可以帮你查询资产信息。请提供至少一个查询键："
            "资产编号、资产编码 或 采购单号（格式示例：`资产编码：FAJT221000600`）。"
        )
        assistant: dict[str, Any] = {"role": "assistant", "content": content}
        return {"messages": [*messages, assistant]}

    workflow = StateGraph(FinancialWorkflowState)
    workflow.add_node("understand", understand)
    workflow.add_node("tools", tools)
    workflow.add_node("respond", respond)
    workflow.add_node("clarify", clarify)

    workflow.set_entry_point("understand")
    workflow.add_conditional_edges("understand", route_router)
    workflow.add_edge("tools", "respond")
    workflow.add_edge("respond", END)
    workflow.add_edge("clarify", END)

    return workflow.compile()
