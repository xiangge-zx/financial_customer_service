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
      CLARIFY_ASSET_CODE_PROMPT,
    FINANCIAL_CUSTOMER_SERVICE_SYSTEM_PROMPT,
    INTENT_CLASSIFICATION_SYSTEM_PROMPT,
)


# LLM 失败时的兜底规则（不再作为主路径）
_ASSET_CODE_PATTERNS: list[re.Pattern[str]] = [
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

_RAG_REFUSE_MESSAGE = "你好 我是一个专业的财经智能客服 无法回答无关问题"

_NO_EVIDENCE_MARKERS = (
    "知识库中没有找到相关内容",
    "知识库暂无可用文档",
    "未检索到与问题相关的内容",
    "检索词为空",
)


def _extract_asset_code(text: str) -> str | None:
    for pattern in _ASSET_CODE_PATTERNS:
        match = pattern.search(text or "")
        if match:
            return cast(str, match.group(1)).strip()
    return None


def _get_message_content(message: Any) -> str:
    if isinstance(message, dict):
        return str(message.get("content") or "")
    return _message_text(message)


def _is_clarify_response(content: str) -> bool:
    return CLARIFY_ASSET_CODE_PROMPT in content


def _is_awaiting_asset_code(messages: list[Any]) -> bool:
    """上一轮助手是否在追问资产编号。"""
    if len(messages) < 2:
        return False
    for message in reversed(messages[:-1]):
        if not isinstance(message, dict):
            continue
        role = message.get("role")
        if role == "assistant":
            return _is_clarify_response(_get_message_content(message))
        if role == "user":
            break
    return False


def _build_classification_input(messages: list[Any], question: str) -> str:
    """把最近几轮对话拼成意图分类输入，便于识别补槽轮次。"""
    recent = messages[-6:]
    parts: list[str] = []
    for message in recent:
        if not isinstance(message, dict):
            continue
        role = str(message.get("role") or "unknown")
        content = _get_message_content(message).strip()
        if content:
            parts.append(f"{role}: {content}")
    return "\n".join(parts) if parts else question


def _try_slot_fill_classification(
    messages: list[Any],
    question: str,
) -> IntentClassification | None:
    """确定性槽位回填：上一轮已追问资产编号，本轮只提供编号。"""
    asset_code = _extract_asset_code(question)
    if not asset_code or not _is_awaiting_asset_code(messages):
        return None
    return IntentClassification(
        intent="asset_query",
        asset_code=asset_code,
        confidence=1.0,
        reason="槽位回填：上一轮已追问资产编号，本轮提供编号",
    )


def _rule_based_classification(question: str) -> IntentClassification:
    """LLM 失败时的规则兜底：资产关键词走 asset_query，否则 faq_rag。"""
    if any(keyword in question for keyword in _ASSET_INTENT_KEYWORDS):
        return IntentClassification(
            intent="asset_query",
            asset_code=_extract_asset_code(question),
            confidence=0.6,
            reason="规则兜底：命中资产查询关键词",
        )
    return IntentClassification(
        intent="faq_rag",
        asset_code=None,
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


def _classify_with_llm(llm: Any, classification_input: str) -> IntentClassification:
    messages = [
        SystemMessage(content=INTENT_CLASSIFICATION_SYSTEM_PROMPT),
        HumanMessage(content=classification_input),
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
                            "请只输出 JSON 对象，字段为 intent, asset_code, confidence, reason。"
                        )
                    ),
                ]
            )
            return _parse_classification_payload(raw)
        except Exception:
            return _rule_based_classification(classification_input)


def _normalize_classification(
    classification: IntentClassification,
    question: str,
) -> IntentClassification:
    intent = classification.intent
    asset_code = (classification.asset_code or "").strip() or None
    confidence = float(classification.confidence or 0.0)

    if intent == "asset_query" and confidence < _MIN_ASSET_CONFIDENCE:
        return IntentClassification(
            intent="faq_rag",
            asset_code=None,
            confidence=confidence,
            reason=f"置信度不足，改为知识库兜底：{classification.reason}",
        )

    if intent == "asset_query" and not asset_code:
        asset_code = _extract_asset_code(question)

    return IntentClassification(
        intent=intent,
        asset_code=asset_code,
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


def _is_rag_evidence_empty(evidence: str | None) -> bool:
    if not evidence or not evidence.strip():
        return True
    text = evidence.strip()
    if text == "（知识库未返回可用证据）":
        return True
    return any(marker in text for marker in _NO_EVIDENCE_MARKERS)


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

    if route == "asset_queryWithCode":
        asset_code = state.get("asset_code") or ""
        result = state.get("asset_query_result") or {}
        return (
            f"用户问题：{question}\n\n"
            f"资产编码：{asset_code}\n"
            f"资产查询结构化结果（JSON）：\n{json.dumps(result, ensure_ascii=False, indent=2)}\n\n"
            "请用自然语言向用户说明查询结果（资产名称、品牌、规格等）。"
            "只能使用结果中的字段，不要编造未返回的数据。"
        )

    return f"用户问题：{question}\n\n请给出简短、克制的回复。"


def build_financial_workflow(
    *,
    search_tool: Any,
    asset_repo: AssetRepository,
    llm: Any,
) -> Any:
    """装配并 compile 财经客服图，由 create_financial_agent 调用。

    依赖三项外部能力（均在 agent 层造好再注入）：
    - llm：意图分类 + 最终回复
    - search_tool：faq_rag 时检索 knowledge/
    - asset_repo：asset_query 时查 MySQL asset_dossier

    节点：understand → (tools|clarify) → respond|END。
    """

    def understand(state: FinancialWorkflowState) -> dict[str, Any]:
        messages = state["messages"]
        last_user = messages[-1]["content"] if messages else ""
        question = str(last_user or "")

        slot_fill = _try_slot_fill_classification(messages, question)
        if slot_fill is not None:
            classification = slot_fill
        else:
            classification_input = _build_classification_input(messages, question)
            classification = _normalize_classification(
                _classify_with_llm(llm, classification_input),
                question,
            )

        if classification.intent == "asset_query":
            route = (
                "asset_queryWithCode"
                if classification.asset_code
                else "asset_queryMissingCode"
            )
            return {
                "intent": "asset_query",
                "route": route,
                "asset_code": classification.asset_code,
                "last_question": question,
                "intent_confidence": classification.confidence,
                "intent_reason": classification.reason,
            }

        return {
            "intent": "faq_rag",
            "route": "faq_rag",
            "asset_code": None,
            "last_question": question,
            "intent_confidence": classification.confidence,
            "intent_reason": classification.reason,
        }

    def route_router(state: FinancialWorkflowState) -> str:
        route = state.get("route")
        if route in {"faq_rag", "asset_queryWithCode"}:
            return "tools"
        return "clarify"

    def tools(state: FinancialWorkflowState) -> dict[str, Any]:
        route = state.get("route")
        question = state.get("last_question", "")

        if route == "faq_rag":
            evidence = search_tool.invoke(question)
            return {"rag_evidence": evidence}

        if route == "asset_queryWithCode":
            asset_code = state.get("asset_code")
            if not asset_code:
                return {"asset_query_result": None}
            query = AssetQueryInput(asset_code=asset_code)
            result = asset_repo.query_asset_by_code(query)
            return {"asset_query_result": result.to_dict()}

        return {}

    def respond(state: FinancialWorkflowState) -> dict[str, Any]:
        messages = state["messages"]

        if state.get("route") == "faq_rag" and _is_rag_evidence_empty(
            state.get("rag_evidence")
        ):
            content = _RAG_REFUSE_MESSAGE
        else:
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
        assistant: dict[str, Any] = {
            "role": "assistant",
            "content": CLARIFY_ASSET_CODE_PROMPT,
        }
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
