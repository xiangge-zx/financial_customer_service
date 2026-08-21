from pathlib import Path

from langchain_core.embeddings import FakeEmbeddings
from langchain_core.messages import AIMessage

from app.assets import PlaceholderAssetRepository
from app.config import Settings
from app.models import IntentClassification
from app.rag import build_or_load_vector_store, create_search_tool
from app.workflow import build_financial_workflow


class FakeWorkflowLLM:
    """离线测试用假模型：支持意图结构化输出与回复生成。"""

    def __init__(
        self,
        *,
        classifications: list[IntentClassification] | None = None,
        reply: str = "这是基于证据的测试回复。",
        raise_on_classify: bool = False,
    ) -> None:
        self.classifications = list(classifications or [])
        self.reply = reply
        self.raise_on_classify = raise_on_classify
        self.classify_calls = 0
        self.respond_calls = 0

    def with_structured_output(self, schema):  # noqa: ANN001
        parent = self

        class _Structured:
            def invoke(self, messages):  # noqa: ANN001
                parent.classify_calls += 1
                if parent.raise_on_classify:
                    raise RuntimeError("structured output unavailable")
                if parent.classifications:
                    return parent.classifications.pop(0)
                return IntentClassification(
                    intent="faq_rag",
                    confidence=0.9,
                    reason="默认 FAQ",
                )

        return _Structured()

    def invoke(self, messages):  # noqa: ANN001
        self.respond_calls += 1
        # JSON 兜底路径：structured_output 失败后会再 invoke 一次要求 JSON
        joined = " ".join(str(getattr(m, "content", m)) for m in messages)
        if "只输出 JSON" in joined or "intent, asset_id, confidence, reason" in joined:
            if self.raise_on_classify:
                raise RuntimeError("json classify unavailable")
            return AIMessage(
                content='{"intent":"faq_rag","asset_id":null,"confidence":0.4,"reason":"json fallback"}'
            )
        return AIMessage(content=self.reply)


def _build_workflow(
    tmp_path: Path,
    *,
    llm: FakeWorkflowLLM | None = None,
):
    knowledge_dir = tmp_path / "knowledge"
    knowledge_dir.mkdir()
    (knowledge_dir / "开户FAQ.md").write_text(
        "个人开户需要携带本人有效身份证件原件。",
        encoding="utf-8",
    )

    settings = Settings(
        api_key="offline-test-key",
        rag_knowledge_dir=str(knowledge_dir),
        rag_index_dir=str(tmp_path / "index"),
        rag_top_k=2,
        embedding_model="fake-embeddings",
    )
    store = build_or_load_vector_store(
        settings,
        embeddings=FakeEmbeddings(size=16),
    )
    tool = create_search_tool(store, top_k=2)
    return build_financial_workflow(
        search_tool=tool,
        asset_repo=PlaceholderAssetRepository(),
        llm=llm or FakeWorkflowLLM(),
    )


def test_workflow_routes_faq_to_rag(tmp_path: Path) -> None:
    llm = FakeWorkflowLLM(
        classifications=[
            IntentClassification(
                intent="faq_rag",
                confidence=0.92,
                reason="开户 FAQ",
            )
        ],
        reply="开户需要携带有效身份证件。",
    )
    workflow = _build_workflow(tmp_path, llm=llm)

    result = workflow.invoke({"messages": [{"role": "user", "content": "开户需要什么材料"}]})

    assert result["intent"] == "faq_rag"
    assert result["route"] == "faq_rag"
    assert "身份证件" in result["rag_evidence"]
    assert "开户需要携带有效身份证件" in result["messages"][-1]["content"]
    assert llm.respond_calls >= 1


def test_workflow_routes_asset_query_with_id(tmp_path: Path) -> None:
    llm = FakeWorkflowLLM(
        classifications=[
            IntentClassification(
                intent="asset_query",
                asset_id="ABC12345",
                confidence=0.95,
                reason="用户要查资产",
            )
        ],
        reply="已查到资产编码 ABC12345 的占位结果。",
    )
    workflow = _build_workflow(tmp_path, llm=llm)

    result = workflow.invoke(
        {"messages": [{"role": "user", "content": "帮我查询资产信息，资产编号：ABC12345"}]}
    )

    assert result["intent"] == "asset_query"
    assert result["route"] == "asset_queryWithId"
    assert result["asset_id"] == "ABC12345"
    assert result["asset_query_result"]["status"] == "not_connected"
    assert "ABC12345" in result["messages"][-1]["content"]


def test_workflow_clarifies_when_asset_id_missing(tmp_path: Path) -> None:
    llm = FakeWorkflowLLM(
        classifications=[
            IntentClassification(
                intent="asset_query",
                asset_id=None,
                confidence=0.9,
                reason="要查资产但没给编码",
            )
        ]
    )
    workflow = _build_workflow(tmp_path, llm=llm)

    result = workflow.invoke({"messages": [{"role": "user", "content": "我想查询资产信息"}]})

    assert result["intent"] == "asset_query"
    assert result["route"] == "asset_queryMissingId"
    assert "请提供至少一个查询键" in result["messages"][-1]["content"]
    assert llm.respond_calls == 0


def test_workflow_unmatched_intent_falls_back_to_rag(tmp_path: Path) -> None:
    llm = FakeWorkflowLLM(
        classifications=[
            IntentClassification(
                intent="faq_rag",
                confidence=0.8,
                reason="闲聊，未匹配资产工作流",
            )
        ],
        reply="我可以帮你查业务知识或资产信息。",
    )
    workflow = _build_workflow(tmp_path, llm=llm)

    result = workflow.invoke({"messages": [{"role": "user", "content": "你好，陪我聊聊天"}]})

    assert result["intent"] == "faq_rag"
    assert result["route"] == "faq_rag"
    assert "rag_evidence" in result
    assert "业务知识或资产信息" in result["messages"][-1]["content"]


def test_workflow_low_confidence_asset_query_falls_back_to_rag(tmp_path: Path) -> None:
    llm = FakeWorkflowLLM(
        classifications=[
            IntentClassification(
                intent="asset_query",
                asset_id="ABC12345",
                confidence=0.2,
                reason="不太确定",
            )
        ],
        reply="改为知识库兜底回复。",
    )
    workflow = _build_workflow(tmp_path, llm=llm)

    result = workflow.invoke(
        {"messages": [{"role": "user", "content": "好像和资产有关？ABC12345"}]}
    )

    assert result["intent"] == "faq_rag"
    assert result["route"] == "faq_rag"
    assert "改为知识库兜底回复" in result["messages"][-1]["content"]


def test_workflow_classify_failure_uses_rule_fallback(tmp_path: Path) -> None:
    llm = FakeWorkflowLLM(raise_on_classify=True, reply="规则兜底后的回复。")
    workflow = _build_workflow(tmp_path, llm=llm)

    result = workflow.invoke(
        {"messages": [{"role": "user", "content": "帮我查询资产信息，资产编码：FAJT221000600"}]}
    )

    assert result["intent"] == "asset_query"
    assert result["route"] == "asset_queryWithId"
    assert result["asset_id"] == "FAJT221000600"
    assert "规则兜底" in result["intent_reason"]
