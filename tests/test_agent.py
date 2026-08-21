from pathlib import Path

from langchain_core.embeddings import FakeEmbeddings
from langchain_core.messages import AIMessage

from app.agent import create_financial_agent
from app.config import Settings
from app.models import IntentClassification


class FakeWorkflowLLM:
    """离线测试用假模型。"""

    def __init__(self) -> None:
        self.reply = "ok"

    def with_structured_output(self, schema):  # noqa: ANN001
        parent = self

        class _Structured:
            def invoke(self, messages):  # noqa: ANN001
                return IntentClassification(
                    intent="faq_rag",
                    confidence=0.9,
                    reason="test",
                )

        return _Structured()

    def invoke(self, messages):  # noqa: ANN001
        return AIMessage(content=self.reply)


def test_agent_registers_workflow_nodes(tmp_path: Path) -> None:
    knowledge_dir = tmp_path / "knowledge"
    knowledge_dir.mkdir()
    (knowledge_dir / "faq.md").write_text("开户需要身份证。", encoding="utf-8")

    settings = Settings(
        api_key="offline-test-key",
        rag_knowledge_dir=str(knowledge_dir),
        rag_index_dir=str(tmp_path / "index"),
    )
    agent = create_financial_agent(
        settings,
        embeddings=FakeEmbeddings(size=8),
        llm=FakeWorkflowLLM(),
    )

    node_names = set(agent.get_graph().nodes)

    assert "understand" in node_names
    assert "tools" in node_names
    assert "respond" in node_names
    assert "clarify" in node_names
    assert "fallback" not in node_names
