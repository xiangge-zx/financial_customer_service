from pathlib import Path

from langchain_core.embeddings import FakeEmbeddings

from app.agent import create_financial_agent
from app.config import Settings


def test_agent_registers_tools_node(tmp_path: Path) -> None:
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
    )

    node_names = set(agent.get_graph().nodes)

    assert "model" in node_names
    assert "tools" in node_names
