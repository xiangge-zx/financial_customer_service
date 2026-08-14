from pathlib import Path

from langchain_core.embeddings import FakeEmbeddings

from app.config import Settings
from app.rag import build_or_load_vector_store, create_search_tool


def _settings_for(tmp_path: Path, knowledge_dir: Path) -> Settings:
    return Settings(
        api_key="offline-test-key",
        rag_knowledge_dir=str(knowledge_dir),
        rag_index_dir=str(tmp_path / "index"),
        rag_top_k=2,
        embedding_model="fake-embeddings",
    )


def test_build_index_and_search_returns_source(tmp_path: Path) -> None:
    knowledge_dir = tmp_path / "knowledge"
    knowledge_dir.mkdir()
    (knowledge_dir / "开户与账户.md").write_text(
        "个人开户需要携带本人有效身份证件原件。",
        encoding="utf-8",
    )

    settings = _settings_for(tmp_path, knowledge_dir)
    embeddings = FakeEmbeddings(size=16)
    store = build_or_load_vector_store(settings, embeddings=embeddings)
    tool = create_search_tool(store, top_k=2)

    result = tool.invoke("开户需要什么材料")

    assert "开户与账户.md" in result
    assert "身份证件" in result


def test_reuse_cached_index_when_sources_unchanged(tmp_path: Path) -> None:
    knowledge_dir = tmp_path / "knowledge"
    knowledge_dir.mkdir()
    (knowledge_dir / "faq.md").write_text("转账通常实时到账。", encoding="utf-8")

    settings = _settings_for(tmp_path, knowledge_dir)
    embeddings = FakeEmbeddings(size=16)

    first = build_or_load_vector_store(settings, embeddings=embeddings)
    second = build_or_load_vector_store(settings, embeddings=embeddings)

    assert first.index.ntotal == second.index.ntotal
    assert (tmp_path / "index" / "manifest.json").is_file()


def test_empty_knowledge_dir_search_returns_friendly_message(tmp_path: Path) -> None:
    knowledge_dir = tmp_path / "knowledge"
    knowledge_dir.mkdir()

    settings = _settings_for(tmp_path, knowledge_dir)
    store = build_or_load_vector_store(
        settings,
        embeddings=FakeEmbeddings(size=8),
    )
    tool = create_search_tool(store, top_k=1)

    result = tool.invoke("任意问题")

    assert "暂无可用文档" in result or "没有找到" in result
