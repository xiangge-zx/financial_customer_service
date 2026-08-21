"""构建“财经客服工作流”LangGraph。"""

from __future__ import annotations

from typing import Any

from langchain_core.embeddings import Embeddings
from langgraph.graph.state import CompiledStateGraph

from app.assets import create_asset_repository
from app.config import Settings
from app.rag import build_or_load_vector_store, create_search_tool
from app.workflow import build_financial_workflow


def _create_deepseek_llm(settings: Settings) -> Any:
    from langchain_deepseek import ChatDeepSeek

    return ChatDeepSeek(
        model=settings.model,
        api_key=settings.api_key,
        api_base=settings.api_base,
        temperature=settings.temperature,
        max_tokens=settings.max_tokens,
        timeout=settings.timeout_seconds,
        max_retries=settings.max_retries,
    )


def create_financial_agent(
    settings: Settings | None = None,
    *,
    embeddings: Embeddings | None = None,
    llm: Any | None = None,
) -> CompiledStateGraph:
    """创建带 DeepSeek 意图识别、RAG 与资产查询能力的财经客服工作流。"""

    resolved_settings = settings or Settings.from_env()
    model = llm or _create_deepseek_llm(resolved_settings)

    vector_store = build_or_load_vector_store(
        resolved_settings,
        embeddings=embeddings,
    )
    search_tool = create_search_tool(
        vector_store,
        top_k=resolved_settings.rag_top_k,
    )

    asset_repo = create_asset_repository(resolved_settings)
    return build_financial_workflow(
        search_tool=search_tool,
        asset_repo=asset_repo,
        llm=model,
    )
