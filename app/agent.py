"""构建“财经客服工作流”LangGraph。

本模块是装配入口（工厂）：把配置、LLM、向量库检索工具、资产仓储
拼进 LangGraph，返回可 stream / invoke 的 CompiledStateGraph。
真正的节点逻辑在 app/workflow.py。
"""

from __future__ import annotations

from typing import Any

from langchain_core.embeddings import Embeddings
from langgraph.graph.state import CompiledStateGraph

from app.assets import create_asset_repository
from app.config import Settings
from app.rag import build_or_load_vector_store, create_search_tool
from app.workflow import build_financial_workflow


def _create_deepseek_llm(settings: Settings) -> Any:
    """按 Settings 创建 DeepSeek Chat 模型客户端。

    延迟 import langchain_deepseek：避免未装依赖或未配密钥时，
    一 import app.agent 就失败（测试可注入假 llm 绕过）。
    """
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
    """创建带 DeepSeek 意图识别、RAG 与资产查询能力的财经客服工作流。

    参数均可注入，便于单测：
    - settings：不传则 Settings.from_env() 读 .env
    - embeddings：不传则索引器按 EMBEDDING_MODEL 本地加载
    - llm：不传则走 DeepSeek；测试可塞假模型

    返回值交给 cli.run_cli：agent.stream / invoke。
    """

    # 1) 配置：API Key、模型、RAG 路径、MySQL 等（缺 DEEPSEEK_API_KEY 会抛 RuntimeError）
    resolved_settings = settings or Settings.from_env()

    # 2) LLM：意图识别 + 最终回复共用同一客户端
    model = llm or _create_deepseek_llm(resolved_settings)

    # 3) 向量库：扫描 knowledge/，有缓存则复用 .rag_index/，源文件变更则重建
    vector_store = build_or_load_vector_store(
        resolved_settings,
        embeddings=embeddings,
    )

    # 4) 检索 Tool：封装 similarity_search，供工作流 faq_rag 分支调用
    search_tool = create_search_tool(
        vector_store,
        top_k=resolved_settings.rag_top_k,
    )

    # 5) 资产仓储：有 MYSQL_PASSWORD 连库查 asset_dossier，否则占位提示
    asset_repo = create_asset_repository(resolved_settings)

    # 6) 编译图：understand → route → tools|clarify → respond
    return build_financial_workflow(
        search_tool=search_tool,
        asset_repo=asset_repo,
        llm=model,
    )
