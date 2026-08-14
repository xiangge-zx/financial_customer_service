"""接入本地知识库检索工具的 LangChain 财经客服 Agent。"""

from langchain.agents import create_agent
from langchain_core.embeddings import Embeddings
from langchain_deepseek import ChatDeepSeek
from langgraph.graph.state import CompiledStateGraph

from app.config import Settings
from app.prompts import FINANCIAL_CUSTOMER_SERVICE_SYSTEM_PROMPT
from app.rag import build_or_load_vector_store, create_search_tool


def create_financial_agent(
    settings: Settings | None = None,
    *,
    embeddings: Embeddings | None = None,
) -> CompiledStateGraph:
    """创建带 RAG 检索能力的财经客服 Agent。

    Agent 可通过 search_project_knowledge 搜索 knowledge/ 业务文档；
    仍不注册实时行情、账户或外部业务系统查询工具。
    """
    resolved_settings = settings or Settings.from_env()

    model = ChatDeepSeek(
        model=resolved_settings.model,
        api_key=resolved_settings.api_key,
        api_base=resolved_settings.api_base,
        temperature=resolved_settings.temperature,
        max_tokens=resolved_settings.max_tokens,
        timeout=resolved_settings.timeout_seconds,
        max_retries=resolved_settings.max_retries,
    )

    vector_store = build_or_load_vector_store(
        resolved_settings,
        embeddings=embeddings,
    )
    search_tool = create_search_tool(
        vector_store,
        top_k=resolved_settings.rag_top_k,
    )

    return create_agent(
        model=model,
        tools=[search_tool],
        system_prompt=FINANCIAL_CUSTOMER_SERVICE_SYSTEM_PROMPT,
        name="financial_customer_service_agent",
    )
