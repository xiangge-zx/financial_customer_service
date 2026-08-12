"""创建不绑定任何业务工具的 LangChain 财经客服 Agent。"""

from langchain.agents import create_agent
from langchain_deepseek import ChatDeepSeek
from langgraph.graph.state import CompiledStateGraph

from app.config import Settings
from app.prompts import FINANCIAL_CUSTOMER_SERVICE_SYSTEM_PROMPT


def create_financial_agent(settings: Settings | None = None) -> CompiledStateGraph:
    """创建第一版财经客服 Agent。

    第一版只依赖大模型自身知识，不注册查询、计算或写入类工具。
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

    return create_agent(
        model=model,
        tools=[],
        system_prompt=FINANCIAL_CUSTOMER_SERVICE_SYSTEM_PROMPT,
        name="financial_customer_service_agent",
    )
