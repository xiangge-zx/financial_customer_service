"""将向量检索封装为 Agent 可调用的工具。"""

from __future__ import annotations

from pathlib import Path

from langchain.tools import tool
from langchain_community.vectorstores import FAISS


def _source_label(metadata: dict) -> str:
    source = metadata.get("source")
    if not source:
        return "未知来源"
    return Path(str(source)).name


def create_search_tool(vector_store: FAISS, *, top_k: int = 4):
    """创建搜索项目知识库的 Tool。"""

    @tool
    def search_project_knowledge(query: str) -> str:
        """搜索项目 knowledge/ 目录中的业务文档。

        在回答开户、账户、理财产品、转账、挂失等业务 FAQ，或需要引用项目内
        说明文档时使用。传入简洁的中文检索词或用户问题即可。
        """
        question = (query or "").strip()
        if not question:
            return "检索词为空，请提供需要查询的业务问题。"

        docs = vector_store.similarity_search(question, k=top_k)
        if not docs:
            return "知识库中没有找到相关内容。"

        useful_parts: list[str] = []
        for doc in docs:
            content = (doc.page_content or "").strip()
            if not content or content == "知识库暂无可用文档。":
                continue
            useful_parts.append(f"[来源: {_source_label(doc.metadata)}]\n{content}")

        if not useful_parts:
            return "知识库暂无可用文档，或未检索到与问题相关的内容。"

        return "\n\n".join(useful_parts)

    return search_project_knowledge
