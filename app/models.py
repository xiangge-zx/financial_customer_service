"""LangGraph 工作流状态与领域模型（Contract）。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, NotRequired, TypedDict

from pydantic import BaseModel, Field


Message = dict[str, Any]

Intent = Literal["faq_rag", "asset_query"]
Route = Literal[
    "faq_rag",
    "asset_queryWithCode",
    "asset_queryMissingCode",
]


class IntentClassification(BaseModel):
    """DeepSeek 结构化意图识别结果。"""

    intent: Intent = Field(
        description="asset_query=匹配资产查询工作流；faq_rag=未匹配现有工作流，走知识库"
    )
    asset_code: str | None = Field(
        default=None,
        description="从用户问题中提取的资产编码/编号；仅 asset_query 时填写",
    )
    confidence: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="分类置信度，0～1",
    )
    reason: str = Field(default="", description="简要分类理由")


class FinancialWorkflowState(TypedDict):
    """LangGraph state。

    为了与现有 CLI 保持一致，messages 使用 dict 结构（而不是 LangChain Message 对象）。
    """

    messages: list[Message]

    # understand 节点输出
    intent: NotRequired[Intent]
    route: NotRequired[Route]
    asset_code: NotRequired[str]
    last_question: NotRequired[str]
    intent_confidence: NotRequired[float]
    intent_reason: NotRequired[str]

    # tools 节点输出
    rag_evidence: NotRequired[str]
    asset_query_result: NotRequired[dict[str, Any]]


@dataclass(frozen=True)
class AssetQueryInput:
    asset_code: str
    # 后续扩展可把 query_scope 拓展成 legacy/new/both
    query_scope: Literal["legacy_and_new"] = "legacy_and_new"


@dataclass(frozen=True)
class AssetRecord:
    asset_name: str | None
    brand: str | None
    spec: str | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "asset_name": self.asset_name,
            "brand": self.brand,
            "spec": self.spec,
        }


@dataclass(frozen=True)
class AssetQueryResult:
    """资产查询返回结果。"""

    status: Literal["found", "not_found", "error", "not_connected"]
    asset_code: str
    query_scope: Literal["legacy_and_new"]
    sql_template: str
    records: tuple[AssetRecord, ...] = ()
    message: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "asset_code": self.asset_code,
            "query_scope": self.query_scope,
            "sql_template": self.sql_template,
            "records": [record.to_dict() for record in self.records],
            "message": self.message,
        }
