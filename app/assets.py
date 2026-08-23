"""资产查询实现与数据访问层。"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

from app.models import AssetQueryInput, AssetQueryResult, AssetRecord

if TYPE_CHECKING:
    from app.config import Settings


SQL_ASSET_BY_CODE = """
SELECT
  ad.asset_name,
  ad.asset_category
FROM asset_dossier ad
WHERE ad.asset_code = %s
""".strip()


class AssetRepository(Protocol):
    """资产查询数据访问层。"""

    def query_asset_by_id(self, query: AssetQueryInput) -> AssetQueryResult:
        """按资产编码查询并返回结构化结果。"""


class PlaceholderAssetRepository:
    """未配置数据库时使用的占位实现。"""

    def query_asset_by_id(self, query: AssetQueryInput) -> AssetQueryResult:
        return AssetQueryResult(
            status="not_connected",
            asset_id=query.asset_id,
            query_scope=query.query_scope,
            sql_template=SQL_ASSET_BY_CODE,
            message=(
                "当前未配置 MySQL 连接，无法查询 asset_dossier。"
                "请在 .env 中填写 MYSQL_* 配置后重试。"
            ),
        )


class MySQLAssetRepository:
    """连接本地 MySQL 查询 customer.asset_dossier。"""

    def __init__(
        self,
        *,
        host: str,
        port: int,
        user: str,
        password: str,
        database: str,
    ) -> None:
        self.host = host
        self.port = port
        self.user = user
        self.password = password
        self.database = database

    def query_asset_by_id(self, query: AssetQueryInput) -> AssetQueryResult:
        try:
            import pymysql
            from pymysql.cursors import DictCursor
        except ImportError as exc:
            return AssetQueryResult(
                status="error",
                asset_id=query.asset_id,
                query_scope=query.query_scope,
                sql_template=SQL_ASSET_BY_CODE,
                message="缺少 pymysql 依赖，请执行 pip install pymysql。",
            )

        try:
            with pymysql.connect(
                host=self.host,
                port=self.port,
                user=self.user,
                password=self.password,
                database=self.database,
                charset="utf8mb4",
                cursorclass=DictCursor,
                connect_timeout=5,
                read_timeout=10,
            ) as connection:
                with connection.cursor() as cursor:
                    cursor.execute(SQL_ASSET_BY_CODE, (query.asset_id,))
                    rows = cursor.fetchall()
        except Exception as exc:  # 连接/权限/SQL 错误统一转为可读提示
            return AssetQueryResult(
                status="error",
                asset_id=query.asset_id,
                query_scope=query.query_scope,
                sql_template=SQL_ASSET_BY_CODE,
                message=f"数据库查询失败：{exc}",
            )

        if not rows:
            return AssetQueryResult(
                status="not_found",
                asset_id=query.asset_id,
                query_scope=query.query_scope,
                sql_template=SQL_ASSET_BY_CODE,
                message=f"未在 {self.database}.asset_dossier 中找到资产编码 {query.asset_id}。",
            )

        records = tuple(
            AssetRecord(
                asset_name=row.get("asset_name"),
                asset_category=row.get("asset_category"),
            )
            for row in rows
        )
        return AssetQueryResult(
            status="found",
            asset_id=query.asset_id,
            query_scope=query.query_scope,
            sql_template=SQL_ASSET_BY_CODE,
            records=records,
            message=f"已在 {self.database}.asset_dossier 查询到 {len(records)} 条记录。",
        )


def create_asset_repository(settings: Settings) -> AssetRepository:
    """根据配置创建资产仓储，供工作流 asset_query 分支调用。

    - 未配置 MYSQL_PASSWORD：返回 PlaceholderAssetRepository（只提示未连库，不尝试连接）
    - 已配置：返回 MySQLAssetRepository，按 asset_code 查 customer.asset_dossier
    """
    if not settings.mysql_password:
        return PlaceholderAssetRepository()
    return MySQLAssetRepository(
        host=settings.mysql_host,
        port=settings.mysql_port,
        user=settings.mysql_user,
        password=settings.mysql_password,
        database=settings.mysql_database,
    )
