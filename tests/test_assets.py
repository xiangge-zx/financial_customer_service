from unittest.mock import MagicMock, patch
import sys

from app.assets import (
    MySQLAssetRepository,
    PlaceholderAssetRepository,
    SQL_ASSET_BY_CODE,
    create_asset_repository,
)
from app.config import Settings
from app.models import AssetQueryInput


def test_sql_template_uses_parameterized_asset_code() -> None:
    assert "%s" in SQL_ASSET_BY_CODE
    assert "asset_dossier" in SQL_ASSET_BY_CODE
    assert "ad.asset_code" in SQL_ASSET_BY_CODE
    assert "ad.brand" in SQL_ASSET_BY_CODE
    assert "ad.spec" in SQL_ASSET_BY_CODE
    assert "asset_category" not in SQL_ASSET_BY_CODE


def test_placeholder_asset_repository_returns_not_connected_result() -> None:
    repo = PlaceholderAssetRepository()

    result = repo.query_asset_by_code(AssetQueryInput(asset_code="FAJT221000600"))

    assert result.status == "not_connected"
    assert result.asset_code == "FAJT221000600"
    assert "asset_dossier" in result.sql_template
    assert "未配置 MySQL" in result.message


def test_create_asset_repository_uses_mysql_when_password_configured() -> None:
    settings = Settings(
        api_key="test-key",
        mysql_password="secret",
    )

    repo = create_asset_repository(settings)

    assert isinstance(repo, MySQLAssetRepository)


def test_create_asset_repository_uses_placeholder_without_password() -> None:
    settings = Settings(api_key="test-key")

    repo = create_asset_repository(settings)

    assert isinstance(repo, PlaceholderAssetRepository)


def _mock_pymysql_modules(*, rows: list[dict[str, str]]) -> tuple[MagicMock, MagicMock, patch]:
    cursor = MagicMock()
    cursor.fetchall.return_value = rows
    connection = MagicMock()
    connection.__enter__.return_value = connection
    connection.cursor.return_value.__enter__.return_value = cursor

    mock_cursors = MagicMock()
    mock_cursors.DictCursor = MagicMock()
    mock_pymysql = MagicMock()
    mock_pymysql.connect.return_value.__enter__.return_value = connection
    mock_pymysql.cursors = mock_cursors

    module_patch = patch.dict(
        sys.modules,
        {"pymysql": mock_pymysql, "pymysql.cursors": mock_cursors},
    )
    return mock_pymysql, cursor, module_patch


def test_mysql_asset_repository_returns_found_records() -> None:
    mock_pymysql, cursor, module_patch = _mock_pymysql_modules(
        rows=[{"asset_name": "测试资产", "brand": "示例品牌", "spec": "规格A"}]
    )
    with module_patch:
        repo = MySQLAssetRepository(
            host="localhost",
            port=3306,
            user="root",
            password="secret",
            database="customer",
        )

        result = repo.query_asset_by_code(AssetQueryInput(asset_code="FAJT221000600"))

    assert result.status == "found"
    assert result.records[0].asset_name == "测试资产"
    assert result.records[0].brand == "示例品牌"
    assert result.records[0].spec == "规格A"
    cursor.execute.assert_called_once_with(SQL_ASSET_BY_CODE, ("FAJT221000600",))
    mock_pymysql.connect.assert_called_once()


def test_mysql_asset_repository_returns_not_found() -> None:
    _, cursor, module_patch = _mock_pymysql_modules(rows=[])
    with module_patch:
        repo = MySQLAssetRepository(
            host="localhost",
            port=3306,
            user="root",
            password="secret",
            database="customer",
        )

        result = repo.query_asset_by_code(AssetQueryInput(asset_code="FAJT221000600"))

    assert result.status == "not_found"
    assert "未在 customer.asset_dossier 中找到" in result.message
    cursor.execute.assert_called_once_with(SQL_ASSET_BY_CODE, ("FAJT221000600",))
