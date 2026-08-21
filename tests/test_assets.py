from unittest.mock import MagicMock, patch

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


def test_placeholder_asset_repository_returns_not_connected_result() -> None:
    repo = PlaceholderAssetRepository()

    result = repo.query_asset_by_id(AssetQueryInput(asset_id="FAJT221000600"))

    assert result.status == "not_connected"
    assert result.asset_id == "FAJT221000600"
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


@patch("pymysql.connect")
def test_mysql_asset_repository_returns_found_records(mock_connect: MagicMock) -> None:
    cursor = MagicMock()
    cursor.fetchall.return_value = [
        {"asset_name": "测试资产", "asset_category": "办公设备"}
    ]
    connection = MagicMock()
    connection.__enter__.return_value = connection
    connection.cursor.return_value.__enter__.return_value = cursor
    mock_connect.return_value.__enter__.return_value = connection

    repo = MySQLAssetRepository(
        host="localhost",
        port=3306,
        user="root",
        password="secret",
        database="customer",
    )

    result = repo.query_asset_by_id(AssetQueryInput(asset_id="FAJT221000600"))

    assert result.status == "found"
    assert result.records[0].asset_name == "测试资产"
    assert result.records[0].asset_category == "办公设备"
    cursor.execute.assert_called_once_with(SQL_ASSET_BY_CODE, ("FAJT221000600",))


@patch("pymysql.connect")
def test_mysql_asset_repository_returns_not_found(mock_connect: MagicMock) -> None:
    cursor = MagicMock()
    cursor.fetchall.return_value = []
    connection = MagicMock()
    connection.__enter__.return_value = connection
    connection.cursor.return_value.__enter__.return_value = cursor
    mock_connect.return_value.__enter__.return_value = connection

    repo = MySQLAssetRepository(
        host="localhost",
        port=3306,
        user="root",
        password="secret",
        database="customer",
    )

    result = repo.query_asset_by_id(AssetQueryInput(asset_id="FAJT221000600"))

    assert result.status == "not_found"
    assert "未在 customer.asset_dossier 中找到" in result.message
