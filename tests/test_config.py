import pytest

from app.config import Settings


@pytest.fixture(autouse=True)
def disable_dotenv_loading(monkeypatch: pytest.MonkeyPatch) -> None:
    """避免用户创建真实 .env 后影响配置单元测试。"""
    monkeypatch.setattr("app.config.load_dotenv", lambda: False)


def test_default_model_is_deepseek_v4_flash(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DEEPSEEK_API_KEY", "test-key")
    monkeypatch.delenv("DEEPSEEK_MODEL", raising=False)

    settings = Settings.from_env()

    assert settings.model == "deepseek-v4-flash"


def test_api_key_is_required(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)

    with pytest.raises(RuntimeError, match="DEEPSEEK_API_KEY"):
        Settings.from_env()


def test_temperature_range_is_validated() -> None:
    settings = Settings(api_key="test-key", temperature=2.1)

    with pytest.raises(ValueError, match="TEMPERATURE"):
        settings.validate()
