"""从环境变量读取 DeepSeek 配置。"""

from dataclasses import dataclass
import os

from dotenv import load_dotenv


@dataclass(frozen=True)
class Settings:
    """运行财经客服 Agent 所需的最小配置。"""

    api_key: str
    model: str = "deepseek-v4-flash"
    api_base: str = "https://api.deepseek.com"
    temperature: float = 0.2
    max_tokens: int = 2048
    timeout_seconds: float = 60.0
    max_retries: int = 2

    @classmethod
    def from_env(cls) -> "Settings":
        load_dotenv()

        api_key = os.getenv("DEEPSEEK_API_KEY", "").strip()
        if not api_key:
            raise RuntimeError(
                "未配置 DEEPSEEK_API_KEY，请复制 .env.example 为 .env 后填写密钥。"
            )

        settings = cls(
            api_key=api_key,
            model=os.getenv("DEEPSEEK_MODEL", cls.model).strip(),
            api_base=os.getenv("DEEPSEEK_API_BASE", cls.api_base).strip(),
            temperature=float(os.getenv("DEEPSEEK_TEMPERATURE", cls.temperature)),
            max_tokens=int(os.getenv("DEEPSEEK_MAX_TOKENS", cls.max_tokens)),
            timeout_seconds=float(
                os.getenv("DEEPSEEK_TIMEOUT_SECONDS", cls.timeout_seconds)
            ),
            max_retries=int(os.getenv("DEEPSEEK_MAX_RETRIES", cls.max_retries)),
        )
        settings.validate()
        return settings

    def validate(self) -> None:
        if not self.model:
            raise ValueError("DEEPSEEK_MODEL 不能为空。")
        if not self.api_base.startswith(("http://", "https://")):
            raise ValueError("DEEPSEEK_API_BASE 必须是 http 或 https 地址。")
        if not 0 <= self.temperature <= 2:
            raise ValueError("DEEPSEEK_TEMPERATURE 必须在 0 到 2 之间。")
        if self.max_tokens <= 0:
            raise ValueError("DEEPSEEK_MAX_TOKENS 必须大于 0。")
        if self.timeout_seconds <= 0:
            raise ValueError("DEEPSEEK_TIMEOUT_SECONDS 必须大于 0。")
        if self.max_retries < 0:
            raise ValueError("DEEPSEEK_MAX_RETRIES 不能小于 0。")
