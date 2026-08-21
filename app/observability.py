"""结构化日志接口。Phase 1 提供最小实现，Phase 5 扩展为审计日志。"""

from __future__ import annotations

import json
import logging
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


LOGGER = logging.getLogger("financial_agent")


def new_trace_id() -> str:
    return uuid.uuid4().hex[:16]


def configure_logging(*, level: int = logging.INFO) -> None:
    if LOGGER.handlers:
        return
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    LOGGER.addHandler(handler)
    LOGGER.setLevel(level)
    LOGGER.propagate = False


def log_event(
    event: str,
    *,
    trace_id: str,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """记录一条结构化事件，返回写入内容便于测试。"""

    payload: dict[str, Any] = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "event": event,
        "trace_id": trace_id,
    }
    if extra:
        payload.update(extra)
    LOGGER.info("%s", json.dumps(payload, ensure_ascii=False))
    return payload


def append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
