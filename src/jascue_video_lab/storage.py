from __future__ import annotations

import json
import os
import traceback
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


def jsonable(value: Any) -> Any:
    if isinstance(value, BaseModel):
        return value.model_dump(mode="json")
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json", exclude_none=False)
    return value


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(jsonable(value), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def error_record(stage: str, error: BaseException) -> dict[str, object]:
    return {
        "stage": stage,
        "error_type": type(error).__name__,
        "message": str(error),
        "occurred_at": utc_now(),
        "traceback": "".join(traceback.format_exception(error)),
    }


def append_error(run_dir: Path, stage: str, error: BaseException) -> None:
    path = run_dir / "errors.json"
    records = read_json(path) if path.exists() else []
    records.append(error_record(stage, error))
    write_json(path, records)

