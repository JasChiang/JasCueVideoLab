"""Small durable artifacts shared by planning orchestration entry points."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any


def asked(*parts: str) -> str:
    """Stable compact cache key for one exact provider question."""

    return hashlib.sha256("\0".join(parts).encode("utf-8")).hexdigest()[:16]


def decided(work: Path, name: str, key: str) -> dict | None:
    """Return a paid decision only when its full question still matches."""

    path = Path(work) / f"{name}.json"
    try:
        saved = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return saved.get("value") if saved.get("key") == key else None


def decide(work: Path, name: str, key: str, value: dict) -> dict:
    """Atomically publish the compatibility cache used by existing runs."""

    path = Path(work) / f"{name}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    document = json.dumps({"key": key, "value": value}, ensure_ascii=False)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}-", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(document)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)
    return value


def planning_contract(prompt_name: str, schema: dict[str, Any]) -> str:
    """Hash everything that gives a planning answer its meaning."""

    from montagewright.capabilities import (
        describe_for_prompt,
        describe_limits_for_prompt,
    )
    from montagewright.planner import (
        MAX_OUTPUT_TOKENS,
        MODEL_ID,
        PROMPTS,
        THINKING_HIGH,
    )
    prompt = (PROMPTS / prompt_name).read_text(encoding="utf-8")
    return asked(
        MODEL_ID,
        f"thinking={THINKING_HIGH}|max_output={MAX_OUTPUT_TOKENS}",
        prompt,
        json.dumps(schema, ensure_ascii=False, sort_keys=True),
        describe_for_prompt(),
        describe_limits_for_prompt(),
    )
