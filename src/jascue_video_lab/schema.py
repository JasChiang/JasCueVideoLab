from __future__ import annotations

from copy import deepcopy
from typing import Any

from pydantic import BaseModel


# Interactions structured output supports a documented JSON Schema subset.
# Pydantic emits a few validation-only keywords outside that subset; keep
# those constraints in local model validation and send only supported syntax.
_DROP_KEYS = {"default", "examples", "minLength", "maxLength", "pattern"}


def gemini_response_schema(model: type[BaseModel]) -> dict[str, Any]:
    return _sanitize(deepcopy(model.model_json_schema()))


def _sanitize(value: Any) -> Any:
    if isinstance(value, list):
        return [_sanitize(item) for item in value]
    if not isinstance(value, dict):
        return value
    output: dict[str, Any] = {}
    for key, child in value.items():
        if key in _DROP_KEYS:
            continue
        if key == "const":
            output["enum"] = [_sanitize(child)]
            continue
        if key == "exclusiveMinimum":
            # All current exclusive lower bounds in the lab schemas are ints.
            output["minimum"] = child + 1 if isinstance(child, int) else child
            continue
        if key == "exclusiveMaximum":
            output["maximum"] = child - 1 if isinstance(child, int) else child
            continue
        output[key] = _sanitize(child)
    return output
