"""One adapter for the Gemini Interactions request contract."""

from __future__ import annotations

import json
import math
from typing import Any


def structured_json(schema: dict[str, Any]) -> dict[str, Any]:
    """The current Interactions API structured-text response shape."""

    return {
        "type": "text",
        "mime_type": "application/json",
        "schema": schema,
    }


def _count_contents(value: Any) -> Any:
    """Translate Interactions content parts to countTokens content parts."""

    if isinstance(value, str):
        return value
    if not isinstance(value, list):
        raise TypeError("budget preflight only supports text or content lists")

    from google.genai import types

    resolution_levels = {
        "low": types.PartMediaResolutionLevel.MEDIA_RESOLUTION_LOW,
        "medium": types.PartMediaResolutionLevel.MEDIA_RESOLUTION_MEDIUM,
        "high": types.PartMediaResolutionLevel.MEDIA_RESOLUTION_HIGH,
        "ultra_high": types.PartMediaResolutionLevel.MEDIA_RESOLUTION_ULTRA_HIGH,
    }

    parts = []
    for part in value:
        if not isinstance(part, dict):
            raise TypeError("budget preflight content parts must be dictionaries")
        kind = part.get("type")
        if kind == "text":
            parts.append(types.Part.from_text(text=str(part.get("text", ""))))
            continue
        if kind in {"video", "audio", "image", "document"}:
            uri = str(part.get("uri") or "")
            if not uri:
                raise ValueError(f"{kind} content has no URI to count")
            resolution = part.get("resolution")
            counted_resolution = None
            if resolution is not None:
                try:
                    counted_resolution = resolution_levels[str(resolution).lower()]
                except KeyError as error:
                    raise ValueError(
                        f"unsupported media resolution {resolution!r}"
                    ) from error
            parts.append(
                types.Part.from_uri(
                    file_uri=uri,
                    mime_type=part.get("mime_type"),
                    media_resolution=counted_resolution,
                )
            )
            continue
        raise TypeError(f"budget preflight cannot count content type {kind!r}")
    return types.Content(role="user", parts=parts)


def count_request_tokens(
    client: Any,
    *,
    model: str,
    input_value: Any,
    response_format: Any = None,
) -> int:
    """Count paid input before dispatch, with room for the output contract.

    countTokens measures the media and prompt exactly. The response schema is
    request metadata rather than ``contents``, so reserve a deliberately
    conservative two bytes per token for its serialized representation and a
    five-percent envelope around the provider count.
    """

    models = getattr(client, "models", None)
    counter = getattr(models, "count_tokens", None)
    if counter is None:
        # Test doubles do not carry the SDK's models surface. Production
        # clients always do; this fallback only lets pure unit tests exercise
        # reservation behaviour without making a network request.
        encoded = json.dumps(input_value, ensure_ascii=False, default=str)
        counted = max(1, math.ceil(len(encoded.encode("utf-8")) / 2))
    else:
        try:
            result = counter(model=model, contents=_count_contents(input_value))
        except Exception as error:
            raise RuntimeError(
                "Gemini token counting failed; the paid interaction was not sent"
            ) from error
        counted = int(getattr(result, "total_tokens", 0) or 0)
        if counted <= 0:
            raise RuntimeError(
                "Gemini token counting returned no total; the paid interaction "
                "was not sent"
            )

    schema_bytes = len(
        json.dumps(response_format, ensure_ascii=False, sort_keys=True).encode(
            "utf-8"
        )
    ) if response_format is not None else 0
    return math.ceil(counted * 1.05) + math.ceil(schema_bytes / 2)
