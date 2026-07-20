from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any

from .storage import read_json


GEMINI_35_FLASH_STANDARD_INPUT_USD_PER_MILLION = 1.50
GEMINI_35_FLASH_STANDARD_OUTPUT_USD_PER_MILLION = 9.00


def summarize_usage_files(paths: list[Path], *, relative_to: Path | None = None) -> dict[str, Any]:
    records: list[dict[str, Any]] = []
    total_input = 0
    total_output = 0
    total_thought = 0
    input_by_modality: dict[str, int] = defaultdict(int)
    for path in sorted(paths):
        payload = read_json(path)
        usage = payload.get("usage") or {}
        if not usage:
            continue
        input_tokens = int(usage.get("total_input_tokens") or 0)
        output_tokens = int(usage.get("total_output_tokens") or 0)
        thought_tokens = int(usage.get("total_thought_tokens") or 0)
        total_input += input_tokens
        total_output += output_tokens
        total_thought += thought_tokens
        modalities: dict[str, int] = {}
        for item in usage.get("input_tokens_by_modality") or []:
            modality = str(item.get("modality") or "UNKNOWN")
            tokens = int(item.get("tokens") or 0)
            modalities[modality] = modalities.get(modality, 0) + tokens
            input_by_modality[modality] += tokens
        records.append(
            {
                "path": str(path.relative_to(relative_to)) if relative_to else str(path),
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "thought_tokens": thought_tokens,
                "input_tokens_by_modality": modalities,
            }
        )
    billed_output = total_output + total_thought
    input_cost = total_input / 1_000_000 * GEMINI_35_FLASH_STANDARD_INPUT_USD_PER_MILLION
    output_cost = (
        billed_output / 1_000_000 * GEMINI_35_FLASH_STANDARD_OUTPUT_USD_PER_MILLION
    )
    return {
        "model": "gemini-3.5-flash",
        "pricing_basis": "Standard paid-tier public list price; actual invoice may be free-tier or differ",
        "input_usd_per_million_tokens": GEMINI_35_FLASH_STANDARD_INPUT_USD_PER_MILLION,
        "output_including_thought_usd_per_million_tokens": (
            GEMINI_35_FLASH_STANDARD_OUTPUT_USD_PER_MILLION
        ),
        "request_count": len(records),
        "total_input_tokens": total_input,
        "total_output_tokens": total_output,
        "total_thought_tokens": total_thought,
        "billed_output_tokens": billed_output,
        "input_tokens_by_modality": dict(sorted(input_by_modality.items())),
        "estimated_input_cost_usd": round(input_cost, 8),
        "estimated_output_cost_usd": round(output_cost, 8),
        "estimated_total_cost_usd": round(input_cost + output_cost, 8),
        "requests": records,
    }


def summarize_usage_and_list_price(root: Path) -> dict[str, Any]:
    return summarize_usage_files(
        list(root.rglob("*.raw_interaction.json")), relative_to=root
    )
