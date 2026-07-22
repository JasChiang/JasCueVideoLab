from __future__ import annotations

from collections import defaultdict
import hashlib
import json
from pathlib import Path
from typing import Any

from .storage import read_json


STANDARD_PRICING_USD_PER_MILLION: dict[str, dict[str, float]] = {
    "gemini-3.5-flash": {"input": 1.50, "output_including_thought": 9.00},
    "gemini-3.6-flash": {"input": 1.50, "output_including_thought": 7.50},
}


def summarize_usage_files(paths: list[Path], *, relative_to: Path | None = None) -> dict[str, Any]:
    records: list[dict[str, Any]] = []
    total_input = 0
    total_output = 0
    total_thought = 0
    input_by_modality: dict[str, int] = defaultdict(int)
    usage_by_model: dict[str, dict[str, int | float]] = {}
    seen_payloads: set[str] = set()
    duplicate_paths: list[str] = []
    for path in sorted(paths):
        payload = read_json(path)
        fingerprint = hashlib.sha256(
            json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
        ).hexdigest()
        if fingerprint in seen_payloads:
            duplicate_paths.append(
                str(path.relative_to(relative_to)) if relative_to else str(path)
            )
            continue
        seen_payloads.add(fingerprint)
        usage = payload.get("usage") or {}
        if not usage:
            continue
        model_id = str(payload.get("model") or "")
        if not model_id:
            raise ValueError(f"usage artifact has no model id: {path}")
        if model_id not in STANDARD_PRICING_USD_PER_MILLION:
            raise ValueError(f"no Standard pricing is registered for {model_id!r}: {path}")
        input_tokens = int(usage.get("total_input_tokens") or 0)
        output_tokens = int(usage.get("total_output_tokens") or 0)
        thought_tokens = int(usage.get("total_thought_tokens") or 0)
        total_input += input_tokens
        total_output += output_tokens
        total_thought += thought_tokens
        model_usage = usage_by_model.setdefault(
            model_id,
            {"request_count": 0, "input_tokens": 0, "output_tokens": 0, "thought_tokens": 0},
        )
        model_usage["request_count"] = int(model_usage["request_count"]) + 1
        model_usage["input_tokens"] = int(model_usage["input_tokens"]) + input_tokens
        model_usage["output_tokens"] = int(model_usage["output_tokens"]) + output_tokens
        model_usage["thought_tokens"] = int(model_usage["thought_tokens"]) + thought_tokens
        modalities: dict[str, int] = {}
        for item in usage.get("input_tokens_by_modality") or []:
            modality = str(item.get("modality") or "UNKNOWN")
            tokens = int(item.get("tokens") or 0)
            modalities[modality] = modalities.get(modality, 0) + tokens
            input_by_modality[modality] += tokens
        records.append(
            {
                "path": str(path.relative_to(relative_to)) if relative_to else str(path),
                "model": model_id,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "thought_tokens": thought_tokens,
                "input_tokens_by_modality": modalities,
            }
        )
    billed_output = total_output + total_thought
    input_cost = 0.0
    output_cost = 0.0
    model_breakdown: dict[str, dict[str, int | float]] = {}
    for model_id, model_usage in sorted(usage_by_model.items()):
        rates = STANDARD_PRICING_USD_PER_MILLION[model_id]
        model_input = int(model_usage["input_tokens"])
        model_output = int(model_usage["output_tokens"])
        model_thought = int(model_usage["thought_tokens"])
        model_billed_output = model_output + model_thought
        model_input_cost = model_input / 1_000_000 * rates["input"]
        model_output_cost = (
            model_billed_output / 1_000_000 * rates["output_including_thought"]
        )
        input_cost += model_input_cost
        output_cost += model_output_cost
        model_breakdown[model_id] = {
            **model_usage,
            "billed_output_tokens": model_billed_output,
            "input_usd_per_million_tokens": rates["input"],
            "output_including_thought_usd_per_million_tokens": rates[
                "output_including_thought"
            ],
            "estimated_input_cost_usd": round(model_input_cost, 8),
            "estimated_output_cost_usd": round(model_output_cost, 8),
            "estimated_total_cost_usd": round(model_input_cost + model_output_cost, 8),
        }
    models = sorted(model_breakdown)
    single_model_rates = (
        STANDARD_PRICING_USD_PER_MILLION[models[0]] if len(models) == 1 else None
    )
    return {
        "model": models[0] if len(models) == 1 else ("mixed" if models else None),
        "pricing_basis": "Standard paid-tier public list price; actual invoice may be free-tier or differ",
        "input_usd_per_million_tokens": (
            single_model_rates["input"] if single_model_rates else None
        ),
        "output_including_thought_usd_per_million_tokens": (
            single_model_rates["output_including_thought"] if single_model_rates else None
        ),
        "models": model_breakdown,
        "request_count": len(records),
        "total_input_tokens": total_input,
        "total_output_tokens": total_output,
        "total_thought_tokens": total_thought,
        "billed_output_tokens": billed_output,
        "input_tokens_by_modality": dict(sorted(input_by_modality.items())),
        "estimated_input_cost_usd": round(input_cost, 8),
        "estimated_output_cost_usd": round(output_cost, 8),
        "estimated_total_cost_usd": round(input_cost + output_cost, 8),
        "duplicate_artifact_count": len(duplicate_paths),
        "duplicate_artifact_paths": duplicate_paths,
        "requests": records,
    }


def summarize_usage_and_list_price(root: Path) -> dict[str, Any]:
    return summarize_usage_files(
        list(root.rglob("*.raw_interaction.json")), relative_to=root
    )
