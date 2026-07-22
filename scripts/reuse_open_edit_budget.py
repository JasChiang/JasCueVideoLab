#!/usr/bin/env python3
"""Apply a reviewed budget sequence to newly rendered segment variants, with no AI call."""

from __future__ import annotations

import argparse
from copy import deepcopy
from pathlib import Path
from typing import Any

from jascue_video_lab.feature_cut import _output_media_metadata
from jascue_video_lab.media import sha256_file
from jascue_video_lab.storage import read_json, utc_now, write_json
try:
    from scripts.reconcile_open_edit_budget import BudgetPlan, concat_segments
except ModuleNotFoundError:  # Direct `python scripts/...py` invocation.
    from reconcile_open_edit_budget import BudgetPlan, concat_segments


def resolve_budget_segments(
    budget: BudgetPlan,
    manifest: dict[str, object],
) -> tuple[dict[str, list[Path]], float]:
    if budget.project_id != manifest.get("project_id"):
        raise ValueError("budget plan and render manifest project IDs differ")
    horizontal_section = manifest.get("horizontal")
    vertical_section = manifest.get("vertical")
    if not isinstance(horizontal_section, dict) or not isinstance(vertical_section, dict):
        raise ValueError("render manifest must contain both aspect sections")
    horizontal = {
        item["feature_id"]: item for item in horizontal_section.get("chapters", [])
    }
    vertical = {item["feature_id"]: item for item in vertical_section.get("chapters", [])}
    if set(horizontal) != set(vertical):
        raise ValueError("rendered aspect timelines differ")
    decision_ids = {decision.feature_id for decision in budget.decisions}
    if decision_ids != set(horizontal):
        raise ValueError("budget decisions do not cover the rendered timeline")
    missing = sorted(set(budget.sequence) - set(horizontal))
    if missing:
        raise ValueError(f"budget sequence references missing segments: {missing}")
    durations: list[float] = []
    paths: dict[str, list[Path]] = {"16x9": [], "9x16": []}
    for feature_id in budget.sequence:
        h = horizontal[feature_id]
        v = vertical[feature_id]
        h_duration = _chapter_duration_ms(h, feature_id=feature_id)
        v_duration = _chapter_duration_ms(v, feature_id=feature_id)
        if h_duration != v_duration:
            raise ValueError(f"aspect durations differ for {feature_id}")
        durations.append(h_duration / 1000)
        paths["16x9"].append(Path(str(h["segment_path"])))
        paths["9x16"].append(Path(str(v["segment_path"])))
    total = sum(durations)
    if not budget.target_min_seconds <= total <= budget.target_max_seconds:
        raise ValueError(f"reused budget duration is out of range: {total:.3f}s")
    return paths, total


def _chapter_duration_ms(item: dict[str, Any], *, feature_id: str) -> int:
    duration = item.get("duration_ms")
    if isinstance(duration, int) and duration > 0:
        return duration
    source_in = item.get("source_in_ms")
    source_out = item.get("source_out_ms")
    if isinstance(source_in, int) and isinstance(source_out, int) and source_out > source_in:
        return source_out - source_in
    raise ValueError(f"rendered segment has no valid duration: {feature_id}")


def build_budgeted_manifest(
    budget: BudgetPlan,
    manifest: dict[str, Any],
    outputs: dict[str, str],
    output_media: dict[str, dict[str, Any]],
    *,
    source_budget_plan_path: Path,
    source_manifest_path: Path,
) -> dict[str, Any]:
    """Return a manifest whose chapter order exactly matches the reused edit."""

    result = deepcopy(manifest)
    for section_name, aspect in (("horizontal", "16x9"), ("vertical", "9x16")):
        section = result.get(section_name)
        if not isinstance(section, dict):
            raise ValueError(f"render manifest is missing {section_name}")
        chapter_by_id = {
            str(item["feature_id"]): item for item in section.get("chapters", [])
        }
        section["chapters"] = [
            deepcopy(chapter_by_id[feature_id]) for feature_id in budget.sequence
        ]
        section["output_path"] = outputs[aspect]
        section["media"] = deepcopy(output_media[aspect])
    result["budget_reuse"] = {
        "interpretation": "deterministic_sequence_reuse_no_model_call",
        "source_budget_plan": str(source_budget_plan_path),
        "source_budget_plan_sha256": sha256_file(source_budget_plan_path),
        "source_render_manifest": str(source_manifest_path),
        "source_render_manifest_sha256": sha256_file(source_manifest_path),
        "sequence": list(budget.sequence),
    }
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("budget_plan_json", type=Path)
    parser.add_argument("render_manifest_json", type=Path)
    parser.add_argument("output_dir", type=Path)
    args = parser.parse_args()

    budget_path = args.budget_plan_json.expanduser().resolve()
    manifest_path = args.render_manifest_json.expanduser().resolve()
    budget = BudgetPlan.model_validate(read_json(budget_path))
    manifest = read_json(manifest_path)
    paths, total = resolve_budget_segments(budget, manifest)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    outputs: dict[str, str] = {}
    for aspect, segments in paths.items():
        output = args.output_dir / f"open-edit-budgeted-{aspect}.mp4"
        concat_segments(segments, output)
        outputs[aspect] = str(output.resolve())
    output_media = {
        aspect: _output_media_metadata(Path(output_path))
        for aspect, output_path in outputs.items()
    }
    budgeted_manifest_path = args.output_dir / "render-manifest.budgeted.json"
    write_json(
        budgeted_manifest_path,
        build_budgeted_manifest(
            budget,
            manifest,
            outputs,
            output_media,
            source_budget_plan_path=budget_path,
            source_manifest_path=manifest_path,
        ),
    )
    write_json(
        args.output_dir / "budget-plan.reused.json",
        {
            "interpretation": "deterministic_sequence_reuse_no_model_call",
            "source_budget_plan": str(budget_path),
            "source_budget_plan_sha256": sha256_file(budget_path),
            "render_manifest": str(manifest_path),
            "render_manifest_sha256": sha256_file(manifest_path),
            "sequence": budget.sequence,
            "duration_seconds": round(total, 3),
            "outputs": outputs,
            "budgeted_render_manifest": str(budgeted_manifest_path.resolve()),
            "generated_at": utc_now(),
        },
    )
    print((args.output_dir / "budget-plan.reused.json").resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
