#!/usr/bin/env python3
"""Replay cached one-frame identity seeds through SAM without any model API.

This deliberately accepts only adaptive confirmation records containing one
confirmed frame and no seed risk.  It maps that immutable frame back to the
source clip, verifies the source and frame lineage, tracks at most five seconds,
and applies Montagewright's strict single-seed continuity gate.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from montagewright.measure.media import sha256_file
from montagewright.measure.sam_tracking import track_bbox_sam21
from montagewright.reframe import observations_from_sam


def _json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _frame_index(identity_frames: Path) -> dict[str, tuple[str, Path]]:
    indexed: dict[str, tuple[str, Path]] = {}
    for path in sorted(identity_frames.glob("*/identity-seed-*.jpg")):
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        indexed[digest] = (path.parent.name, path)
    return indexed


def _source_for(source_dir: Path, source_id: str) -> Path:
    matches = [
        path for path in source_dir.iterdir()
        if path.is_file() and path.stem == source_id
        and path.suffix.lower() in {".mp4", ".mov", ".m4v"}
    ]
    if len(matches) != 1:
        raise RuntimeError(
            f"expected exactly one source for {source_id}, found {len(matches)}"
        )
    return matches[0]


def _bounded_interval(
    seed_ms: int,
    sighting_window: list[float],
    max_seconds: float,
) -> tuple[int, int]:
    sighting_start = max(0, round(float(sighting_window[0]) * 1000))
    sighting_end = round(float(sighting_window[1]) * 1000)
    width = max(1, round(max_seconds * 1000))
    start = max(sighting_start, seed_ms - width // 2)
    end = min(sighting_end, start + width)
    if end - start < width:
        start = max(sighting_start, end - width)
    if not start <= seed_ms < end:
        raise RuntimeError("seed is outside its cached sighting window")
    if end <= start:
        raise RuntimeError("cached sighting window is empty")
    return start, end


def _clean_cases(cache_dir: Path, identity_frames: Path) -> list[dict[str, Any]]:
    frames = _frame_index(identity_frames)
    cases: list[dict[str, Any]] = []
    for cache_path in sorted(cache_dir.glob("identity-*-adaptive-seed-v1.json")):
        payload = _json(cache_path)
        confirmed = payload.get("confirmed") or []
        if len(confirmed) != 1:
            continue
        seed = confirmed[0]
        if seed.get("seed_risk_flags"):
            continue
        mapped = frames.get(str(seed.get("frame_sha256") or ""))
        if mapped is None:
            continue
        source_id, frame_path = mapped
        cases.append({
            "source_id": source_id,
            "frame_path": frame_path,
            "cache_path": cache_path,
            "target": payload["target"],
            "video_sha256": payload["video_sha256"],
            "seed": seed,
        })
    return sorted(cases, key=lambda one: one["source_id"])


def _markdown(results: list[dict[str, Any]]) -> str:
    lines = [
        "# Cached single-seed SAM replay",
        "",
        "No Gemini/API client is constructed by this replay.",
        "",
        "| Source | Interval | Samples | Tracked | Result | Risks |",
        "|---|---:|---:|---:|---|---|",
    ]
    for one in results:
        risks = ", ".join(one["continuity_risks"]) or "—"
        lines.append(
            f"| {one['source_id']} | {one['start_ms']/1000:.3f}–"
            f"{one['end_ms']/1000:.3f}s | {one['total_samples']} | "
            f"{one['tracked_observations']} | {one['status']} | {risks} |"
        )
    lines.extend([
        "",
        "`accepted` means the cached Gemini box agreed with the SAM seed mask and "
        "every analysed sample passed the strict continuity gate.",
        "",
    ])
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--library", type=Path, required=True)
    parser.add_argument("--work", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--analysis-fps", type=float, default=4.0)
    parser.add_argument("--max-side", type=int, default=960)
    parser.add_argument("--max-seconds", type=float, default=5.0)
    parser.add_argument("--device", choices=("cpu", "mps", "cuda"), default="cpu")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--list-only", action="store_true")
    args = parser.parse_args()

    cache_dir = args.library / "reference-grounding"
    identity_frames = args.work / "identity-frames"
    cases = _clean_cases(cache_dir, identity_frames)
    if args.limit is not None:
        cases = cases[: args.limit]
    if not cases:
        raise RuntimeError("no clean one-frame adaptive seed records were found")
    print(json.dumps({
        "mode": "offline_cached_single_seed",
        "gemini_calls": 0,
        "cases": [one["source_id"] for one in cases],
    }, ensure_ascii=False), flush=True)
    if args.list_only:
        return 0

    args.output.mkdir(parents=True, exist_ok=True)
    if any(args.output.iterdir()):
        raise FileExistsError(f"output directory is not empty: {args.output}")

    results: list[dict[str, Any]] = []
    for index, case in enumerate(cases, start=1):
        source_id = case["source_id"]
        seed = case["seed"]
        source = _source_for(args.source_dir, source_id)
        if sha256_file(source) != case["video_sha256"]:
            raise RuntimeError(f"source hash changed for {source_id}")
        seed_ms = int(seed["frame_time_ms"])
        start_ms, end_ms = _bounded_interval(
            seed_ms, seed["sighting_window"], args.max_seconds
        )
        print(
            f"[{index}/{len(cases)}] {source_id} "
            f"{start_ms/1000:.3f}-{end_ms/1000:.3f}s seed={seed_ms/1000:.3f}s",
            flush=True,
        )
        native_box = [round(float(value) * 1000) for value in seed["box"]]
        track = track_bbox_sam21(
            video_path=source,
            checkpoint_path=args.checkpoint,
            seed_time_ms=seed_ms,
            seed_box_2d=native_box,
            target_description=case["target"],
            output_dir=args.output / source_id,
            seed_source="cached_gemini_exact_frame_single_seed",
            asset_id=seed["video_asset_id"],
            seed_frame_pts=int(seed["frame_pts"]),
            seed_frame_sha256=seed["frame_sha256"],
            seed_source_width=int(seed["width"]),
            seed_source_height=int(seed["height"]),
            analysis_fps=args.analysis_fps,
            max_side=args.max_side,
            device=args.device,
            allowed_start_ms=start_ms,
            allowed_end_ms=end_ms,
        )
        observations, states = observations_from_sam(
            track,
            clip_start_seconds=start_ms / 1000.0,
            semantic_anchors=((seed_ms / 1000.0, tuple(seed["box"])),),
            require_identity_validation=True,
        )
        risks = sorted(
            key.split(":", 1)[1]
            for key in states
            if key.startswith("_continuity_risk:")
        )
        accepted = bool(observations) and states.get("_identity_by_continuity") == 1
        result = {
            "source_id": source_id,
            "source": str(source.resolve()),
            "seed_cache": str(case["cache_path"].resolve()),
            "seed_frame": str(case["frame_path"].resolve()),
            "seed_ms": seed_ms,
            "start_ms": start_ms,
            "end_ms": end_ms,
            "analysis_fps": args.analysis_fps,
            "total_samples": track.total_samples,
            "tracked_observations": len(observations),
            "state_counts": track.state_counts,
            "validation_states": states,
            "continuity_risks": risks,
            "status": "accepted" if accepted else "rejected",
            "gemini_calls": 0,
            "debug_video": str(
                (args.output / source_id / "segmentation-debug.mp4").resolve()
            ),
        }
        results.append(result)
        print(
            f"    {result['status']}: {len(observations)}/{track.total_samples} "
            f"observations; risks={risks or ['none']}",
            flush=True,
        )

    report = {
        "contract_version": "offline-single-seed-sam-replay-v1",
        "gemini_calls": 0,
        "case_count": len(results),
        "accepted_count": sum(one["status"] == "accepted" for one in results),
        "rejected_count": sum(one["status"] == "rejected" for one in results),
        "results": results,
    }
    (args.output / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (args.output / "report.md").write_text(_markdown(results), encoding="utf-8")
    print(
        f"done: {report['accepted_count']}/{report['case_count']} accepted; "
        f"report={args.output / 'report.json'}",
        flush=True,
    )
    return 0 if report["rejected_count"] == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
