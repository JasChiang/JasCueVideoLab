from __future__ import annotations

import itertools
import uuid
from pathlib import Path
from typing import Sequence

from .gemini import GeminiLabClient
from .geometry import box_iou, center_distance
from .models import ExtractedFrame, GroundingProposal, MediaInfo
from .overlay import draw_grounding_overlay
from .storage import append_error, read_json, write_json


def run_repeated_grounding(
    *,
    artifact_root: Path,
    frame_json: Path,
    prompt_template: str,
    event_id: str,
    event_description: str,
    entity_id: str,
    target_description: str,
    output_dir: Path,
    runs: int,
    temperature: float,
    reference_box: Sequence[int] | None = None,
) -> dict[str, object]:
    if runs < 1:
        raise ValueError("runs must be at least 1")
    media = MediaInfo.model_validate(read_json(artifact_root / "media.json"))
    frame = ExtractedFrame.model_validate(read_json(frame_json))
    output_dir.mkdir(parents=True, exist_ok=True)
    proposals: list[tuple[str, GroundingProposal]] = []
    rows: list[dict[str, object]] = []
    failures = 0
    client = GeminiLabClient(temperature=temperature)
    try:
        for number in range(1, runs + 1):
            label = f"repeat-{number:02d}"
            run_id = f"{label}-{uuid.uuid4().hex[:8]}"
            run_dir = output_dir / label
            try:
                proposal = client.ground_frame(
                    media=media,
                    frame=frame,
                    event_id=event_id,
                    event_description=event_description,
                    entity_id=entity_id,
                    target_description=target_description,
                    prompt_template=prompt_template,
                    run_id=run_id,
                    output_dir=run_dir,
                )
                draw_grounding_overlay(Path(frame.path), proposal, run_dir / "debug.png")
                proposals.append((label, proposal))
                candidate = proposal.candidates[0] if proposal.candidates else None
                row: dict[str, object] = {
                    "run": label,
                    "schema_valid": True,
                    "visible": proposal.visible,
                    "candidate_count": len(proposal.candidates),
                    "box_2d": list(candidate.box_2d) if candidate else None,
                    "confidence": candidate.confidence if candidate else None,
                }
                if candidate and reference_box is not None:
                    row["reference_iou"] = box_iou(candidate.box_2d, reference_box)
                    row["reference_center_distance"] = center_distance(
                        candidate.box_2d, reference_box
                    )
                rows.append(row)
            except Exception as error:
                failures += 1
                append_error(output_dir, f"{label}:grounding", error)
                rows.append(
                    {
                        "run": label,
                        "schema_valid": False,
                        "error_type": type(error).__name__,
                        "error": str(error),
                    }
                )
    finally:
        client.close()

    pairwise: list[dict[str, object]] = []
    for (left_label, left), (right_label, right) in itertools.combinations(proposals, 2):
        left_candidate = left.candidates[0] if left.candidates else None
        right_candidate = right.candidates[0] if right.candidates else None
        comparable = left_candidate is not None and right_candidate is not None
        pairwise.append(
            {
                "left": left_label,
                "right": right_label,
                "comparable": comparable,
                "iou": (
                    box_iou(left_candidate.box_2d, right_candidate.box_2d)
                    if comparable
                    else None
                ),
                "center_distance": (
                    center_distance(left_candidate.box_2d, right_candidate.box_2d)
                    if comparable
                    else None
                ),
            }
        )

    summary: dict[str, object] = {
        "model": "gemini-3.5-flash",
        "frame_hash": frame.frame_hash,
        "frame_pts": frame.frame_pts,
        "frame_time_ms": frame.frame_time_ms,
        "event_id": event_id,
        "entity_id": entity_id,
        "target_description": target_description,
        "runs_requested": runs,
        "runs_succeeded": len(proposals),
        "failure_count": failures,
        "reference_box": list(reference_box) if reference_box is not None else None,
        "runs": rows,
        "pairwise": pairwise,
    }
    write_json(output_dir / "summary.json", summary)
    return summary
