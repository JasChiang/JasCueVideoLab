from __future__ import annotations

from difflib import SequenceMatcher
from itertools import combinations
from pathlib import Path
from typing import Any

from .geometry import box_iou, center_distance
from .models import ContentMap, Event, GroundingCandidate, GroundingProposal, MatchStatus
from .storage import read_json, write_json


def label_similarity(left: str, right: str) -> float:
    return SequenceMatcher(None, left.casefold().strip(), right.casefold().strip()).ratio()


def _match_events(left: list[Event], right: list[Event]) -> list[tuple[Event, Event, float]]:
    candidates = sorted(
        (
            (label_similarity(a.label, b.label), a, b)
            for a in left
            for b in right
        ),
        key=lambda item: item[0],
        reverse=True,
    )
    matched_left: set[str] = set()
    matched_right: set[str] = set()
    matches: list[tuple[Event, Event, float]] = []
    for similarity, a, b in candidates:
        if a.event_id in matched_left or b.event_id in matched_right:
            continue
        matched_left.add(a.event_id)
        matched_right.add(b.event_id)
        matches.append((a, b, similarity))
    return matches


def _groundings(run_dir: Path) -> list[GroundingProposal]:
    proposals = []
    for path in run_dir.glob("events/*/groundings/*/grounding.json"):
        proposals.append(GroundingProposal.model_validate(read_json(path)))
    return proposals


def _schema_results(run_dir: Path) -> list[dict[str, Any]]:
    paths = [run_dir / "content_map.schema_validation.json"]
    paths.extend(run_dir.glob("events/*/groundings/*/grounding.schema_validation.json"))
    return [dict(path=str(path.relative_to(run_dir)), **read_json(path)) for path in paths if path.exists()]


def _grounding_label(content: ContentMap, proposal: GroundingProposal) -> str:
    entity = next((item for item in content.entities if item.entity_id == proposal.entity_id), None)
    parts = [entity.label if entity else proposal.entity_id]
    candidate = _single_matched_candidate(proposal)
    if candidate is not None:
        parts.append(candidate.label)
    return " / ".join(parts)


def _single_matched_candidate(
    proposal: GroundingProposal,
) -> GroundingCandidate | None:
    if proposal.match_status != MatchStatus.MATCHED or len(proposal.candidates) != 1:
        return None
    return proposal.candidates[0]


def _grounding_kind(content: ContentMap, proposal: GroundingProposal) -> str:
    entity = next((item for item in content.entities if item.entity_id == proposal.entity_id), None)
    return entity.kind.value if entity else "unknown"


def _match_groundings(
    left: list[GroundingProposal],
    right: list[GroundingProposal],
    left_content: ContentMap,
    right_content: ContentMap,
) -> list[tuple[GroundingProposal, GroundingProposal, float, str, str, str, str]]:
    candidates = sorted(
        (
            (
                label_similarity(_grounding_label(left_content, a), _grounding_label(right_content, b))
                * (1.0 if _grounding_kind(left_content, a) == _grounding_kind(right_content, b) else 0.5),
                a,
                b,
                _grounding_label(left_content, a),
                _grounding_label(right_content, b),
                _grounding_kind(left_content, a),
                _grounding_kind(right_content, b),
            )
            for a in left
            for b in right
        ),
        key=lambda item: item[0],
        reverse=True,
    )
    used_left: set[tuple[str, str]] = set()
    used_right: set[tuple[str, str]] = set()
    matches = []
    for similarity, a, b, left_label, right_label, left_kind, right_kind in candidates:
        if similarity < 0.5:
            continue
        left_key = (a.event_id, a.entity_id)
        right_key = (b.event_id, b.entity_id)
        if left_key in used_left or right_key in used_right:
            continue
        used_left.add(left_key)
        used_right.add(right_key)
        matches.append((a, b, similarity, left_label, right_label, left_kind, right_kind))
    return matches


def compare_runs(run_dirs: list[Path], output_path: Path, human_annotations: Path | None = None) -> dict[str, Any]:
    runs = []
    for run_dir in run_dirs:
        content_path = run_dir / "content_map.json"
        if not content_path.exists():
            runs.append({"run_id": run_dir.name, "valid_content_map": False, "run_dir": run_dir})
            continue
        runs.append(
            {
                "run_id": run_dir.name,
                "valid_content_map": True,
                "run_dir": run_dir,
                "content_map": ContentMap.model_validate(read_json(content_path)),
                "groundings": _groundings(run_dir),
                "schema_validation": _schema_results(run_dir),
            }
        )
    pairs = []
    valid_runs = [run for run in runs if run["valid_content_map"]]
    for left, right in combinations(valid_runs, 2):
        left_map: ContentMap = left["content_map"]
        right_map: ContentMap = right["content_map"]
        matches = _match_events(left_map.events, right_map.events)
        event_metrics = []
        bbox_metrics = []
        left_groundings: list[GroundingProposal] = left["groundings"]
        right_groundings: list[GroundingProposal] = right["groundings"]
        for left_event, right_event, similarity in matches:
            keyframe_delta = None
            if left_event.recommended_keyframe_ms is not None and right_event.recommended_keyframe_ms is not None:
                keyframe_delta = abs(left_event.recommended_keyframe_ms - right_event.recommended_keyframe_ms)
            event_metrics.append(
                {
                    "left_event_id": left_event.event_id,
                    "right_event_id": right_event.event_id,
                    "left_label": left_event.label,
                    "right_label": right_event.label,
                    "label_similarity": similarity,
                    "start_delta_ms": abs(left_event.start_ms - right_event.start_ms),
                    "end_delta_ms": abs(left_event.end_ms - right_event.end_ms),
                    "keyframe_delta_ms": keyframe_delta,
                }
            )
            event_left_groundings = [g for g in left_groundings if g.event_id == left_event.event_id]
            event_right_groundings = [g for g in right_groundings if g.event_id == right_event.event_id]
            for left_proposal, right_proposal, entity_similarity, left_entity_label, right_entity_label, left_entity_kind, right_entity_kind in _match_groundings(
                event_left_groundings,
                event_right_groundings,
                left_map,
                right_map,
            ):
                left_candidate = _single_matched_candidate(left_proposal)
                right_candidate = _single_matched_candidate(right_proposal)
                if left_candidate is None or right_candidate is None:
                    bbox_metrics.append(
                        {
                            "left_entity_id": left_proposal.entity_id,
                            "right_entity_id": right_proposal.entity_id,
                            "left_entity_label": left_entity_label,
                            "right_entity_label": right_entity_label,
                            "left_entity_kind": left_entity_kind,
                            "right_entity_kind": right_entity_kind,
                            "entity_label_similarity": entity_similarity,
                            "left_event_id": left_event.event_id,
                            "right_event_id": right_event.event_id,
                            "comparable": False,
                            "reason": (
                                "one or both proposals are not a unique semantic match; "
                                "ambiguous candidates require human selection"
                            ),
                        }
                    )
                    continue
                left_box = left_candidate.box_2d
                right_box = right_candidate.box_2d
                bbox_metrics.append(
                    {
                        "left_entity_id": left_proposal.entity_id,
                        "right_entity_id": right_proposal.entity_id,
                        "left_entity_label": left_entity_label,
                        "right_entity_label": right_entity_label,
                        "left_entity_kind": left_entity_kind,
                        "right_entity_kind": right_entity_kind,
                        "entity_label_similarity": entity_similarity,
                        "left_event_id": left_event.event_id,
                        "right_event_id": right_event.event_id,
                        "comparable": True,
                        "frame_time_delta_ms": abs(
                            left_proposal.frame_time_ms - right_proposal.frame_time_ms
                        ),
                        "center_distance_normalized": center_distance(left_box, right_box),
                        "iou": box_iou(left_box, right_box),
                    }
                )
        pairs.append(
            {
                "left_run": left["run_id"],
                "right_run": right["run_id"],
                "event_count_left": len(left_map.events),
                "event_count_right": len(right_map.events),
                "event_count_difference": abs(len(left_map.events) - len(right_map.events)),
                "event_metrics": event_metrics,
                "bbox_metrics": bbox_metrics,
            }
        )
    report: dict[str, Any] = {
        "runs": [
            {
                "run_id": run["run_id"],
                "valid_content_map": run["valid_content_map"],
                "schema_validation": run.get("schema_validation", []),
            }
            for run in runs
        ],
        "pairwise": pairs,
    }
    if human_annotations:
        report["human_annotation_comparison"] = compare_human(valid_runs, human_annotations)
    write_json(output_path, report)
    return report


def compare_human(runs: list[dict[str, Any]], annotation_path: Path) -> list[dict[str, Any]]:
    annotations = read_json(annotation_path)
    reference_events = annotations.get("events", [])
    output = []
    for run in runs:
        content: ContentMap = run["content_map"]
        entities = {entity.entity_id: entity for entity in content.entities}
        event_candidates = sorted(
            (
                (label_similarity(reference["label"], event.label), index, event)
                for index, reference in enumerate(reference_events)
                for event in content.events
            ),
            key=lambda item: item[0],
            reverse=True,
        )
        assigned: dict[int, tuple[Event, float]] = {}
        used_event_ids: set[str] = set()
        for similarity, index, event in event_candidates:
            if index in assigned or event.event_id in used_event_ids:
                continue
            assigned[index] = (event, similarity)
            used_event_ids.add(event.event_id)
        matches = []
        for index, reference in enumerate(reference_events):
            if index not in assigned:
                matches.append({"reference_label": reference["label"], "matched": False})
                continue
            predicted, similarity = assigned[index]
            matches.append(
                {
                    "reference_label": reference["label"],
                    "matched": True,
                    "predicted_event_id": predicted.event_id,
                    "predicted_label": predicted.label,
                    "label_similarity": similarity,
                    "start_delta_ms": abs(reference["start_ms"] - predicted.start_ms),
                    "end_delta_ms": abs(reference["end_ms"] - predicted.end_ms),
                }
            )
        box_matches = []
        for reference in annotations.get("boxes", []):
            proposals: list[GroundingProposal] = run.get("groundings", [])
            visible = [
                proposal
                for proposal in proposals
                if _single_matched_candidate(proposal) is not None
            ]
            if not visible:
                box_matches.append(
                    {
                        "reference_entity_label": reference["entity_label"],
                        "comparable": False,
                        "reason": "run has no visible grounding candidates",
                    }
                )
                continue
            def proposal_similarity(proposal: GroundingProposal) -> float:
                entity_label = entities.get(proposal.entity_id).label if proposal.entity_id in entities else proposal.entity_id
                candidate = _single_matched_candidate(proposal)
                assert candidate is not None
                candidate_label = candidate.label
                return max(
                    label_similarity(reference["entity_label"], entity_label),
                    label_similarity(reference["entity_label"], candidate_label),
                )

            predicted = max(
                visible,
                key=lambda proposal: (
                    proposal_similarity(proposal),
                    -abs(reference["frame_time_ms"] - proposal.frame_time_ms),
                ),
            )
            predicted_candidate = _single_matched_candidate(predicted)
            assert predicted_candidate is not None
            predicted_box = predicted_candidate.box_2d
            reference_box = reference["box_2d"]
            similarity = proposal_similarity(predicted)
            frame_delta = abs(reference["frame_time_ms"] - predicted.frame_time_ms)
            comparable = similarity >= 0.75 and frame_delta <= 250
            if not comparable:
                reasons = []
                if similarity < 0.75:
                    reasons.append("entity label similarity below 0.75")
                if frame_delta > 250:
                    reasons.append("frame time delta exceeds 250 ms")
                box_matches.append(
                    {
                        "reference_entity_label": reference["entity_label"],
                        "predicted_entity_id": predicted.entity_id,
                        "label_similarity": similarity,
                        "reference_frame_time_ms": reference["frame_time_ms"],
                        "predicted_frame_time_ms": predicted.frame_time_ms,
                        "frame_time_delta_ms": frame_delta,
                        "comparable": False,
                        "reason": "; ".join(reasons),
                    }
                )
                continue
            box_matches.append(
                {
                    "reference_entity_label": reference["entity_label"],
                    "predicted_entity_id": predicted.entity_id,
                    "label_similarity": similarity,
                    "reference_frame_time_ms": reference["frame_time_ms"],
                    "predicted_frame_time_ms": predicted.frame_time_ms,
                    "frame_time_delta_ms": frame_delta,
                    "comparable": True,
                    "center_distance_normalized": center_distance(reference_box, predicted_box),
                    "iou": box_iou(reference_box, predicted_box),
                }
            )
        output.append(
            {"run_id": run["run_id"], "event_matches": matches, "bbox_matches": box_matches}
        )
    return output
