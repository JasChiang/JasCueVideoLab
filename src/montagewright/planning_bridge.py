"""Project existing edit facts into the local planning authority.

This module is intentionally the seam between today's planner payloads and
the provider-neutral :mod:`montagewright.planning_state` contract.  Keeping
the projection here prevents ``cli.py`` from becoming the schema, persistence
layer, and orchestration layer at the same time.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from montagewright.planning_state import (
    PLANNING_DELTA_VERSION,
    PLANNING_STATE_VERSION,
    MaterialSpanRecord,
    PlanningState,
    PlanningStateDelta,
    apply_planning_delta,
    canonical_json,
    write_planning_revision,
)
from montagewright.spans import seconds_of


def _material_digest(records: Sequence[MaterialSpanRecord]) -> str:
    identity = [
        {
            "source_id": one.source_id,
            "span_id": one.span_id,
            "in_seconds": one.in_seconds,
            "out_seconds": one.out_seconds,
            "eligibility": one.eligibility,
            "evidence": list(one.evidence),
        }
        for one in records
    ]
    return hashlib.sha256(canonical_json(identity).encode("utf-8")).hexdigest()


def material_planning_state(
    material: Sequence[Any],
    cards: Mapping[str, dict[str, Any] | None],
    *,
    story_obligations: Iterable[str] = (),
    coverage_obligations: Iterable[str] = (),
    music_cue_refs: Iterable[str] = (),
    grounding_target_refs: Iterable[str] = (),
) -> PlanningState:
    """Build revision zero without silently losing uncertain card spans.

    ``spans_of`` intentionally refuses a segment whose motion role is
    unknown.  That is appropriate for immediate rendering, but it must not
    make the segment disappear from the planning inventory: later visual or
    grounding evidence may resolve it.  Such intervals are kept here as
    ``unknown/deferred`` while executable spans remain ``eligible/available``.
    """

    records: list[MaterialSpanRecord] = []
    for item in sorted(material, key=lambda one: one.source_id):
        executable = {one.span_id: one for one in item.spans}
        card = cards.get(item.source_id) or {}
        written = card.get("segments") or []
        seen: set[str] = set()
        for index, entry in enumerate(written):
            span_id = f"{item.source_id}:s{index:02d}"
            start = seconds_of(entry.get("from"))
            end = seconds_of(entry.get("to"))
            if start is None or end is None or end <= start:
                continue
            start = max(0.0, min(float(start), float(item.duration_seconds)))
            end = max(0.0, min(float(end), float(item.duration_seconds)))
            if end <= start:
                continue
            seen.add(span_id)
            is_executable = span_id in executable
            status = str(entry.get("status") or "unknown")
            motion = str(entry.get("motion_role") or "unknown")
            reason = str(entry.get("why") or "").strip() or None
            hard_invalid = status != "eligible" or motion in {
                "setup_reframe", "disturbance"
            }
            records.append(MaterialSpanRecord(
                source_id=item.source_id,
                span_id=span_id,
                in_seconds=round(start, 3),
                out_seconds=round(end, 3),
                eligibility=(
                    "eligible" if is_executable else
                    "hard_invalid" if hard_invalid else "unknown"
                ),
                disposition=(
                    "available" if is_executable else
                    "rejected" if hard_invalid else "deferred"
                ),
                evidence=(f"card_status:{status}", f"motion_role:{motion}"),
                reason=reason,
            ))
        # Legacy/no-segment cards expose one whole executable span.  Also
        # retain any executable span not represented by the written card so
        # the authority exactly covers what selection is allowed to name.
        for span_id, span in executable.items():
            if span_id in seen:
                continue
            records.append(MaterialSpanRecord(
                source_id=item.source_id,
                span_id=span_id,
                in_seconds=float(span.starts_seconds),
                out_seconds=float(span.ends_seconds),
                eligibility="eligible",
                disposition="available",
                evidence=("local_executable_span",),
            ))

    records.sort(key=lambda one: (one.source_id, one.in_seconds, one.span_id))
    return PlanningState(
        contract_version=PLANNING_STATE_VERSION,
        material_digest=_material_digest(records),
        revision=0,
        parent_sha256=None,
        spans=tuple(records),
        story_obligations=tuple(dict.fromkeys(str(x) for x in story_obligations if x)),
        coverage_obligations=tuple(
            dict.fromkeys(str(x) for x in coverage_obligations if x)
        ),
        music_cue_refs=tuple(dict.fromkeys(str(x) for x in music_cue_refs if x)),
        grounding_target_refs=tuple(
            dict.fromkeys(str(x) for x in grounding_target_refs if x)
        ),
    )


def selection_planning_state(
    base: PlanningState, shots: Sequence[dict[str, Any]]
) -> PlanningState:
    """Apply a selection as a checked CAS revision over known material."""

    selected = tuple(dict.fromkeys(
        str(shot.get("span_id") or "") for shot in shots if shot.get("span_id")
    ))
    chosen = set(selected)
    updates = []
    for span in base.spans:
        if span.eligibility == "hard_invalid":
            disposition = "rejected"
        elif span.span_id in chosen:
            disposition = "selected"
        else:
            disposition = "deferred"
        updates.append(span.model_copy(update={"disposition": disposition}))
    delta = PlanningStateDelta(
        contract_version=PLANNING_DELTA_VERSION,
        base_sha256=base.sha256(),
        span_updates=tuple(updates),
        selected_span_ids=selected,
        alternate_span_ids=(),
    )
    return apply_planning_delta(base, delta)


def publish_planning_state(
    work: Path,
    state: PlanningState,
    *,
    request: Any,
    response: Any,
    validation: Any,
    stage: str = "edit",
) -> Path:
    """Publish or verify an identical immutable revision during resume."""

    destination = Path(work) / "planning" / stage / f"rev-{state.revision}"
    if destination.exists():
        try:
            stored = PlanningState.model_validate_json(
                (destination / "state.json").read_text(encoding="utf-8")
            )
        except Exception as error:
            raise RuntimeError(
                f"stored planning revision is unreadable: {destination}"
            ) from error
        if stored.sha256() != state.sha256():
            raise RuntimeError(
                f"stored planning revision conflicts with current material: "
                f"{destination}"
            )
        return destination
    return write_planning_revision(
        work, stage, state,
        request=request, response=response, validation=validation,
    )
