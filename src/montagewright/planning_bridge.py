"""Project existing edit facts into the local planning authority.

This module is intentionally the seam between today's planner payloads and
the provider-neutral :mod:`montagewright.planning_state` contract.  Keeping
the projection here prevents ``cli.py`` from becoming the schema, persistence
layer, and orchestration layer at the same time.
"""

from __future__ import annotations

import hashlib
import json
import os
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


def _sources_of(records: Sequence[MaterialSpanRecord], span_ids: set[str]) -> list[str]:
    return sorted({
        one.source_id for one in records if one.span_id in span_ids
    })


def _named(sources: Sequence[str], limit: int = 6) -> str:
    shown = ", ".join(sources[:limit])
    return shown + (f" and {len(sources) - limit} more" if len(sources) > limit else "")


def _revision_difference(
    stored: PlanningState, current: PlanningState
) -> list[str]:
    """Say what moved between two revisions, in operator terms.

    The check that calls this is right to refuse, but "conflicts with
    current material: <path>" is not something anyone can act on.  The
    common cause is a run that published revision zero from an incomplete
    card library -- so the useful answer is which sources appeared, and
    that the frozen revision is the thing to remove.
    """

    notes: list[str] = []
    was = {one.span_id for one in stored.spans}
    now = {one.span_id for one in current.spans}
    appeared = _sources_of(current.spans, now - was)
    vanished = _sources_of(stored.spans, was - now)
    if appeared:
        notes.append(
            f"{len(now - was)} spans on {len(appeared)} sources are new: "
            f"{_named(appeared)}"
        )
    if vanished:
        notes.append(
            f"{len(was - now)} spans on {len(vanished)} sources are gone: "
            f"{_named(vanished)}"
        )
    stored_by_id = {one.span_id: one for one in stored.spans}
    moved = sorted(
        one.span_id for one in current.spans
        if one.span_id in stored_by_id and (
            (one.in_seconds, one.out_seconds, one.eligibility, one.evidence)
            != (
                stored_by_id[one.span_id].in_seconds,
                stored_by_id[one.span_id].out_seconds,
                stored_by_id[one.span_id].eligibility,
                stored_by_id[one.span_id].evidence,
            )
        )
    )
    if moved:
        notes.append(f"{len(moved)} spans changed shape: {_named(moved)}")
    for field in (
        "story_obligations", "coverage_obligations",
        "music_cue_refs", "grounding_target_refs",
    ):
        before = tuple(getattr(stored, field))
        after = tuple(getattr(current, field))
        if before != after:
            notes.append(
                f"{field}: {len(before)} entries became {len(after)}"
            )
    return notes


def publish_planning_state(
    work: Path,
    state: PlanningState,
    *,
    request: Any,
    response: Any,
    validation: Any,
    stage: str = "edit",
    allow_incomplete_rollover: bool = False,
) -> Path:
    """Publish or verify an immutable revision during resume.

    A failed attempt may have published a selection revision before any
    executable downstream artifact existed.  If a later local contract
    repair produces a different selection at that same revision number, the
    old fact is still useful audit history but must not permanently poison
    the run.  Callers may opt into archiving that incomplete leaf and
    publishing its replacement.  Once Rhythm, crop paths, rendered segments,
    or a report exists, the revision remains strictly immutable.
    """

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
            if allow_incomplete_rollover:
                _archive_incomplete_revision(work, stage, destination, stored)
                return write_planning_revision(
                    work, stage, state,
                    request=request, response=response, validation=validation,
                )
            told = "\n".join(
                f"  {note}" for note in _revision_difference(stored, state)
            )
            raise RuntimeError(
                "stored planning revision no longer describes the current "
                f"material: {destination}\n{told}\n"
                "  the revision is frozen on purpose -- selections downstream "
                "name spans in it. If it was published by an incomplete run "
                "(one that stopped on budget, say), remove "
                f"{destination.parent} and run again; cards, proxies and "
                "transcripts are cached and are not paid for twice."
            )
        return destination
    return write_planning_revision(
        work, stage, state,
        request=request, response=response, validation=validation,
    )


def _archive_incomplete_revision(
    work: Path,
    stage: str,
    destination: Path,
    stored: PlanningState,
) -> Path:
    """Move an unconsumed conflicting leaf aside without deleting evidence."""

    work = Path(work)
    output = work.parent
    downstream = [
        work / "rhythm.json",
        work / "crops.json",
        output / "preview.mp4",
        output / "picture.mp4",
        output / "deliverable.mp4",
        output / "report.json",
    ]
    segments = output / "segments"
    consumed = [path for path in downstream if path.exists()]
    if segments.is_dir() and any(segments.iterdir()):
        consumed.append(segments)
    if consumed:
        named = ", ".join(str(path) for path in consumed[:4])
        raise RuntimeError(
            "stored planning revision no longer describes the current "
            f"material: {destination}\n"
            "  automatic rollover is unsafe because downstream artifacts "
            f"already consume it: {named}"
        )

    archive = (
        work / "planning" / "archive" / stage /
        f"rev-{stored.revision}-{stored.sha256()[:16]}"
    )
    archive.parent.mkdir(parents=True, exist_ok=True)
    if archive.exists():
        raise RuntimeError(
            "cannot archive conflicting planning revision because its audit "
            f"destination already exists: {archive}"
        )
    # Same-filesystem rename preserves the complete immutable directory and
    # makes rev-N available before the replacement is exclusively published.
    os.rename(destination, archive)
    return archive
