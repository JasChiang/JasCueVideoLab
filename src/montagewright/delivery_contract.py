"""Audit an editorial timeline against the producer's work order.

These are release obligations, not creative goals.  A logo minimum, embargo
window or required product co-occurrence is checked against the actual ordered
shots; persuasive prose in ``why`` can never satisfy it.
"""

from __future__ import annotations

from typing import Any

from montagewright.job import MusicPolicy, TimelineObligation


def _seconds(value: object) -> float:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return max(0.0, float(value))
    text = str(value or "0:00")
    try:
        minutes, seconds = text.split(":", 1)
        return max(0.0, int(minutes) * 60 + float(seconds))
    except (TypeError, ValueError):
        return 0.0


def _look_entities(shot: dict[str, Any]) -> list[set[str]]:
    states: list[set[str]] = []
    for look in shot.get("looks") or []:
        visible = {
            str(value) for value in look.get("co_visible_entity_ids") or []
            if value not in {None, "", "none"}
        }
        if look.get("entity_id") not in {None, "", "none"}:
            visible.add(str(look["entity_id"]))
        states.append(visible)
    return states


def _window(obligation: TimelineObligation, total: float) -> tuple[float, float]:
    window = obligation.window
    if window.final_seconds is not None:
        return max(0.0, total - window.final_seconds), total
    return (
        float(window.start_seconds or 0.0),
        min(total, float(window.end_seconds if window.end_seconds is not None else total)),
    )


def editorial_obligation_faults(
    shots: list[dict[str, Any]], obligations: tuple[TimelineObligation, ...],
) -> list[str]:
    """Return deterministic picture-track disagreements in timeline order."""

    timeline: list[tuple[float, float, list[set[str]]]] = []
    cursor = 0.0
    for shot in shots:
        duration = _seconds(shot.get("seconds_needed"))
        timeline.append((cursor, cursor + duration, _look_entities(shot)))
        cursor += duration
    faults: list[str] = []
    for obligation in obligations:
        if obligation.track != "picture":
            continue
        begins, ends = _window(obligation, cursor)
        if ends <= begins:
            faults.append(f"{obligation.obligation_id}: window is outside the cut")
            continue
        refs = set(obligation.refs)
        matched = 0.0
        for shot_start, shot_end, visible_states in timeline:
            overlap = max(0.0, min(shot_end, ends) - max(shot_start, begins))
            if overlap <= 0:
                continue
            entities = set().union(*visible_states) if visible_states else set()
            condition = bool(entities & refs)
            if obligation.kind == "required_cooccurrence":
                condition = any(refs <= state for state in visible_states)
            elif obligation.kind == "exclusive_presence":
                condition = any(
                    bool(state & refs) and not (state - refs)
                    for state in visible_states
                )
            if condition:
                matched += overlap
        if obligation.kind == "forbidden_presence":
            if matched > 1e-6:
                faults.append(
                    f"{obligation.obligation_id}: forbidden refs appear for "
                    f"{matched:.3f}s inside the protected window"
                )
        elif obligation.kind in {
            "required_presence", "required_cooccurrence", "exclusive_presence"
        }:
            needed = max(
                0.0,
                obligation.minimum_seconds
                if obligation.kind == "required_cooccurrence" else 0.0,
            )
            if matched <= 1e-6 or matched + 1e-6 < needed:
                faults.append(
                    f"{obligation.obligation_id}: required refs satisfy "
                    f"{matched:.3f}s in the protected window; needs "
                    f"{needed:.3f}s"
                )
        elif obligation.kind == "minimum_presence" and (
            matched + 1e-6 < obligation.minimum_seconds
        ):
            faults.append(
                f"{obligation.obligation_id}: {matched:.3f}s present, needs "
                f"{obligation.minimum_seconds:.3f}s"
            )
    return faults


def music_policy_faults(
    plan: dict[str, Any], policy: MusicPolicy,
) -> list[str]:
    if not policy.allowed_ranges:
        return []
    used: list[tuple[float, float]] = []
    for raw in plan.get("music_spans") or []:
        if isinstance(raw, (list, tuple)) and len(raw) == 2:
            used.append((_seconds(raw[0]), _seconds(raw[1])))
    if not used:
        start = _seconds(plan.get("music_from_seconds"))
        duration = sum(_seconds(one.get("seconds_needed")) for one in plan.get("shots") or [])
        if duration > 0:
            used.append((start, start + duration))
    allowed = [
        (one.start_seconds, one.end_seconds) for one in policy.allowed_ranges
    ]
    return [
        f"music span {start:.3f}-{end:.3f}s falls outside every licensed range"
        for start, end in used
        if not any(start >= left - 1e-6 and end <= right + 1e-6 for left, right in allowed)
    ]


def graphic_obligation_faults(
    plan: Any | None, obligations: tuple[TimelineObligation, ...], *, total: float,
) -> list[str]:
    """Check approved, resolved title-card windows against read-time promises."""

    graphic_rules = [one for one in obligations if one.track == "graphic"]
    if not graphic_rules:
        return []
    cues = list(getattr(plan, "cues", ()) or ()) if plan is not None else []
    facts = {
        fact.fact_id: fact for fact in (getattr(plan, "facts", ()) or ())
    } if plan is not None else {}
    faults: list[str] = []
    for obligation in graphic_rules:
        begins, ends = _window(obligation, total)
        matched = 0.0
        for cue in cues:
            fact = facts.get(cue.primary_fact_id)
            names = {
                cue.graphic_id, cue.kind, cue.primary_fact_id,
                str(getattr(fact, "exact_text", "")),
            }
            if not names.intersection(obligation.refs) or cue.status != "approved":
                continue
            matched += max(
                0.0,
                min(cue.at_seconds + cue.duration_seconds, ends)
                - max(cue.at_seconds, begins),
            )
        if obligation.kind == "forbidden_presence" and matched > 1e-6:
            faults.append(
                f"{obligation.obligation_id}: forbidden graphic appears for "
                f"{matched:.3f}s"
            )
        elif obligation.kind in {"required_presence", "minimum_read"}:
            required = (
                obligation.minimum_seconds
                if obligation.kind == "minimum_read" else 1e-6
            )
            if matched + 1e-6 < required:
                faults.append(
                    f"{obligation.obligation_id}: approved graphic reads for "
                    f"{matched:.3f}s, needs {required:.3f}s"
                )
    return faults
