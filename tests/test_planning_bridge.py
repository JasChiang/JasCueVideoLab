from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pytest

from montagewright.planning_bridge import (
    material_planning_state,
    publish_planning_state,
    selection_planning_state,
)
from montagewright.spans import Span


@dataclass
class Material:
    source_id: str
    duration_seconds: float
    spans: tuple[Span, ...]


def test_inventory_keeps_unknown_motion_deferred_instead_of_erasing_it():
    material = [Material(
        "C1", 8.0,
        (Span("C1:s00", "C1", 0.0, 3.0, "authored", "usable reveal"),),
    )]
    cards = {"C1": {"segments": [
        {"from": "0:00", "to": "0:03", "status": "eligible",
         "motion_role": "authored", "why": "usable reveal"},
        {"from": "0:03", "to": "0:07", "status": "eligible",
         "motion_role": "unknown", "why": "needs visual verification"},
    ]}}

    state = material_planning_state(material, cards)

    assert [one.span_id for one in state.spans] == ["C1:s00", "C1:s01"]
    assert state.spans[0].disposition == "available"
    assert state.spans[1].eligibility == "unknown"
    assert state.spans[1].disposition == "deferred"


def test_inventory_rejects_card_invalid_and_camera_reset_segments():
    material = [Material("C1", 8.0, ())]
    cards = {"C1": {"segments": [
        {"from": "0:00", "to": "0:03", "status": "rejected",
         "motion_role": "locked", "why": "bad picture"},
        {"from": "0:03", "to": "0:07", "status": "eligible",
         "motion_role": "setup_reframe", "why": "camera reset"},
    ]}}
    state = material_planning_state(material, cards)
    assert [one.eligibility for one in state.spans] == [
        "hard_invalid", "hard_invalid"
    ]
    assert [one.disposition for one in state.spans] == ["rejected", "rejected"]


def test_selection_is_a_checked_revision_and_never_deletes_the_pool():
    material = [Material(
        "C1", 8.0,
        (
            Span("C1:s00", "C1", 0.0, 3.0, "authored", "one"),
            Span("C1:s01", "C1", 3.0, 7.0, "locked", "two"),
        ),
    )]
    base = material_planning_state(material, {})
    selected = selection_planning_state(base, [{"span_id": "C1:s01"}])

    assert len(selected.spans) == len(base.spans) == 2
    assert selected.selected_span_ids == ("C1:s01",)
    assert [one.disposition for one in selected.spans] == [
        "deferred", "selected"
    ]


def test_selection_cannot_name_a_span_missing_from_material_authority():
    base = material_planning_state(
        [Material("C1", 4.0, (Span("C1:s00", "C1", 0.0, 4.0),))], {}
    )
    with pytest.raises(ValueError, match="unknown IDs"):
        selection_planning_state(base, [{"span_id": "invented"}])


def test_publish_resume_accepts_identical_state_and_rejects_split_truth(
    tmp_path: Path,
):
    base = material_planning_state(
        [Material("C1", 4.0, (Span("C1:s00", "C1", 0.0, 4.0),))], {}
    )
    first = publish_planning_state(
        tmp_path, base, request={"stage": "material"}, response={},
        validation={"valid": True},
    )
    assert publish_planning_state(
        tmp_path, base, request={"ignored": "on resume"}, response={},
        validation={"valid": True},
    ) == first

    changed = material_planning_state(
        [Material("C2", 4.0, (Span("C2:s00", "C2", 0.0, 4.0),))], {}
    )
    with pytest.raises(RuntimeError, match="conflicts"):
        publish_planning_state(
            tmp_path, changed, request={}, response={}, validation={}
        )
