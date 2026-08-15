from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pytest

from montagewright.planning_bridge import (
    material_planning_state,
    publish_planning_state,
    selection_planning_state,
)
from montagewright.planning_state import PlanningState
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
    with pytest.raises(RuntimeError, match="no longer describes"):
        publish_planning_state(
            tmp_path, changed, request={}, response={}, validation={}
        )


def test_a_conflicting_revision_says_what_moved_and_how_to_recover(
    tmp_path: Path,
):
    """A path and the word "conflict" is not something anyone can act on.

    The case this is written from: a run described twenty-one of seventy-four
    rushes before the provider's credits ran out, froze that inventory as
    revision zero, and the next run -- with the full library -- could only
    say that the two disagreed.
    """

    incomplete = material_planning_state(
        [Material("C1", 4.0, (Span("C1:s00", "C1", 0.0, 4.0),))], {}
    )
    publish_planning_state(
        tmp_path, incomplete, request={}, response={}, validation={},
    )
    complete = material_planning_state(
        [
            Material("C1", 4.0, (Span("C1:s00", "C1", 0.0, 4.0),)),
            Material("C2", 4.0, (Span("C2:s00", "C2", 0.0, 4.0),)),
            Material("C3", 4.0, (Span("C3:s00", "C3", 0.0, 4.0),)),
        ],
        {},
    )

    with pytest.raises(RuntimeError) as fault:
        publish_planning_state(
            tmp_path, complete, request={}, response={}, validation={}
        )

    said = str(fault.value)
    assert "C2" in said and "C3" in said, "name the material that appeared"
    assert "C1" not in said.split("are new:")[1].split("\n")[0], (
        "the unchanged source is not what moved"
    )
    assert str(tmp_path / "planning" / "edit") in said, "say what to remove"


def test_incomplete_conflicting_leaf_is_archived_and_replaced(tmp_path: Path):
    base = material_planning_state(
        [Material("C1", 8.0, (
            Span("C1:s00", "C1", 0.0, 4.0),
            Span("C1:s01", "C1", 4.0, 8.0),
        ))],
        {},
    )
    publish_planning_state(
        tmp_path, base, request={}, response={}, validation={}
    )
    first = selection_planning_state(base, [{"span_id": "C1:s00"}])
    publish_planning_state(
        tmp_path, first, request={}, response={}, validation={}
    )
    replacement = selection_planning_state(base, [{"span_id": "C1:s01"}])

    current = publish_planning_state(
        tmp_path, replacement, request={}, response={}, validation={},
        allow_incomplete_rollover=True,
    )

    assert PlanningState.model_validate_json(
        (current / "state.json").read_text(encoding="utf-8")
    ).sha256() == replacement.sha256()
    archived = (
        tmp_path / "planning" / "archive" / "edit" /
        f"rev-1-{first.sha256()[:16]}" / "state.json"
    )
    assert PlanningState.model_validate_json(
        archived.read_text(encoding="utf-8")
    ).sha256() == first.sha256()


def test_conflicting_leaf_cannot_roll_over_after_downstream_consumes_it(
    tmp_path: Path,
):
    base = material_planning_state(
        [Material("C1", 8.0, (
            Span("C1:s00", "C1", 0.0, 4.0),
            Span("C1:s01", "C1", 4.0, 8.0),
        ))],
        {},
    )
    publish_planning_state(
        tmp_path, base, request={}, response={}, validation={}
    )
    first = selection_planning_state(base, [{"span_id": "C1:s00"}])
    publish_planning_state(
        tmp_path, first, request={}, response={}, validation={}
    )
    (tmp_path / "rhythm.json").write_text("{}", encoding="utf-8")
    replacement = selection_planning_state(base, [{"span_id": "C1:s01"}])

    with pytest.raises(RuntimeError, match="downstream artifacts already consume"):
        publish_planning_state(
            tmp_path, replacement, request={}, response={}, validation={},
            allow_incomplete_rollover=True,
        )
    assert (tmp_path / "planning" / "edit" / "rev-1").is_dir()
