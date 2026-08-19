"""The merged editorial-plan schema stays inside the shape the API accepts.

The whole risk of merging DIRECTION + SELECTION + RHYTHM into one call is the
grammar-complexity ceiling that a nested plan schema hit before. These probes
prove the merged schema only adds sibling scalars to the known-good selection
shape -- same nesting depth, modestly larger -- and that the incoherent
commitment machinery is gone while the new fields are present.
"""

from __future__ import annotations

import json

from montagewright import planner


def _depth(obj: object) -> int:
    if isinstance(obj, dict):
        return 1 + max([0] + [_depth(v) for v in obj.values()])
    if isinstance(obj, list):
        return 1 + max([0] + [_depth(v) for v in obj])
    return 0


def _both():
    kwargs = dict(
        min_shots=5, max_shots=10, commitment_ids=["c1"], action_ids=["a1"],
    )
    selection = planner._selection_schema(["C1:s00"], **kwargs)
    plan = planner._editorial_plan_schema(["C1:s00"], **kwargs)
    return selection, plan


def test_merged_schema_stays_under_the_selection_ceiling():
    selection, plan = _both()
    # No deeper than selection, which the API already serves.
    assert _depth(plan) <= _depth(selection), (_depth(plan), _depth(selection))
    # And only modestly larger: sibling scalars, not a second deep tree.
    assert len(json.dumps(plan)) < 1.6 * len(json.dumps(selection))


def test_merged_schema_drops_the_commitment_machinery():
    _, plan = _both()
    props = plan["properties"]
    for gone in ("target_shot_count", "covered", "uncovered", "candidate_options"):
        assert gone not in props, gone
    shot = plan["properties"]["shots"]["items"]["properties"]
    assert "tier" not in shot


def test_merged_schema_adds_fallback_music_and_optional_length():
    _, plan = _both()
    props = plan["properties"]
    shot = props["shots"]["items"]["properties"]
    # Fallback is a substitution field, and the music sync is folded in.
    assert "fallback_source" in shot
    for name in ("sync_to", "beats", "cut_on_beat"):
        assert name in shot, name
    # target_seconds exists but is optional -- omit it for free length.
    assert "target_seconds" in props
    assert "target_seconds" not in plan["required"]
    # The story-level scalars and music fields ride as siblings of shots.
    for sibling in (
        "reasoning", "material_assessment", "direction", "unusable",
        "music_under_speech", "music_suggestion",
        "music_from_seconds", "music_spans", "shots",
    ):
        assert sibling in props, sibling


def test_merged_prompt_carries_the_coverage_and_fallback_rules():
    text = planner._editorial_plan_prompt()
    # One decision, coverage as distinct shots, alternate as a fallback.
    assert "editorial plan" in text
    assert "覆蓋" in text
    assert "備胎" in text
    # The three briefs are all present.
    assert "定調" in text and "選鏡" in text and "節奏" in text
