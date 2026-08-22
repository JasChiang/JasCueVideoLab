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
    assert "commitment_id" not in shot


def test_merged_schema_adds_fallback_music_and_optional_length():
    _, plan = _both()
    props = plan["properties"]
    shot = props["shots"]["items"]["properties"]
    # Fallback is a substitution field, and the music sync is folded in.
    assert "fallback_span_id" in shot
    assert "enum" not in shot["fallback_span_id"]
    for name in ("sync_to", "beats", "cut_on_beat"):
        assert name in shot, name
    for name in (
        "story_point", "continuity_mode", "cut_motivation",
        "source_event_ref", "source_event_relation", "event_tolerance_frames",
    ):
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


def test_merged_audio_track_uses_local_id_audit_not_a_large_schema_enum():
    plan = planner._editorial_plan_schema(
        ["C1:s00"], audio_span_ids=["C1:t00-t02"], action_ids=["a1"],
    )
    audio = plan["properties"]["audio_assignments"]
    assert "audio_assignments" in plan["required"]
    assert "enum" not in audio["items"]["properties"]["audio_span_id"]
    shot = plan["properties"]["shots"]["items"]["properties"]
    assert "enum" not in shot["span_id"]
    assert "enum" not in shot["action_id"]


def test_merged_schema_forbids_sub_second_timestamps():
    import re

    selection, plan = _both()
    shot = plan["properties"]["shots"]["items"]["properties"]
    tightened = shot["start_offset_seconds"]["pattern"]
    # Gemini samples video at ~1 fps and cannot perceive sub-second time, so the
    # merged schema forbids the hallucinated decimal on every observed-time
    # field. A whole second passes; 0:02.5 is rejected.
    for field in ("start_offset_seconds", "seconds_needed"):
        pat = re.compile(shot[field]["pattern"])
        assert pat.fullmatch("0:02")
        assert pat.fullmatch("0:02.5") is None, field
    look_seconds = shot["looks"]["items"]["properties"]["seconds"]["pattern"]
    assert re.compile(look_seconds).fullmatch("0:01")
    assert re.compile(look_seconds).fullmatch("0:01.5") is None
    # The default selection schema is untouched -- it still tolerates a decimal,
    # so tightening is scoped to the merged copy only (default path byte-stable).
    sel_shot = selection["properties"]["shots"]["items"]["properties"]
    assert "(?:\\.\\d+)?" in sel_shot["start_offset_seconds"]["pattern"]
    assert "(?:\\.\\d+)?" not in tightened


def test_merged_prompt_carries_the_coverage_and_fallback_rules():
    text = planner._editorial_plan_prompt()
    # One decision, coverage as distinct shots, alternate as a fallback.
    assert "editorial plan" in text
    assert "覆蓋" in text
    assert "備胎" in text
    # The three briefs are all present.
    assert "定調" in text and "選鏡" in text and "節奏" in text
    # And the whole-second timestamp rule (1 fps -> no invented sub-second).
    assert "0:02.5" in text and "整秒" in text
    # It is one authored decision, not the old rhythm role claiming that
    # picture and order were already locked inside the same supposedly merged call.
    assert "畫面與順序已選好" not in text
    assert "continuity_mode" in text and "J-cut" in text


def test_event_catalog_publishes_measured_span_and_action_boundaries():
    from montagewright.planner import MaterialItem, editorial_event_catalog
    from montagewright.spans import Span

    item = MaterialItem(
        source_id="C1", duration_seconds=10.0, summary="phone folds",
        spans=(Span("C1:s00", "C1", 1.0, 8.0, "usable", "locked"),),
        action_windows=(("a01", 2.0, 5.0),),
    )
    refs, text = editorial_event_catalog([item])
    assert "span_start:C1:s00" in refs
    assert "action_complete:C1:a01" in refs
    assert "5.000s" not in text
    assert "action_complete:C1:a01" in text
def test_look_projection_preserves_simultaneous_grounded_identities():
    from montagewright.schema import looks_of

    looks = looks_of({"looks": [{
        "at": "Fold8",
        "entity_id": "sku.fold8",
        "co_visible_entity_ids": ["sku.flip8", "sku.ultra"],
        "seconds": 2.0,
        "framing": "thirds",
    }]})

    assert looks[0].co_visible_entity_ids == ["sku.flip8", "sku.ultra"]
