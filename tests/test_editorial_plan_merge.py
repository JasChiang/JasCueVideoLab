"""The merged one-call editorial plan is consumed directly.

These tests never call a real model. They mock the call to prove the merge's
two structural promises -- ONE call with the footage attached ONCE, and a
flat plan reaches local execution without candidate commitments.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from montagewright import planner
from montagewright.planner import (
    MaterialItem,
    decide_editorial_plan,
)
from montagewright.spans import Span


def _executable_shot(
    span_id: str, *, subject: str, story_point: str,
) -> dict[str, object]:
    return {
        "span_id": span_id,
        "start_offset_seconds": "0:00",
        "action_id": "none",
        "action_treatment": "none",
        "camera_intent": "hold",
        "agrees_with_direction": True,
        "direction_disagreement_reason": "",
        "pacing_exception": False,
        "pacing_exception_reason": "",
        "looks": [{
            "at": subject, "seconds": "0:03", "framing": "centre",
            "presentation_intent": "centered_hold", "entity_id": "none",
            "must_be_whole": False,
        }],
        "energy": "medium",
        "seconds_needed": "0:03",
        "audio_role": "discard",
        "audio_completion": "none",
        "picture_role": "primary_action",
        "audio_reason": "",
        "why": story_point,
        "story_point": story_point,
        "continuity_mode": "associative_montage",
        "cut_motivation": "content",
        "source_event_ref": "none",
        "source_event_relation": "none",
        "event_tolerance_frames": 0,
    }


def _executable_plan(span_id: str, *, subject: str) -> dict[str, object]:
    return {
        "reasoning": "use the clearest product view",
        "material_assessment": "one clean usable take",
        "direction": "clean product launch",
        "target_seconds": "0:03",
        "music_under_speech": "duck",
        "audio_assignments": [],
        "shots": [_executable_shot(
            span_id, subject=subject, story_point="show the interface",
        )],
    }


def _material():
    return [
        MaterialItem(
            source_id="C1", duration_seconds=8.0, summary="a",
            spans=(Span("C1:s00", "C1", 0.0, 4.0, "wide", "locked"),),
        ),
        MaterialItem(
            source_id="C2", duration_seconds=8.0, summary="b",
            spans=(Span("C2:s00", "C2", 0.0, 4.0, "detail", "locked"),),
        ),
    ]


_PLAN = {
    "reasoning": "r", "material_assessment": "m", "direction": "d",
    "target_seconds": "1:00",
    "shots": [
        {"span_id": "C1:s00", "seconds_needed": "0:03",
         "camera_intent": "hold", "why": "establish",
         "looks": [{"at": "x", "entity_id": "device.fold"}],
         "fallback_span_id": "C2:s00"},
        {"span_id": "C2:s00", "seconds_needed": "0:03",
         "camera_intent": "reveal", "why": "detail", "looks": []},
    ],
    "music_from_seconds": 16.0, "music_spans": [],
    "unusable": [],
}


def test_decide_editorial_plan_makes_one_call_with_one_stringout(
    monkeypatch, tmp_path,
):
    calls = {"ask": 0, "request": None}

    def fake_ask(client, **request):
        calls["ask"] += 1
        calls["request"] = request
        return SimpleNamespace(
            status="completed", output_text=json.dumps(_PLAN),
            usage={"total_input_tokens": 100, "total_output_tokens": 20},
        )

    monkeypatch.setattr(planner, "ask", fake_ask)
    reel = tmp_path / "stringout.mp4"
    reel.write_bytes(b"offline-test")
    monkeypatch.setattr(
        planner, "upload_now",
        lambda *a, **k: SimpleNamespace(uri="files/stringout"),
    )

    plan, _ = decide_editorial_plan(
        _material(), brief="b", aspect="9:16", seconds=60.0,
        client=object(), planning_video=reel,
    )
    # Exactly one model call and exactly one video part, with the editorial
    # question after the media rather than measurements anchoring the viewing.
    assert calls["ask"] == 1
    parts = calls["request"]["input"]
    assert [one["type"] for one in parts].count("video") == 1
    assert parts[0]["type"] == "video"
    assert parts[-1]["type"] == "text"
    assert [s["span_id"] for s in plan["shots"]] == ["C1:s00", "C2:s00"]
    assert plan["target_seconds"] == 60.0
    assert plan["aspect"] == "9:16"


def test_merged_plan_repairs_picture_span_used_as_audio_id(monkeypatch, tmp_path):
    material = [MaterialItem(
        source_id="C1", duration_seconds=8.0, summary="interview",
        spans=(Span("C1:s00", "C1", 0.0, 4.0, "speaker", "locked"),),
        audio_spans=(("C1:t00", 0.0, 3.0),),
    )]
    broken = _executable_plan("C1:s00", subject="speaker")
    broken["audio_assignments"] = [{
        "audio_span_id": "C1:s00", "starts_at_shot_index": 0,
        "offset_seconds": "0:00", "completion": "complete_thought",
        "gain_db": 0.0, "why": "answer",
    }]
    repaired = json.loads(json.dumps(broken))
    repaired["audio_assignments"][0]["audio_span_id"] = "C1:t00"
    answers = iter((broken, repaired))
    requests = []

    def fake_ask(client, **request):
        requests.append(request)
        return SimpleNamespace(
            status="completed", output_text=json.dumps(next(answers)),
            usage={"total_input_tokens": 100, "total_output_tokens": 20},
        )

    monkeypatch.setattr(planner, "ask", fake_ask)
    monkeypatch.setattr(
        planner, "upload_now",
        lambda *a, **k: SimpleNamespace(uri="files/stringout"),
    )
    reel = tmp_path / "stringout.mp4"
    reel.write_bytes(b"offline-test")
    plan, usage = decide_editorial_plan(
        material, brief="keep the answer", aspect="9:16", seconds=3.0,
        client=object(), planning_video=reel, allow_paid_repair=True,
    )
    assert len(requests) == 2
    assert [one["type"] for one in requests[1]["input"]] == ["text"]
    assert plan["audio_assignments"][0]["audio_span_id"] == "C1:t00"
    assert usage.input_tokens == 200


def test_merged_plan_preserves_invalid_paid_draft_without_repair(
    monkeypatch, tmp_path,
):
    material = [MaterialItem(
        source_id="C1", duration_seconds=8.0, summary="interview",
        spans=(Span("C1:s00", "C1", 0.0, 4.0, "speaker", "locked"),),
        audio_spans=(("C1:t00", 0.0, 3.0),),
    )]
    broken = _executable_plan("C1:s00", subject="speaker")
    broken["audio_assignments"] = [{
        "audio_span_id": "C1:s00", "starts_at_shot_index": 0,
        "offset_seconds": "0:00", "completion": "complete_thought",
        "gain_db": 0.0, "why": "answer",
    }]
    monkeypatch.setattr(
        planner, "ask", lambda *a, **k: SimpleNamespace(
            status="completed", output_text=json.dumps(broken), usage={}
        ),
    )
    monkeypatch.setattr(
        planner, "upload_now",
        lambda *a, **k: SimpleNamespace(uri="files/stringout"),
    )
    reel = tmp_path / "stringout.mp4"
    reel.write_bytes(b"offline-test")
    with pytest.raises(planner.EditorialPlanUnrenderable) as stopped:
        decide_editorial_plan(
            material, brief="b", client=object(), planning_video=reel,
        )
    assert stopped.value.draft["audio_assignments"][0]["audio_span_id"] == "C1:s00"


def test_merged_speaker_audio_is_routed_only_once():
    chosen = _executable_plan("C1:s00", subject="speaker")
    chosen["shots"][0].update({
        "picture_role": "speaker", "action_id": "none",
        "audio_role": "sync_action", "audio_completion": "complete_thought",
    })
    chosen["audio_assignments"] = [{
        "audio_span_id": "C1:t00", "starts_at_shot_index": 0,
        "offset_seconds": "0:00", "completion": "complete_thought",
        "gain_db": 0.0, "why": "answer",
    }]
    planner.expand_spans(chosen, list(_material()[0].spans))
    planner.expand_audio_assignments(chosen, ["C1:t00"])
    repairs = planner.normalize_selection(chosen, _material())
    assert chosen["audio_assignments"][0]["offset_seconds"] == 0.0
    assert chosen["audio_assignments"][0]["audio_id"] == "a00"
    assert chosen["shots"][0]["audio_role"] == "discard"
    assert chosen["shots"][0]["audio_completion"] == "none"
    assert any("top-level narrative" in repair for repair in repairs)


def test_multiple_sources_without_stringout_stop_before_upload(monkeypatch):
    monkeypatch.setattr(
        planner, "ask", lambda *a, **k: (_ for _ in ()).throw(
            AssertionError("paid call must not run")
        ),
    )
    with pytest.raises(planner.PlannerError, match="one validated stringout"):
        decide_editorial_plan(_material(), brief="b", client=object())


def test_planning_route_uses_duration_not_source_count():
    many_short = [
        MaterialItem(source_id=f"C{index}", duration_seconds=10, summary="")
        for index in range(74)
    ]
    assert planner.editorial_planning_route(many_short) == "direct_stringout"
    long_interview = [
        MaterialItem(source_id="INT", duration_seconds=3600, summary="")
    ]
    assert planner.editorial_planning_route(long_interview) == "logged_selects"
    many_long = [
        MaterialItem(source_id=f"L{index}", duration_seconds=700, summary="")
        for index in range(3)
    ]
    assert planner.editorial_planning_route(many_long) == "logged_selects"


def test_material_log_must_bin_every_source_and_local_only_applies_gemini_rank():
    material = [
        MaterialItem(source_id="A", duration_seconds=1000, summary=""),
        MaterialItem(source_id="B", duration_seconds=1000, summary=""),
        MaterialItem(source_id="C", duration_seconds=1000, summary=""),
    ]
    incomplete = {
        "assessment": "", "bins": [{
            "name": "hero", "purpose": "", "source_ids": ["A", "B"],
        }],
        "selects": [{"source_id": "A", "why": "", "roles": []}],
    }
    with pytest.raises(planner.PlannerError, match="unbinned=.*C"):
        planner.prepare_material_log(incomplete, material)

    complete = {
        "assessment": "", "bins": [{
            "name": "all", "purpose": "", "source_ids": ["A", "B", "C"],
        }],
        # Gemini preference is C, then A. The local reel restores rushes order
        # and merely respects that inclusion decision.
        "selects": [
            {"source_id": "C", "why": "detail", "roles": []},
            {"source_id": "A", "why": "establish", "roles": []},
            {"source_id": "B", "why": "fallback", "roles": []},
        ],
    }
    planner.prepare_material_log(complete, material)
    selected = planner.material_log_selects(complete, material, max_seconds=2100)
    assert [one.source_id for one in selected] == ["A", "C"]


def test_long_single_source_logging_must_classify_every_named_span():
    spans = tuple(
        SimpleNamespace(
            span_id=f"INT:v{index:02d}",
            starts_seconds=float(index * 900),
            ends_seconds=float(index * 900 + 500),
        )
        for index in range(4)
    )
    material = [MaterialItem(
        source_id="INT", duration_seconds=3600, summary="interview",
        spans=spans,
    )]
    payload = {
        "assessment": "four interview sections",
        "bins": [{
            "name": "answers", "purpose": "story",
            "source_ids": ["INT"],
            "span_ids": [span.span_id for span in spans],
        }],
        "selects": [{
            "source_id": "INT",
            "span_ids": [span.span_id for span in spans[:2]],
            "why": "main", "roles": [],
        }],
        "target_coverage": [],
    }
    planner.prepare_material_log(payload, material)
    selected = planner.material_log_selects(payload, material)
    assert selected[0].planning_ranges == ((0.0, 500.0), (900.0, 1400.0))
    all_selected = {
        **payload,
        "selects": [{
            **payload["selects"][0],
            "span_ids": [span.span_id for span in spans],
        }],
    }
    with pytest.raises(planner.PlannerError, match="no executable selects"):
        planner.material_log_selects(all_selected, material)
    duplicate = {
        **payload,
        "selects": [payload["selects"][0], payload["selects"][0]],
    }
    with pytest.raises(planner.PlannerError, match="duplicate selects"):
        planner.prepare_material_log(duplicate, material)
    incomplete = {
        **payload,
        "bins": [{**payload["bins"][0], "span_ids": [spans[0].span_id]}],
    }
    with pytest.raises(planner.PlannerError, match="unbinned spans"):
        planner.prepare_material_log(incomplete, material)


def _grounding_spec(policy="context_allowed", targets=("device.fold",)):
    return SimpleNamespace(identity_lock=SimpleNamespace(
        framing=SimpleNamespace(
            required_target_ids=targets,
            editorial_presence_policy=policy,
        ),
        identity=SimpleNamespace(
            targets=tuple(SimpleNamespace(target_id=one) for one in targets)
        ),
    ))


def test_large_log_requires_primary_and_distinct_alternate_per_target():
    material = [
        MaterialItem(source_id="A", duration_seconds=10, summary="fold hero"),
        MaterialItem(source_id="B", duration_seconds=10, summary="fold detail"),
        MaterialItem(
            source_id="C", duration_seconds=10, summary="room",
            carries_identity=False,
            identity_absent_targets=("device.fold",),
        ),
    ]
    payload = {
        "assessment": "",
        "bins": [{
            "name": "all", "purpose": "", "source_ids": ["A", "B", "C"],
        }],
        "selects": [
            {"source_id": one, "why": "", "roles": []}
            for one in ("A", "B", "C")
        ],
        "target_coverage": [{
            "target_id": "device.fold",
            "primary_source_ids": ["A"],
            "alternate_source_ids": [],
            "why": "",
        }],
    }
    with pytest.raises(planner.PlannerError, match="no alternate source"):
        planner.prepare_material_log(
            payload, material, grounding_spec=_grounding_spec()
        )

    payload["target_coverage"][0]["alternate_source_ids"] = ["B"]
    planner.prepare_material_log(
        payload, material, grounding_spec=_grounding_spec()
    )


def test_large_log_target_coverage_must_survive_selected_span_ranges():
    spans = (
        Span("A:fold", "A", 0, 100, "Fold hero", "identity locked"),
        Span("A:room", "A", 500, 600, "empty room", "context"),
    )
    material = [MaterialItem(
        source_id="A",
        duration_seconds=900,
        summary="Fold appears only near the start",
        spans=spans,
        identity_windows_by_target=(("device.fold", ((0.0, 100.0),)),),
    )]
    payload = {
        "assessment": "",
        "bins": [{
            "name": "all", "purpose": "", "source_ids": ["A"],
            "span_ids": [span.span_id for span in spans],
        }],
        "selects": [{
            "source_id": "A", "span_ids": ["A:room"],
            "why": "context", "roles": [],
        }],
        "target_coverage": [{
            "target_id": "device.fold",
            "primary_source_ids": ["A"],
            "alternate_source_ids": [],
            "why": "source contains Fold somewhere",
        }],
    }
    with pytest.raises(
        planner.PlannerError, match="selected spans cannot claim target"
    ):
        planner.prepare_material_log(
            payload, material, grounding_spec=_grounding_spec()
        )

    payload["selects"][0]["span_ids"] = ["A:fold"]
    planner.prepare_material_log(
        payload, material, grounding_spec=_grounding_spec()
    )
    catalog = planner._describe_editorial_catalog(material)
    assert "device.fold可見區間=0.0–100.0s" in catalog


def test_target_only_rejects_context_from_log_and_from_finished_selection():
    material = [
        MaterialItem(
            source_id="A", duration_seconds=10, summary="fold",
            spans=(Span("A:s00", "A", 0, 5, "fold", "locked"),),
        ),
        MaterialItem(
            source_id="C", duration_seconds=10, summary="room",
            spans=(Span("C:s00", "C", 0, 5, "room", "locked"),),
            carries_identity=False,
            identity_absent_targets=("device.fold",),
        ),
    ]
    payload = {
        "assessment": "",
        "bins": [{
            "name": "all", "purpose": "", "source_ids": ["A", "C"],
        }],
        "selects": [
            {"source_id": "A", "why": "hero", "roles": []},
            {"source_id": "C", "why": "room", "roles": []},
        ],
        "target_coverage": [{
            "target_id": "device.fold",
            "primary_source_ids": ["A"],
            "alternate_source_ids": [],
            "why": "only one eligible source",
        }],
    }
    spec = _grounding_spec("target_only")
    with pytest.raises(planner.PlannerError, match="context-only sources"):
        planner.prepare_material_log(payload, material, grounding_spec=spec)

    target = _executable_shot(
        "A:s00", subject="fold", story_point="hero"
    )
    target["source_id"] = "A"
    target["looks"][0]["entity_id"] = "device.fold"
    context = _executable_shot(
        "C:s00", subject="room", story_point="context"
    )
    context["source_id"] = "C"
    faults = planner.grounding_presence_disagreements(
        [target, context], material, spec
    )
    assert any("k01 violates target_only" in fault for fault in faults)


def test_flat_plan_has_no_synthesised_legacy_contract():
    # Shot count is emergent and the fallback names an executable span.
    assert len(_PLAN["shots"]) == 2
    assert _PLAN["shots"][0]["fallback_span_id"] == "C2:s00"
    assert "target_shot_count" not in _PLAN
    assert "candidate_options" not in _PLAN
    assert "commitment_id" not in _PLAN["shots"][0]
    assert "tier" not in _PLAN.get("shots", [{}])[0]


def test_editorial_plan_is_on_by_default_with_a_legacy_escape(monkeypatch):
    from montagewright.cli import _editorial_plan_enabled

    monkeypatch.delenv("MONTAGEWRIGHT_EDITORIAL_PLAN", raising=False)
    args = argparse.Namespace(editorial_plan=False)
    assert _editorial_plan_enabled(args) is True

    args = argparse.Namespace(editorial_plan=True)
    assert _editorial_plan_enabled(args) is True

    monkeypatch.setenv("MONTAGEWRIGHT_EDITORIAL_PLAN", "1")
    assert _editorial_plan_enabled(argparse.Namespace(editorial_plan=False)) is True

    monkeypatch.setenv("MONTAGEWRIGHT_EDITORIAL_PLAN", "0")
    assert _editorial_plan_enabled(argparse.Namespace(editorial_plan=True)) is False

    monkeypatch.delenv("MONTAGEWRIGHT_EDITORIAL_PLAN", raising=False)
    assert _editorial_plan_enabled(argparse.Namespace(
        editorial_plan=True, legacy_three_pass=True,
    )) is False


def test_direct_identity_claims_include_primary_and_fallback_and_can_spend_it():
    from montagewright.cli import (
        _editorial_plan_identity_pairs,
        _spend_editorial_fallbacks,
    )

    plan = {
        "shots": [{
            "span_id": "C1:s00", "fallback_span_id": "C2:s00",
            "looks": [{"entity_id": "device.x"}],
        }],
    }
    assert _editorial_plan_identity_pairs(plan) == {
        ("C1", "device.x"), ("C2", "device.x"),
    }
    notes = _spend_editorial_fallbacks(
        plan, {("C1", "device.x"): {"status": "hard_negative"}},
    )
    assert plan["shots"][0]["span_id"] == "C2:s00"
    assert plan["shots"][0]["fallback_used_for_span_id"] == "C1:s00"
    assert notes


def test_recorded_plan_replays_through_the_paid_response_contract(tmp_path):
    from montagewright.planner import load_editorial_plan_replay

    path = tmp_path / "plan.json"
    path.write_text(
        json.dumps(_executable_plan("C1:s00", subject="phone")),
        encoding="utf-8",
    )
    replay = load_editorial_plan_replay(
        path, _material(), aspect="9:16", seconds=3.0,
    )
    assert replay["aspect"] == "9:16"
    assert replay["target_seconds"] == 3.0
    assert replay["shots"][0]["span_id"] == "C1:s00"


def test_replay_rejects_a_shape_the_paid_schema_would_reject(tmp_path):
    import pytest
    from montagewright.planner import PlannerError, load_editorial_plan_replay

    broken = _executable_plan("C1:s00", subject="phone")
    del broken["shots"][0]["story_point"]
    path = tmp_path / "broken.json"
    path.write_text(json.dumps(broken), encoding="utf-8")
    with pytest.raises(PlannerError, match=r"shots\[0\]\.story_point is required"):
        load_editorial_plan_replay(path, _material(), aspect="9:16")


def test_direct_audit_stops_before_paid_selection_repair_by_default():
    import inspect
    from montagewright import cli

    source = inspect.getsource(cli.command_render)
    guard = source.index("allow_paid_plan_repair")
    second_call = source.index("provider_selection, usage_selection = select_shots(")
    assert guard < second_call
    assert "second paid planning call" in source
    assert "decide_rhythm_first=not direct_editorial_plan" in source
    assert 'if getattr(args, "editorial_plan_replay", None) is not None' in source
    assert source.index("pre_identity_faults = audit_cached_selection(") < source.index(
        "confirmed_identities.update(_confirm_material_identity("
    )


def test_recorded_plan_reaches_real_ffmpeg_render_without_a_client(tmp_path):
    from montagewright.cli import _edl_from_selection, _spend_editorial_fallbacks
    from montagewright.pipeline import probe, run
    from montagewright.planner import (
        audit_cached_selection, expand_spans, load_editorial_plan_replay,
        normalize_selection,
    )

    primary_path = (
        Path(__file__).parents[1]
        / "fixtures/generated/A_silent_phone_ui.mp4"
    ).resolve()
    fallback_path = (
        Path(__file__).parents[1]
        / "fixtures/generated/B_person_phone_motion_16x9.mp4"
    ).resolve()
    primary_id, fallback_id = primary_path.stem, fallback_path.stem
    primary_span, fallback_span = f"{primary_id}:s00", f"{fallback_id}:s00"
    material = [
        MaterialItem(
            primary_id, 8.0, "phone UI",
            spans=(Span(
                primary_span, primary_id, 0.0, 6.0, "phone UI", "locked",
            ),),
            subjects=("phone UI",),
        ),
        MaterialItem(
            fallback_id, 8.0, "person holding phone",
            spans=(Span(
                fallback_span, fallback_id, 0.0, 6.0, "phone", "locked",
            ),),
            subjects=("phone UI",),
        ),
    ]
    recorded = _executable_plan(primary_span, subject="phone UI")
    recorded["shots"][0]["fallback_span_id"] = fallback_span
    replay_path = tmp_path / "recorded-plan.json"
    replay_path.write_text(
        json.dumps(recorded),
        encoding="utf-8",
    )
    plan = load_editorial_plan_replay(
        replay_path, material, aspect="9:16", seconds=3.0,
    )
    # Simulate the exact-frame identity gate disproving the primary. This is
    # the same deterministic handoff used after a real grounding response;
    # no provider or SAM download is involved in the smoke test.
    plan["shots"][0]["looks"][0]["entity_id"] = "device.test"
    _spend_editorial_fallbacks(
        plan, {(primary_id, "device.test"): {"status": "hard_negative"}},
    )
    plan["shots"][0]["looks"][0]["entity_id"] = "none"
    assert plan["shots"][0]["span_id"] == fallback_span
    expand_spans(
        plan, [span for item in material for span in item.spans],
        source_motion={item.source_id: item.camera_motion for item in material},
    )
    assert normalize_selection(plan, material, commitments=None) == ()
    assert audit_cached_selection(
        plan, material, plan, commitments=None, duration_mode="preferred",
    ) == []
    edl, _ = _edl_from_selection(
        plan, fallback_path.parent, {}, material=material,
    )
    assert edl.clips[0].source_id == fallback_id
    output = tmp_path / "render"
    result, _, _, _ = run(
        edl, {fallback_id: probe(fallback_id, fallback_path)}, None, output,
        target_aspect=9 / 16,
        intent=str(plan["direction"]),
        decide_rhythm_first=False,
        target_seconds=3.0,
        duration_mode="preferred",
        client=None,
        checkpoint=None,
    )
    assert result.preview.exists()
    assert result.deliverable.exists()
    assert (output / "work/editorial-timeline-v2.json").exists()


def test_merged_selection_normalizes_mmss_before_looks_of_reads_it():
    # The crash: the merged call returns looks[].seconds as MM:SS (the schema
    # says so), and looks_of does float() on it. The merged path must run the
    # same expand_spans the three-call path runs, or it raises. This drives the
    # exact repro: MM:SS in, floats out, looks_of does not raise.
    from montagewright.planner import expand_spans
    from montagewright.schema import looks_of
    from montagewright.spans import Span
    from types import SimpleNamespace

    plan = {
        "reasoning": "r", "material_assessment": "m", "direction": "d",
        "target_seconds": "1:00", "unusable": [],
        "shots": [
            {"span_id": "C1:s00", "start_offset_seconds": "0:00",
             "seconds_needed": "0:03", "camera_intent": "reveal", "why": "w",
             "looks": [
                 {"at": "left", "seconds": "0:01.5", "framing": "thirds"},
                 {"at": "right", "seconds": "0:02.5", "framing": "centre"},
             ]},
        ],
    }
    selection = plan
    # Raw from the model: MM:SS strings, exactly what crashed.
    assert selection["shots"][0]["looks"][0]["seconds"] == "0:01.5"

    offered = [Span("C1:s00", "C1", 0.0, 10.0, "left", "authored")]
    expand_spans(
        selection, offered,
        source_motion={"C1": SimpleNamespace(kind="authored")},
    )
    # After the merged path's normalization: floats, resolved.
    looks = selection["shots"][0]["looks"]
    assert looks[0]["seconds"] == 1.5 and looks[1]["seconds"] == 2.5
    assert isinstance(selection["shots"][0]["seconds_needed"], float)
    assert selection["shots"][0]["source_id"] == "C1"
    # And the reader that crashed now succeeds.
    read = looks_of(selection["shots"][0])
    assert [round(one.seconds, 3) for one in read] == [1.5, 2.5]


def test_subject_location_is_checkpointed_immediately(monkeypatch, tmp_path):
    frame = tmp_path / "frame.jpg"
    frame.write_bytes(b"same sampled pixels")
    calls = []

    monkeypatch.setattr(
        planner, "upload_now",
        lambda *a, **k: SimpleNamespace(uri="files/frame"),
    )

    def fake_ask(client, **request):
        calls.append(request)
        return SimpleNamespace(
            status="completed",
            output_text=json.dumps({
                "frames": [{
                    "frame_index": 0, "present": True,
                    "centre_x": 0.5, "centre_y": 0.5,
                    "width": 0.2, "height": 0.4,
                }],
                "disambiguation": "black shirt",
            }),
            usage={"total_input_tokens": 10, "total_output_tokens": 10},
        )

    monkeypatch.setattr(planner, "ask", fake_ask)
    first, _ = planner.locate_subject(
        [frame], "person in black", client=object(),
        cache_dir=tmp_path / "subject-cache",
    )
    second, usage = planner.locate_subject(
        [frame], "person in black", client=object(),
        cache_dir=tmp_path / "subject-cache",
    )

    assert first == second
    assert len(calls) == 1
    assert calls[0]["generation_config"]["max_output_tokens"] == 4096
    assert usage == planner.Usage(0, 0, 0)
