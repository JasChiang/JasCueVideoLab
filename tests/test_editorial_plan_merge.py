"""Milestone 2: the merged one-call editorial plan, opt-in and adapted.

These tests never call a real model. They mock the call to prove the merge's
two structural promises -- ONE call with the footage attached ONCE, and a
flat plan that bridges into the direction+selection shapes the existing
downstream reads -- and that the default render path is untouched.
"""

from __future__ import annotations

import argparse
import json
from types import SimpleNamespace

from montagewright import planner
from montagewright.planner import (
    MaterialItem,
    decide_editorial_plan,
    editorial_plan_to_legacy,
)
from montagewright.spans import Span


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
         "fallback_source": "C2"},
        {"span_id": "C2:s00", "seconds_needed": "0:03",
         "camera_intent": "reveal", "why": "detail", "looks": []},
    ],
    "music_from_seconds": 16.0, "music_spans": [],
    "unusable": [],
}


def test_decide_editorial_plan_makes_one_call_and_attaches_material_once(monkeypatch):
    calls = {"ask": 0, "material": 0, "music": 0}

    def fake_ask(client, **request):
        calls["ask"] += 1
        return SimpleNamespace(
            status="completed", output_text=json.dumps(_PLAN),
            usage={"total_input_tokens": 100, "total_output_tokens": 20},
        )

    monkeypatch.setattr(planner, "ask", fake_ask)
    monkeypatch.setattr(
        planner, "_attach_material",
        lambda *a, **k: calls.__setitem__("material", calls["material"] + 1) or [],
    )
    monkeypatch.setattr(
        planner, "_attach_music",
        lambda *a, **k: calls.__setitem__("music", calls["music"] + 1) or {"type": "x"},
    )

    plan, _ = decide_editorial_plan(
        _material(), brief="b", aspect="9:16", seconds=60.0,
        client=object(),
    )
    # Exactly one model call; the footage attached exactly once (the whole
    # point of the merge on cost -- not three times across three stages).
    assert calls["ask"] == 1
    assert calls["material"] == 1
    assert [s["span_id"] for s in plan["shots"]] == ["C1:s00", "C2:s00"]
    assert plan["target_seconds"] == 60.0
    assert plan["aspect"] == "9:16"


def test_adapter_bridges_the_flat_plan_into_direction_and_selection():
    direction, selection = editorial_plan_to_legacy(_PLAN, aspect="9:16")

    # Selection is the shots, verbatim; count is emergent, no quota.
    assert [s["span_id"] for s in selection["shots"]] == ["C1:s00", "C2:s00"]
    assert direction["target_shot_count"] == 2
    assert selection["music_from_seconds"] == 16.0

    # One primary option per shot. The declared fallback_source is NOT
    # synthesised into an alternate option: its real span in the material is
    # unknown here, so a fabricated "C2:s00" would name a span the resolver
    # rejects (verified end-to-end). It stays recorded on the shot for
    # milestone 3 to read directly; the legacy path never spent it anyway.
    options = direction["candidate_options"]
    primaries = [o for o in options if o["tier"] == "primary"]
    alternates = [o for o in options if o["tier"] == "alternate"]
    assert len(primaries) == 2
    assert alternates == []
    assert selection["shots"][0].get("fallback_source") == "C2"
    # Every synthesised option carries a resolver-legal target, never "None".
    assert all(o["target_id"] and o["target_id"] != "None" for o in options)
    # The commitment machinery the merge deletes is absent from the plan.
    assert "target_shot_count" not in _PLAN
    assert "candidate_options" not in _PLAN
    assert "tier" not in _PLAN.get("shots", [{}])[0]


def test_editorial_plan_is_off_by_default(monkeypatch):
    from montagewright.cli import _editorial_plan_enabled

    monkeypatch.delenv("MONTAGEWRIGHT_EDITORIAL_PLAN", raising=False)
    args = argparse.Namespace(editorial_plan=False)
    assert _editorial_plan_enabled(args) is False

    args = argparse.Namespace(editorial_plan=True)
    assert _editorial_plan_enabled(args) is True

    monkeypatch.setenv("MONTAGEWRIGHT_EDITORIAL_PLAN", "1")
    assert _editorial_plan_enabled(argparse.Namespace(editorial_plan=False)) is True


def test_adapter_output_survives_resolve_candidate_commitments():
    # The bridge must not just look right -- its synthesised candidate_options
    # have to pass the resolver the legacy path feeds them to, or a real
    # --editorial-plan run would crash there. This caught a "None" target and a
    # fabricated fallback span before they could waste a paid run.
    from montagewright.planner import editorial_plan_to_legacy, MaterialItem
    from montagewright.candidate_commitments import resolve_candidate_commitments
    from montagewright.spans import Span

    plan = {
        "reasoning": "r", "material_assessment": "m", "direction": "d",
        "target_seconds": 20.0, "music_under_speech": "bed", "unusable": [],
        "shots": [
            {"source_id": "C1", "span_id": "C1:s00", "camera_intent": "hold",
             "seconds_needed": 3.0,
             "looks": [{"at": "phone", "framing": "centre"}],
             "audio_role": "discard", "picture_role": "primary_action",
             "energy": "medium", "why": "w", "fallback_source": "C2"},
            {"source_id": "C3", "span_id": "C3:s00", "camera_intent": "reveal",
             "seconds_needed": 4.0,
             "looks": [{"at": "screen", "framing": "thirds",
                        "entity_id": "device.x"}],
             "audio_role": "discard", "picture_role": "primary_action",
             "energy": "medium", "why": "w2"},
        ],
    }
    direction, _ = editorial_plan_to_legacy(plan, aspect="9:16")
    material = [
        MaterialItem("C1", 10.0, "a",
                     spans=(Span("C1:s00", "C1", 0.0, 5.0, "phone", "locked"),)),
        MaterialItem("C3", 10.0, "b",
                     spans=(Span("C3:s00", "C3", 0.0, 5.0, "screen", "authored"),)),
    ]
    # Must not raise CommitmentError.
    resolved = resolve_candidate_commitments(
        direction, material, material_digest="a" * 64, aspect="9:16",
        target_seconds=20.0, grounding_target_ids=("device.x",),
        grounding_sha256="b" * 64,
    )
    assert len(resolved.options) == 2
