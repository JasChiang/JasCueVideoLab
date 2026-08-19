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

    # One commitment per shot; the shot's own source is the primary, and a
    # declared fallback_source becomes the ALTERNATE (a fallback, never a
    # second shot).
    options = direction["candidate_options"]
    primaries = [o for o in options if o["tier"] == "primary"]
    alternates = [o for o in options if o["tier"] == "alternate"]
    assert len(primaries) == 2
    assert len(alternates) == 1  # only the first shot named a fallback
    assert alternates[0]["span_id"] == "C2:s00"
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
