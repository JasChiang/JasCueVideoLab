from __future__ import annotations

from types import SimpleNamespace

import pytest

from montagewright.cost import BudgetSpent, Ledger
from montagewright.gemini import count_request_tokens, structured_json
from montagewright.planner import ask


class _Models:
    def __init__(self, tokens: int):
        self.tokens = tokens
        self.calls = 0

    def count_tokens(self, **kwargs):
        self.calls += 1
        self.last = kwargs
        return SimpleNamespace(total_tokens=self.tokens)


class _Interactions:
    def __init__(self, usage: dict | None = None):
        self.calls = 0
        self.usage = usage or {
            "total_input_tokens": 100,
            "total_output_tokens": 20,
            "total_thought_tokens": 10,
            "total_cached_tokens": 40,
        }

    def create(self, **request):
        self.calls += 1
        return SimpleNamespace(
            status="completed", output_text="{}", usage=self.usage
        )


class _Client:
    def __init__(self, *, tokens=100, usage=None):
        self.models = _Models(tokens)
        self.interactions = _Interactions(usage)


def test_structured_json_matches_the_installed_interactions_contract():
    from google.genai._gaos.types.interactions.textresponseformat import (
        TextResponseFormat,
    )

    value = structured_json({"type": "object", "properties": {}})
    parsed = TextResponseFormat.model_validate(value)
    assert parsed.type == "text"
    assert parsed.mime_type == "application/json"


def test_count_tokens_includes_a_conservative_schema_envelope():
    client = _Client(tokens=100)
    counted = count_request_tokens(
        client,
        model="gemini-3.6-flash",
        input_value="hello",
        response_format=structured_json({"type": "object"}),
    )
    assert client.models.calls == 1
    assert counted > 105


def test_interactions_resolution_is_translated_for_count_tokens():
    from google.genai import types

    client = _Client(tokens=100)
    count_request_tokens(
        client,
        model="gemini-3.6-flash",
        input_value=[{
            "type": "video",
            "uri": "https://example.invalid/clip.mp4",
            "mime_type": "video/mp4",
            "resolution": "low",
        }],
    )
    part = client.models.last["contents"].parts[0]
    assert (
        part.media_resolution.level
        == types.PartMediaResolutionLevel.MEDIA_RESOLUTION_LOW
    )


def test_a_call_that_cannot_fit_is_never_dispatched():
    client = _Client(tokens=1_000_000)
    ledger = Ledger(cap_usd=0.01)

    with pytest.raises(BudgetSpent, match="was not sent"):
        ask(
            client,
            model="gemini-3.6-flash",
            input="hello",
            generation_config={"max_output_tokens": 1_000},
            ledger=ledger,
            budget_stage="direction",
        )

    assert client.interactions.calls == 0
    assert not ledger.entries
    assert not ledger.reservations


def test_a_completed_call_replaces_its_reservation_with_actual_usage():
    client = _Client(tokens=100)
    ledger = Ledger(cap_usd=1.0)

    ask(
        client,
        model="gemini-3.6-flash",
        input="hello",
        generation_config={"max_output_tokens": 10_000},
        ledger=ledger,
        budget_stage="selection",
    )

    assert client.interactions.calls == 1
    assert not ledger.reservations
    assert ledger.entries[0]["stage"] == "selection"
    assert ledger.entries[0]["cached"] == 40
    assert ledger.entries[0]["output"] == 30


def test_production_ledger_rejects_an_unpriced_model():
    with pytest.raises(ValueError, match="fixed"):
        Ledger(cap_usd=1.0, model_id="gemini-something-else")


def test_paid_attempts_survive_a_later_run_in_the_same_output_folder(tmp_path):
    journal = tmp_path / "spend-events.jsonl"
    first = Ledger(cap_usd=10.0, journal_path=journal)
    first.record("selection", input_tokens=1_000, output_tokens=100)

    second = Ledger(cap_usd=10.0, journal_path=journal)
    second.record("replan", input_tokens=2_000, output_tokens=200)

    cumulative = second.cumulative_summary()
    assert cumulative["calls"] == 2
    assert cumulative["spent_usd"] > second.spent_usd
    assert set(cumulative["by_stage"]) == {"selection", "replan"}


def test_planning_contract_changes_with_its_schema():
    from montagewright.cli import _planning_contract

    first = _planning_contract(
        "direction_zh-TW.txt", {"type": "object", "properties": {}}
    )
    second = _planning_contract(
        "direction_zh-TW.txt",
        {"type": "object", "properties": {"new": {"type": "string"}}},
    )
    assert first != second
