from __future__ import annotations

from datetime import date, datetime, timezone
from types import SimpleNamespace

import pytest

from montagewright.cost import BudgetSpent, Ledger, pricing_for
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
        model="gemini-3.7-flash",
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
        model="gemini-3.7-flash",
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
            model="gemini-3.7-flash",
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
        model="gemini-3.7-flash",
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
    rates = pricing_for("gemini-3.7-flash")
    assert ledger.entries[0]["input_rate"] == rates["input"]
    assert ledger.entries[0]["cached_input_rate"] == rates["cached_input"]
    assert ledger.entries[0]["output_rate"] == rates["output"]
    assert ledger.entries[0]["pricing_period"] in {"promotional", "standard"}


def test_a_mixed_model_call_is_reserved_and_settled_at_its_actual_rate():
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

    rates = pricing_for("gemini-3.6-flash")
    assert ledger.entries[0]["model_id"] == "gemini-3.6-flash"
    assert ledger.entries[0]["input_rate"] == rates["input"]
    assert ledger.entries[0]["output_rate"] == rates["output"]


def test_a_transient_500_retries_once_under_one_reservation(monkeypatch):
    from montagewright import planner

    class ServerError(RuntimeError):
        code = 500

    client = _Client(tokens=100)
    original = client.interactions.create
    attempts = [0]

    def flaky(**request):
        attempts[0] += 1
        if attempts[0] == 1:
            raise ServerError("500 internal")
        return original(**request)

    client.interactions.create = flaky
    monkeypatch.setattr(planner.time, "sleep", lambda _: None)
    ledger = Ledger(cap_usd=1.0)

    ask(
        client,
        model="gemini-3.7-flash",
        input="hello",
        generation_config={"max_output_tokens": 1_000},
        ledger=ledger,
        budget_stage="selection",
    )

    assert attempts == [2]
    assert len(ledger.entries) == 1
    assert ledger.summary()["uncertain_attempts"] == 1
    assert "known USD total excludes" in ledger.summary()["cost_warning"]
    assert not ledger.reservations


def test_an_expired_file_uri_is_refreshed_once_before_stage_failure():
    class PermissionDenied(RuntimeError):
        code = 403

    class Cache:
        calls = 0

        def refresh_request_uris(self, value, client):
            self.calls += 1
            refreshed = [dict(value[0], uri="files/fresh")]
            return refreshed, 1

    client = _Client(tokens=100)
    original = client.interactions.create
    seen = []

    def expired_once(**request):
        seen.append(request["input"][0]["uri"])
        if len(seen) == 1:
            raise PermissionDenied("403 File expired: permission denied")
        return original(**request)

    client.interactions.create = expired_once
    cache = Cache()
    ask(
        client,
        model="gemini-3.7-flash",
        input=[{"type": "video", "uri": "files/expired"}],
        generation_config={"max_output_tokens": 1_000},
        upload_cache=cache,
    )

    assert seen == ["files/expired", "files/fresh"]
    assert cache.calls == 1

def test_an_uncertain_retry_survives_in_cumulative_cost_without_becoming_a_call(
    tmp_path,
):
    journal = tmp_path / "spend-events.jsonl"
    first = Ledger(cap_usd=10.0, journal_path=journal)
    first.note_uncertain_attempt("selection", status=500)
    first.record("selection", input_tokens=1_000, output_tokens=100)

    second = Ledger(cap_usd=10.0, journal_path=journal)
    cumulative = second.cumulative_summary()

    assert cumulative["calls"] == 1
    assert cumulative["uncertain_attempts"] == 1
    assert cumulative["by_stage"].keys() == {"selection"}
    assert cumulative["cost_warning"]


def test_a_400_is_not_retried_and_releases_the_reservation(monkeypatch):
    class ClientError(RuntimeError):
        code = 400

    client = _Client(tokens=100)

    def invalid(**_):
        client.interactions.calls += 1
        raise ClientError("400 invalid schema")

    client.interactions.create = invalid
    ledger = Ledger(cap_usd=1.0)
    with pytest.raises(ClientError):
        ask(
            client,
            model="gemini-3.7-flash",
            input="hello",
            generation_config={"max_output_tokens": 1_000},
            ledger=ledger,
            budget_stage="selection",
        )
    assert client.interactions.calls == 1
    assert not ledger.entries
    assert not ledger.reservations


def test_production_ledger_rejects_an_unpriced_model():
    with pytest.raises(ValueError, match="no production pricing"):
        Ledger(cap_usd=1.0, model_id="gemini-something-else")


def test_gemini_37_pricing_changes_after_the_published_utc_deadline():
    promo = pricing_for("gemini-3.7-flash", at=date(2026, 12, 31))
    standard = pricing_for(
        "gemini-3.7-flash",
        at=datetime(2027, 1, 1, tzinfo=timezone.utc),
    )

    assert promo == {
        "input": 0.75, "cached_input": 0.075, "output": 3.75,
    }
    assert standard == {
        "input": 1.50, "cached_input": 0.15, "output": 7.50,
    }


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


def test_a_project_spend_cap_is_money_even_though_it_is_a_403():
    """403 PERMISSION_DENIED, not 429: Google is not throttling the request,
    it is refusing to bill it. The translation was gated behind 429, so a
    spend cap came out as a raw ClientError traceback halfway through the
    fifth shot -- with every completed stage sitting in the cache and no
    sentence anywhere saying what to do about it.
    """

    from montagewright.planner import _is_spend_cap, _provider_budget_message

    class Refused(Exception):
        code = 403

    error = Refused(
        "403 PERMISSION_DENIED. {'error': {'code': 403, 'message': 'Spend cap "
        "breached for project: projects/723974504654 for service: "
        "generativelanguage.googleapis.com.', 'status': 'PERMISSION_DENIED'}}"
    )

    assert _is_spend_cap(error), "money, not permissions"
    said = _provider_budget_message(error)
    assert said is not None
    assert "ai.studio/spend" in said, "say where the cap lives"
    assert "resume" in said and "cached" in said, "say that nothing is lost"
