from __future__ import annotations

import json

from jascue_video_lab.billing import summarize_usage_and_list_price


def test_usage_summary_prices_input_output_and_thought_tokens(tmp_path) -> None:
    interaction = {
        "model": "gemini-3.6-flash",
        "usage": {
            "total_input_tokens": 1_000,
            "total_output_tokens": 100,
            "total_thought_tokens": 20,
            "input_tokens_by_modality": [
                {"modality": "VIDEO", "tokens": 700},
                {"modality": "TEXT", "tokens": 300},
            ],
        }
    }
    path = tmp_path / "test.raw_interaction.json"
    path.write_text(json.dumps(interaction), encoding="utf-8")
    summary = summarize_usage_and_list_price(tmp_path)
    assert summary["request_count"] == 1
    assert summary["input_tokens_by_modality"] == {"TEXT": 300, "VIDEO": 700}
    assert summary["billed_output_tokens"] == 120
    assert summary["model"] == "gemini-3.6-flash"
    assert summary["estimated_total_cost_usd"] == 0.0024


def test_usage_summary_deduplicates_copied_raw_interactions(tmp_path) -> None:
    interaction = {
        "model": "gemini-3.6-flash",
        "usage": {
            "total_input_tokens": 1_000,
            "total_output_tokens": 100,
            "total_thought_tokens": 20,
        }
    }
    for name in ("attempt.raw_interaction.json", "canonical.raw_interaction.json"):
        (tmp_path / name).write_text(json.dumps(interaction), encoding="utf-8")

    summary = summarize_usage_and_list_price(tmp_path)

    assert summary["request_count"] == 1
    assert summary["duplicate_artifact_count"] == 1
    assert summary["estimated_total_cost_usd"] == 0.0024


def test_usage_summary_prices_mixed_models_per_response(tmp_path) -> None:
    for name, model in (
        ("old.raw_interaction.json", "gemini-3.5-flash"),
        ("new.raw_interaction.json", "gemini-3.6-flash"),
    ):
        (tmp_path / name).write_text(
            json.dumps(
                {
                    "model": model,
                    "usage": {
                        "total_input_tokens": 1_000,
                        "total_output_tokens": 100,
                        "total_thought_tokens": 20,
                    },
                }
            ),
            encoding="utf-8",
        )

    summary = summarize_usage_and_list_price(tmp_path)

    assert summary["model"] == "mixed"
    assert summary["estimated_total_cost_usd"] == 0.00498
    assert summary["models"]["gemini-3.5-flash"]["estimated_total_cost_usd"] == 0.00258
    assert summary["models"]["gemini-3.6-flash"]["estimated_total_cost_usd"] == 0.0024


def test_usage_summary_refuses_unpriced_or_missing_model(tmp_path) -> None:
    (tmp_path / "unknown.raw_interaction.json").write_text(
        json.dumps({"model": "unknown-model", "usage": {"total_input_tokens": 1}}),
        encoding="utf-8",
    )

    try:
        summarize_usage_and_list_price(tmp_path)
    except ValueError as error:
        assert "no Standard pricing is registered" in str(error)
    else:
        raise AssertionError("unknown model must fail closed")
