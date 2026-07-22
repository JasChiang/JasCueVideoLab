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


def test_usage_summary_counts_identical_immutable_attempts_separately(
    tmp_path,
) -> None:
    interaction = {
        "model": "gemini-3.6-flash",
        "usage": {
            "total_input_tokens": 1_000,
            "total_output_tokens": 100,
            "total_thought_tokens": 20,
        },
    }
    attempts = tmp_path / "attempts"
    attempts.mkdir()
    for name in ("first.raw_interaction.json", "second.raw_interaction.json"):
        (attempts / name).write_text(json.dumps(interaction), encoding="utf-8")
    (tmp_path / "canonical.raw_interaction.json").write_text(
        json.dumps(interaction), encoding="utf-8"
    )

    summary = summarize_usage_and_list_price(tmp_path)

    assert summary["request_count"] == 2
    assert summary["duplicate_artifact_count"] == 1
    assert summary["estimated_total_cost_usd"] == 0.0048


def test_usage_summary_counts_identical_attempts_in_nested_attempts_tree(
    tmp_path,
) -> None:
    interaction = {
        "model": "gemini-3.6-flash",
        "usage": {"total_input_tokens": 100, "total_output_tokens": 10},
    }
    for branch in ("first", "second"):
        directory = tmp_path / "attempts" / "nested" / branch
        directory.mkdir(parents=True)
        (directory / "response.raw_interaction.json").write_text(
            json.dumps(interaction), encoding="utf-8"
        )
    (tmp_path / "canonical.raw_interaction.json").write_text(
        json.dumps(interaction), encoding="utf-8"
    )

    summary = summarize_usage_and_list_price(tmp_path)

    assert summary["request_count"] == 2
    assert summary["duplicate_artifact_count"] == 1
    assert summary["duplicate_artifact_paths"] == [
        "canonical.raw_interaction.json"
    ]


def test_usage_summary_counts_identical_numbered_attempt_directories(
    tmp_path,
) -> None:
    interaction = {
        "model": "gemini-3.6-flash",
        "usage": {"total_input_tokens": 100, "total_output_tokens": 10},
    }
    for attempt in ("attempt-1", "attempt-02"):
        directory = tmp_path / "variant" / attempt
        directory.mkdir(parents=True)
        (directory / "response.raw_interaction.json").write_text(
            json.dumps(interaction), encoding="utf-8"
        )
    (tmp_path / "canonical.raw_interaction.json").write_text(
        json.dumps(interaction), encoding="utf-8"
    )

    summary = summarize_usage_and_list_price(tmp_path)

    assert summary["request_count"] == 2
    assert summary["duplicate_artifact_count"] == 1


def test_usage_summary_counts_identical_legacy_attempt_filenames(
    tmp_path,
) -> None:
    interaction = {
        "model": "gemini-3.6-flash",
        "usage": {"total_input_tokens": 100, "total_output_tokens": 10},
    }
    for name in (
        "content_map.attempt-1.raw_interaction.json",
        "content_map.attempt-02.raw_interaction.json",
    ):
        (tmp_path / name).write_text(json.dumps(interaction), encoding="utf-8")
    (tmp_path / "content_map.raw_interaction.json").write_text(
        json.dumps(interaction), encoding="utf-8"
    )

    summary = summarize_usage_and_list_price(tmp_path)

    assert summary["request_count"] == 2
    assert summary["duplicate_artifact_count"] == 1
    assert summary["duplicate_artifact_paths"] == [
        "content_map.raw_interaction.json"
    ]


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


def test_usage_summary_applies_cached_input_discount(tmp_path) -> None:
    (tmp_path / "cached.raw_interaction.json").write_text(
        json.dumps(
            {
                "model": "gemini-3.6-flash",
                "usage": {
                    "total_input_tokens": 1_000,
                    "total_cached_tokens": 800,
                    "total_output_tokens": 0,
                    "total_thought_tokens": 0,
                },
            }
        ),
        encoding="utf-8",
    )

    summary = summarize_usage_and_list_price(tmp_path)

    assert summary["total_cached_input_tokens"] == 800
    assert summary["total_uncached_input_tokens"] == 200
    assert summary["estimated_total_cost_usd"] == 0.00042


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


def test_missing_usage_is_reported_as_unpriced_not_silently_free(tmp_path) -> None:
    path = tmp_path / "attempts" / "attempt-000001" / "raw_interaction.json"
    path.parent.mkdir(parents=True)
    path.write_text(
        json.dumps(
            {"id": "response-without-usage", "model": "gemini-3.6-flash"}
        ),
        encoding="utf-8",
    )

    summary = summarize_usage_and_list_price(tmp_path)

    assert summary["request_count"] == 1
    assert summary["priced_request_count"] == 0
    assert summary["unpriced_request_count"] == 1
    assert summary["pricing_complete"] is False
    assert summary["cost_interpretation"] == "lower_bound_incomplete_usage_metadata"
    assert summary["unpriced_request_paths"] == [
        "attempts/attempt-000001/raw_interaction.json"
    ]
