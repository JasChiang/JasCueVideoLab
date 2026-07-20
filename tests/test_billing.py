from __future__ import annotations

import json

from jascue_video_lab.billing import summarize_usage_and_list_price


def test_usage_summary_prices_input_output_and_thought_tokens(tmp_path) -> None:
    interaction = {
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
    assert summary["estimated_total_cost_usd"] == 0.00258
