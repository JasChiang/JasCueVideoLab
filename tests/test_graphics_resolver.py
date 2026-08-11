from pathlib import Path

import pytest
from PIL import Image

from montagewright.graphics import (
    CopyFact,
    GraphicCue,
    GraphicStyle,
    GraphicTransform,
    GraphicsPlan,
    compile_graphic,
    graphic_preset_defaults,
)


def _plan_and_cue(*, locked: bool = False) -> tuple[GraphicsPlan, GraphicCue]:
    style, defaults = graphic_preset_defaults("editorial_minimal")
    facts = [
        CopyFact(
            fact_id="primary",
            exact_text="跨語言即時語音翻譯不中斷",
            source_kind="user",
            approved=True,
            approved_by="human_review",
        ),
        CopyFact(
            fact_id="secondary",
            exact_text="跨語言對話不再中斷",
            source_kind="user",
            approved=True,
            approved_by="human_review",
        ),
    ]
    cue = GraphicCue(
        graphic_id="feature",
        kind="feature",
        primary_fact_id="primary",
        secondary_fact_id="secondary",
        template="stat_badge",
        position=defaults.get("position", "auto"),
        composition=defaults.get("composition", "auto"),
        background=defaults.get("background", "auto"),
        motion=defaults.get("motion", "fade"),
        transform=GraphicTransform(locked=locked),
        style=GraphicStyle.model_validate(
            {**style, "preset": "editorial_minimal", "contrast_mode": "auto"}
        ),
        status="approved",
    )
    return GraphicsPlan(facts=facts, cues=[cue]), cue


def test_curated_auto_card_reflows_and_keeps_accessible_fallback(tmp_path: Path) -> None:
    plan, cue = _plan_and_cue()
    frames = [Image.new("RGB", (360, 640), "#74808A") for _ in range(3)]

    _, report = compile_graphic(
        cue, plan, width=360, height=640,
        into=tmp_path / "feature.png", frames=frames,
    )

    assert report["requested_template"] == "stat_badge"
    assert report["resolved_template"] != "stat_badge"
    assert report["contrast_ratio"] >= 4.5


def test_locked_card_does_not_silently_change_template(tmp_path: Path) -> None:
    plan, cue = _plan_and_cue(locked=True)
    frames = [Image.new("RGB", (360, 640), "#74808A") for _ in range(3)]

    with pytest.raises(ValueError, match="too long"):
        compile_graphic(
            cue, plan, width=360, height=640,
            into=tmp_path / "feature.png", frames=frames,
        )
