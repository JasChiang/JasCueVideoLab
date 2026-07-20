from __future__ import annotations

from PIL import Image

from jascue_video_lab.ab_review import render_grounding_ab_review
from jascue_video_lab.storage import write_json


def _summary(path, iou: float) -> None:
    (path.parent / "repeat-01").mkdir(parents=True)
    Image.new("RGB", (16, 9), "black").save(path.parent / "repeat-01" / "debug.png")
    write_json(
        path,
        {
            "target_description": "phone screen",
            "runs": [
                {
                    "run": "repeat-01",
                    "schema_valid": True,
                    "reference_iou": iou,
                    "reference_center_distance": 1.0,
                    "confidence": 0.99,
                }
            ],
        },
    )


def test_render_grounding_ab_review(tmp_path) -> None:
    explicit = tmp_path / "explicit" / "summary.json"
    generic = tmp_path / "generic" / "summary.json"
    _summary(explicit, 0.95)
    _summary(generic, 0.2)

    output = render_grounding_ab_review(
        explicit_summary=explicit,
        generic_summary=generic,
        output_dir=tmp_path / "review",
    )

    document = output.read_text(encoding="utf-8")
    assert "明確描述通過 1/1" in document
    assert "泛化描述通過 0/1" in document
    assert "mean IoU 0.950" in document
