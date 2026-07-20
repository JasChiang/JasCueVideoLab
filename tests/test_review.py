from __future__ import annotations

from PIL import Image

from jascue_video_lab.models import GroundingCandidate, GroundingProposal, Occlusion
from jascue_video_lab.review import render_manual_review
from jascue_video_lab.storage import write_json


def test_render_manual_review_creates_overlay_and_html(tmp_path, provenance) -> None:
    artifact_root = tmp_path / "artifact"
    grounding_dir = artifact_root / "run-01" / "events" / "event-1" / "groundings" / "phone-1"
    grounding_dir.mkdir(parents=True)
    frame_path = grounding_dir.parents[1] / "frame.png"
    Image.new("RGB", (100, 100), "#222222").save(frame_path)
    proposal = GroundingProposal(
        asset_id="sha256:" + "a" * 64,
        event_id="event-1",
        entity_id="phone-1",
        frame_pts=100,
        frame_time_ms=5000,
        frame_hash="b" * 64,
        source_width=100,
        source_height=100,
        visible=True,
        occlusion=Occlusion.NONE,
        visibility_reason="visible",
        candidates=[
            GroundingCandidate(
                box_2d=(100, 100, 900, 900),
                label="Phone",
                confidence=0.9,
                disambiguation_reason="central phone",
            )
        ],
        model_provenance=provenance,
    )
    write_json(grounding_dir / "grounding.json", proposal)
    write_json(
        artifact_root / "comparison.json",
        {
            "human_annotation_comparison": [
                {
                    "run_id": "run-01",
                    "bbox_matches": [
                        {
                            "comparable": True,
                            "predicted_entity_id": "phone-1",
                            "predicted_frame_time_ms": 5000,
                            "reference_entity_label": "Phone",
                            "reference_frame_time_ms": 5000,
                            "iou": 1.0,
                            "center_distance_normalized": 0.0,
                        }
                    ],
                }
            ]
        },
    )
    annotations = tmp_path / "annotations.json"
    write_json(
        annotations,
        {"boxes": [{"entity_label": "Phone", "frame_time_ms": 5000, "box_2d": [100, 100, 900, 900]}]},
    )

    output = render_manual_review(artifact_root, annotations, tmp_path / "review")

    assert output.exists()
    document = output.read_text(encoding="utf-8")
    assert "PASS 1" in document
    assert "IoU 1.000" in document
    assert (output.parent / "run-01-01-phone-1.png").exists()
