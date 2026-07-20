from __future__ import annotations

import pytest

from jascue_video_lab.models import (
    SegmentationSample,
    SemanticIdentityStatus,
    TrackingState,
)
from jascue_video_lab.sam_tracking import (
    approximate_connected_components,
    binary_mask_geometry,
    classify_tracking_state,
    crossed_shot_boundary,
    normalized_box_to_xyxy,
    normalized_polygon_to_mask,
    pad_normalized_box,
)


def test_normalized_box_to_xyxy_is_x_first() -> None:
    assert normalized_box_to_xyxy([100, 200, 600, 800], 2000, 1000) == [
        200.0,
        200.0,
        1200.0,
        800.0,
    ]


def test_padding_preserves_semantic_box_order_and_clamps() -> None:
    assert pad_normalized_box([100, 200, 600, 800], 0.1) == [50, 140, 650, 860]
    assert pad_normalized_box([0, 0, 1000, 1000], 0.5) == [0, 0, 1000, 1000]


def test_normalized_polygon_rasterizes_in_x_y_order() -> None:
    mask = normalized_polygon_to_mask(
        [(100, 200), (600, 200), (600, 800), (100, 800)], 200, 100
    )
    geometry = binary_mask_geometry(mask)
    assert geometry["box_2d"] == pytest.approx([100, 200, 600, 800], abs=6)
    assert geometry["area_ratio"] == pytest.approx(0.3, abs=0.02)


def test_binary_mask_geometry_and_components() -> None:
    np = pytest.importorskip("numpy")
    mask = np.zeros((100, 200), dtype=bool)
    mask[20:60, 40:140] = True
    geometry = binary_mask_geometry(mask)
    assert geometry["area_pixels"] == 4000
    assert geometry["area_ratio"] == pytest.approx(0.2)
    assert geometry["box_2d"] == [200, 200, 700, 600]
    assert geometry["center_2d"] == [450.0, 400.0]
    assert approximate_connected_components(mask) == 1


def test_component_count_ignores_tiny_mask_speckle() -> None:
    np = pytest.importorskip("numpy")
    mask = np.zeros((100, 200), dtype=bool)
    mask[10:90, 20:180] = True
    mask[1, 1] = True
    assert approximate_connected_components(mask) == 1


def test_binary_mask_empty_has_no_guessed_geometry() -> None:
    np = pytest.importorskip("numpy")
    geometry = binary_mask_geometry(np.zeros((8, 8), dtype=bool))
    assert geometry == {
        "area_pixels": 0,
        "area_ratio": 0.0,
        "box_2d": None,
        "center_2d": None,
    }


def test_drift_gate_rejects_cut_even_with_valid_mask() -> None:
    state, reasons = classify_tracking_state(
        area_ratio=0.1,
        connected_components=1,
        mean_positive_probability=0.99,
        previous_area_ratios=[0.1, 0.1],
        center_2d=[500, 500],
        previous_center_2d=[500, 500],
        shot_boundary=True,
    )
    assert state == TrackingState.DRIFT_SUSPECTED
    assert "shot_boundary_requires_new_seed" in reasons


def test_shot_boundary_latches_until_a_new_seed() -> None:
    boundaries = [8, 16]
    assert not crossed_shot_boundary(7, 4, boundaries)
    assert crossed_shot_boundary(8, 4, boundaries)
    assert crossed_shot_boundary(23, 4, boundaries)
    assert not crossed_shot_boundary(4, 4, boundaries)
    assert crossed_shot_boundary(3, 10, boundaries)


def test_drift_gate_distinguishes_low_confidence_from_lost() -> None:
    low, _ = classify_tracking_state(
        area_ratio=0.1,
        connected_components=8,
        mean_positive_probability=0.55,
        previous_area_ratios=[0.1],
        center_2d=[500, 500],
        previous_center_2d=[500, 500],
        shot_boundary=False,
    )
    lost, reasons = classify_tracking_state(
        area_ratio=0,
        connected_components=0,
        mean_positive_probability=None,
        previous_area_ratios=[0.1],
        center_2d=None,
        previous_center_2d=[500, 500],
        shot_boundary=False,
    )
    assert low == TrackingState.LOW_CONFIDENCE
    assert lost == TrackingState.LOST
    assert reasons == ["mask_empty"]


def test_sample_contract_forbids_geometry_when_lost() -> None:
    common = {
        "sample_index": 0,
        "analysis_sample_time_ms": 0,
        "source_pts": None,
        "timing_basis": "uniform_ffmpeg_analysis_sample",
        "mask_path": None,
        "mask_sha256": None,
        "mask_area_pixels": 0,
        "mask_area_ratio": 0,
        "connected_components": 0,
        "derived_tracking_box": [1, 1, 2, 2],
        "center_2d": [1.5, 1.5],
        "mean_positive_probability": None,
        "scene_cut_score": None,
        "shot_boundary": False,
        "tracking_state": "lost",
        "state_reasons": ["mask_empty"],
        "semantic_identity_status": SemanticIdentityStatus.REVALIDATION_REQUIRED,
    }
    with pytest.raises(ValueError, match="empty masks cannot contain geometry"):
        SegmentationSample.model_validate(common)
