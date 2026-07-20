from __future__ import annotations

import pytest

from jascue_video_lab.geometry import (
    box_iou,
    center_distance,
    native_yxyx_to_canonical_xyxy,
    normalized_to_pixels,
)


def test_normalized_to_pixels_full_frame() -> None:
    assert normalized_to_pixels((0, 0, 1000, 1000), 1920, 1080) == (0, 0, 1920, 1080)


def test_normalized_to_pixels_rounds_outward() -> None:
    assert normalized_to_pixels((1, 1, 999, 999), 100, 50) == (0, 0, 100, 50)


def test_iou_identity_and_disjoint() -> None:
    box = (100, 100, 300, 300)
    assert box_iou(box, box) == 1.0
    assert box_iou(box, (400, 400, 500, 500)) == 0.0


def test_iou_partial_overlap() -> None:
    assert box_iou((0, 0, 200, 200), (100, 100, 300, 300)) == pytest.approx(1 / 7)


def test_center_distance_uses_normalized_coordinate_space() -> None:
    assert center_distance((0, 0, 100, 100), (100, 100, 200, 200)) == pytest.approx(2**0.5 * 100)


def test_native_yxyx_to_canonical_xyxy() -> None:
    assert native_yxyx_to_canonical_xyxy((30, 494, 956, 775)) == (494, 30, 775, 956)


def test_run03_native_order_matches_reference_after_adapter() -> None:
    reference = (495, 29, 775, 927)
    canonical = native_yxyx_to_canonical_xyxy((30, 494, 956, 775))
    assert box_iou(canonical, reference) == pytest.approx(0.964, abs=0.001)
