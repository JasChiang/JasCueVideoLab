from __future__ import annotations

import pytest

from montagewright.measure.geometry import (
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


def test_a_cut_is_not_moved_onto_an_action_the_lens_had_not_found_yet():
    """Landing on the gesture is worth moving for; landing on it out of
    focus is not.

    The take this comes from holds a sharp plateau from two seconds on, and
    its in-point was pushed a second earlier onto a heart gesture -- from
    80% of the take's own best focus down to 33%, with the sharp half
    sitting inside the same eligible span.
    """

    from montagewright.focus import FocusSample, moving_into_softer

    hunting = [FocusSample(at_seconds=t / 2, share=s) for t, s in enumerate(
        [0.22, 0.32, 0.15, 0.51, 0.77, 0.83, 0.90, 0.92, 0.96, 0.96, 0.93]
    )]

    # The move that happened: 2.0s back to 1.0s.
    assert moving_into_softer(hunting, 2.0, 1.0, 3.0)
    # Forward into the sharp plateau: fine.
    assert not moving_into_softer(hunting, 1.0, 3.5, 3.0)
    # Inside the plateau: fine.
    assert not moving_into_softer(hunting, 4.0, 4.5, 1.0)
    # No reading at all is not a reason to refuse a move.
    assert not moving_into_softer([], 2.0, 1.0, 3.0)


def test_focus_is_read_against_the_take_and_not_against_other_takes():
    """A white tabletop is sharper than a face by any absolute measure."""

    from montagewright.focus import describe, FocusSample

    holds = [FocusSample(at_seconds=t / 2, share=s)
             for t, s in enumerate([0.9, 0.95, 1.0, 0.92])]
    assert describe(holds, 0.0, 2.0) == ""

    hunts = [FocusSample(at_seconds=t / 2, share=s)
             for t, s in enumerate([0.2, 0.3, 0.8, 1.0])]
    said = describe(hunts, 0.0, 2.0)
    assert "20%" in said and "1:0" not in said  # MM:SS, and it speaks up
