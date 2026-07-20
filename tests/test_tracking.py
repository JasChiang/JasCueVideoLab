from __future__ import annotations

import pytest

from jascue_video_lab.tracking import normalized_box_to_xywh, xywh_to_normalized_box


def test_tracking_box_round_trip() -> None:
    normalized = [415, 450, 470, 632]
    xywh = normalized_box_to_xywh(normalized, 960, 540)
    assert xywh_to_normalized_box(xywh, 960, 540) == normalized


@pytest.mark.parametrize(
    "box", [[0, 0, 0, 100], [-1, 0, 100, 100], [0, 0, 1001, 100], [0, 900, 100, 800]]
)
def test_tracking_box_rejects_invalid_geometry(box: list[int]) -> None:
    with pytest.raises(ValueError):
        normalized_box_to_xywh(box, 960, 540)
