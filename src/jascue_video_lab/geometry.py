from __future__ import annotations

import math
from typing import Sequence


Box = tuple[int, int, int, int]


def native_yxyx_to_canonical_xyxy(box: Sequence[int]) -> Box:
    """Convert Gemini's documented [ymin, xmin, ymax, xmax] to project xyxy."""
    if len(box) != 4:
        raise ValueError("box must contain four coordinates")
    y_min, x_min, y_max, x_max = box
    canonical = (x_min, y_min, x_max, y_max)
    if not (
        0 <= canonical[0] < canonical[2] <= 1000
        and 0 <= canonical[1] < canonical[3] <= 1000
    ):
        raise ValueError("native normalized box is invalid")
    return canonical


def normalized_to_pixels(box: Sequence[int], width: int, height: int) -> Box:
    if len(box) != 4:
        raise ValueError("box must contain four coordinates")
    if width <= 0 or height <= 0:
        raise ValueError("image dimensions must be positive")
    x_min, y_min, x_max, y_max = box
    if not (0 <= x_min < x_max <= 1000 and 0 <= y_min < y_max <= 1000):
        raise ValueError("normalized box is invalid")
    # Floor minima and ceil maxima ensure a non-empty pixel box that contains
    # the normalized proposal, then clamp to Pillow's inclusive image bounds.
    return (
        max(0, min(width - 1, math.floor(x_min * width / 1000))),
        max(0, min(height - 1, math.floor(y_min * height / 1000))),
        max(1, min(width, math.ceil(x_max * width / 1000))),
        max(1, min(height, math.ceil(y_max * height / 1000))),
    )


def box_iou(left: Sequence[float], right: Sequence[float]) -> float:
    lx1, ly1, lx2, ly2 = left
    rx1, ry1, rx2, ry2 = right
    intersection_width = max(0.0, min(lx2, rx2) - max(lx1, rx1))
    intersection_height = max(0.0, min(ly2, ry2) - max(ly1, ry1))
    intersection = intersection_width * intersection_height
    left_area = max(0.0, lx2 - lx1) * max(0.0, ly2 - ly1)
    right_area = max(0.0, rx2 - rx1) * max(0.0, ry2 - ry1)
    union = left_area + right_area - intersection
    return intersection / union if union else 0.0


def center_distance(left: Sequence[float], right: Sequence[float]) -> float:
    left_center = ((left[0] + left[2]) / 2, (left[1] + left[3]) / 2)
    right_center = ((right[0] + right[2]) / 2, (right[1] + right[3]) / 2)
    return math.dist(left_center, right_center)
