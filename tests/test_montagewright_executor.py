"""The execution layer's contract: it carries out plans, it does not judge them."""

from __future__ import annotations

from pathlib import Path

import pytest

from montagewright.executor import (
    CROP_MARGIN,
    CropBox,
    MissingSource,
    Source,
    plan_render,
)
from montagewright.schema import EDL, Clip, Reframe, Subject

UHD = Source("uhd", Path("uhd.mp4"), duration_seconds=10.0, width=3840, height=2160)
PORTRAIT = 9 / 16


def _clip(clip_id: str, start: float, end: float, **kwargs: object) -> Clip:
    return Clip(
        clip_id=clip_id,
        source_id="uhd",
        approx_in_seconds=start,
        approx_out_seconds=end,
        **kwargs,
    )


def _edl(*clips: Clip) -> EDL:
    return EDL(project_id="test", clips=list(clips))


def test_every_clip_becomes_a_segment() -> None:
    """The whole point. A plan of five beats renders five beats."""

    edl = _edl(*[_clip(f"c{index}", index, index + 1) for index in range(5)])
    plan = plan_render(edl, {"uhd": UHD})
    assert [segment.clip_id for segment in plan.segments] == [
        "c0", "c1", "c2", "c3", "c4"
    ]


def test_a_window_past_the_end_still_renders() -> None:
    """An impossible trim degrades to a real one; it does not vanish.

    A missing beat leaves a hole nobody can review. A short one can be seen,
    argued with, and replanned.
    """

    plan = plan_render(_edl(_clip("late", 50.0, 60.0)), {"uhd": UHD})

    assert len(plan.segments) == 1
    segment = plan.segments[0]
    assert segment.duration_seconds > 0.0
    assert segment.out_seconds <= UHD.duration_seconds

    step = next(
        step for step in plan.degradations if step.clip_id == "late"
    )
    assert step.ladder_other == "trim_window_clamped_to_source"
    # The numbers behind the claim travel with it.
    assert step.measured["requested_out"] == 60.0
    assert step.measured["source_duration"] == 10.0


def test_a_partly_long_window_is_trimmed_without_being_called_a_degradation() -> None:
    """Running past the end by a little is a note, not a rung down."""

    plan = plan_render(_edl(_clip("over", 8.0, 20.0)), {"uhd": UHD})

    assert plan.segments[0].out_seconds == 10.0
    assert not plan.degradations
    assert any("out-point trimmed" in note for note in plan.notes)


def test_the_executor_no_longer_aims_from_a_nine_box_name() -> None:
    """That name says which subject is meant, not where to point.

    Read as a coordinate it put "mid_right" at 0.615 for a handset spanning
    0.475 to 0.825, delivering it half out of frame with the backdrop behind
    it -- the same mistake already found on the handoff path, where two
    handsets at 0.359 and 0.635 were panned between 0.192 and 0.808. Every
    position acted on now is measured: the card's box, SAM propagation, or a
    grounding call. Reaching the executor's fallback means none of them had
    an answer, and it says so instead of guessing.
    """

    clip = _clip(
        "framed",
        0.0,
        2.0,
        reframe=Reframe(subject=Subject(description="the left, grey handset")),
    )
    plan = plan_render(_edl(clip), {"uhd": UHD}, target_aspect=PORTRAIT)
    crop = plan.segments[0].crop
    assert crop is not None
    assert abs(crop.x - (1.0 - crop.width) / 2.0) < 1e-9, "centred, not guessed"
    assert any(
        "never located" in step.trigger for step in plan.degradations
    ), plan.degradations


def test_a_subject_nobody_could_place_is_a_degradation() -> None:
    """It stopped being "doing what the plan asked" when the aim went.

    The executor used to satisfy a stated subject out of its nine-box name,
    so arriving here was normal. Now every real position is measured
    upstream, and arriving here means the subject was named and never found
    -- a centred frame is the fallback, and a fallback is recorded.
    """

    clip = _clip(
        "framed",
        0.0,
        2.0,
        reframe=Reframe(subject=Subject(description="the watch face")),
    )
    plan = plan_render(_edl(clip), {"uhd": UHD}, target_aspect=PORTRAIT)
    assert [step.ladder for step in plan.degradations] == ["center_crop"]


def test_a_missing_subject_centres_and_says_so() -> None:
    plan = plan_render(
        _edl(_clip("bare", 0.0, 2.0)), {"uhd": UHD}, target_aspect=PORTRAIT
    )
    step = next(step for step in plan.degradations if step.clip_id == "bare")
    assert step.ladder == "center_crop"
    assert "no subject" in step.trigger


def test_a_matching_aspect_is_not_cropped() -> None:
    plan = plan_render(
        _edl(_clip("wide", 0.0, 2.0)), {"uhd": UHD}, target_aspect=16 / 9
    )
    assert plan.segments[0].crop is None


def test_a_missing_source_is_the_one_refusal() -> None:
    """Not a judgement about the material -- the file simply is not there."""

    edl = EDL(
        project_id="test",
        clips=[
            Clip(
                clip_id="orphan",
                source_id="never_supplied",
                approx_in_seconds=0.0,
                approx_out_seconds=1.0,
            )
        ],
    )
    with pytest.raises(MissingSource, match="never_supplied"):
        plan_render(edl, {"uhd": UHD})


def test_the_plan_reports_its_own_cost() -> None:
    plan = plan_render(
        _edl(_clip("a", 0.0, 2.0), _clip("b", 50.0, 60.0)), {"uhd": UHD}
    )
    assert plan.duration_seconds == pytest.approx(
        sum(segment.duration_seconds for segment in plan.segments)
    )
    assert plan.degraded_clip_ids == {"b"}


class TestCropBox:
    def test_pixel_origin_of_zero_stays_zero(self) -> None:
        assert CropBox(0.0, 0.0, 1.0, 0.5625).to_pixels(3840, 2160)[:2] == (0, 0)

    def test_extent_never_runs_off_the_frame(self) -> None:
        x, y, width, height = CropBox(0.9, 0.9, 0.2, 0.2).to_pixels(3840, 2160)
        assert x + width <= 3840
        assert y + height <= 2160

    def test_pixel_values_are_even(self) -> None:
        for value in CropBox(0.3333, 0.1111, 0.4444, 0.7777).to_pixels(1920, 1080):
            assert value % 2 == 0

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"x": 0.0, "y": 0.0, "width": 0.0, "height": 0.5},
            {"x": 0.0, "y": 0.0, "width": 1.5, "height": 0.5},
            {"x": -0.1, "y": 0.0, "width": 0.5, "height": 0.5},
        ],
    )
    def test_nonsense_extents_are_rejected(self, kwargs: dict[str, float]) -> None:
        with pytest.raises(ValueError):
            CropBox(**kwargs)


def test_speed_ratio_separates_screen_time_from_source_time():
    from montagewright.executor import Segment

    # Recorded speed: the two clocks agree, which is what keeps every
    # existing render bit-identical while the field exists but is unused.
    played = Segment(clip_id="k00", source=UHD, in_seconds=2.0, out_seconds=6.0)
    assert played.duration_seconds == 4.0
    assert played.screen_duration_seconds == 4.0

    # Twice source per screen second reads a four second window in two.
    fast = Segment(
        clip_id="k01", source=UHD, in_seconds=2.0, out_seconds=6.0,
        speed_ratio=2.0,
    )
    assert fast.duration_seconds == 4.0
    assert fast.screen_duration_seconds == 2.0

    # Half source per screen second stretches the same window to eight.
    slow = Segment(
        clip_id="k02", source=UHD, in_seconds=2.0, out_seconds=6.0,
        speed_ratio=0.5,
    )
    assert slow.duration_seconds == 4.0
    assert slow.screen_duration_seconds == 8.0

    # A degenerate ratio never divides by zero; it falls back to recorded.
    guarded = Segment(
        clip_id="k03", source=UHD, in_seconds=2.0, out_seconds=6.0,
        speed_ratio=0.0,
    )
    assert guarded.screen_duration_seconds == 4.0


def test_speed_reads_more_source_than_screen_time_without_moving_the_timeline():
    # A two second screen window at double speed reads four seconds of source,
    # yet still occupies two seconds of the delivered timeline: the rhythm and
    # grounding passes reasoned in screen time and never see the difference.
    plan = plan_render(_edl(_clip("fast", 1.0, 3.0, speed=2.0)), {"uhd": UHD})
    segment = plan.segments[0]
    assert segment.speed_ratio == 2.0
    assert segment.in_seconds == 1.0
    assert segment.out_seconds == 5.0          # source read = screen * speed
    assert segment.duration_seconds == 4.0     # source span
    assert segment.screen_duration_seconds == 2.0   # timeline length


def test_slow_motion_reads_less_source_for_the_same_screen_time():
    plan = plan_render(_edl(_clip("slow", 2.0, 4.0, speed=0.5)), {"uhd": UHD})
    segment = plan.segments[0]
    assert segment.speed_ratio == 0.5
    assert segment.out_seconds == 3.0          # 2s screen * 0.5 = 1s source
    assert segment.screen_duration_seconds == 2.0


def test_speed_is_dropped_while_the_camera_follows_a_subject():
    from montagewright.reframe import CropPath, Keyframe
    from montagewright.executor import CropBox

    following = CropPath(keyframes=[
        Keyframe(0.0, CropBox(0.2, 0.0, 0.3, 1.0)),
        Keyframe(2.0, CropBox(0.5, 0.0, 0.3, 1.0)),
    ])
    assert not following.is_static
    plan = plan_render(
        _edl(_clip("track", 1.0, 3.0, speed=2.0)), {"uhd": UHD},
        crop_paths={"track": following},
    )
    segment = plan.segments[0]
    # Recorded speed wins so the tracked path still describes the frames shown.
    assert segment.speed_ratio == 1.0
    assert segment.out_seconds == 3.0
    step = next(s for s in plan.degradations if s.clip_id == "track")
    assert step.ladder_other == "speed_dropped_under_camera_move"
    assert step.measured["requested_speed"] == 2.0
