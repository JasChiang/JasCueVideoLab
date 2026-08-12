"""Every camera move has to reach its own builder.

A follow branch once lost its grounding call entirely when two branches were
inserted above it, and the whole suite stayed green: nothing exercised
camera_move at all, and the planner happened not to choose follow for several
runs. The gap was found by watching a render.
"""

from __future__ import annotations

import pytest

from montagewright.capabilities import INTENT_NAMES, MOVE_NAMES
from montagewright.reframe import (
    MAX_UPSCALE,
    Observation,
    achieved_upscale,
    build_crop_path,
    build_handoff_path,
    build_sweep_path,
    build_zoom_path,
    visible_fraction,
    zoom_budget,
)
from montagewright.executor import CropBox

WIDE, TALL = 16 / 9, 9 / 16


def _moving(count: int = 5) -> list[Observation]:
    return [
        Observation(
            seconds=index * 0.5,
            centre_x=0.20 + 0.08 * index,
            centre_y=0.5,
            width=0.18,
            height=0.4,
        )
        for index in range(count)
    ]


def _still(count: int = 5) -> list[Observation]:
    return [
        Observation(
            seconds=index * 0.5,
            centre_x=0.5,
            centre_y=0.5,
            width=0.18,
            height=0.4,
        )
        for index in range(count)
    ]


def test_every_declared_move_has_a_builder() -> None:
    """The menu the planner reads and the code that dispatches cannot drift."""

    from montagewright import pipeline

    dispatch = pipeline.follow_subjects.__doc__ or ""
    source = pytest.importorskip("inspect").getsource(pipeline.follow_subjects)
    for name in MOVE_NAMES:
        assert name in source, f"{name} is offered but never dispatched"
    assert dispatch  # the helper is documented


class TestFollow:
    def test_a_moving_subject_is_followed(self) -> None:
        path = build_crop_path(
            _moving(), source_aspect=WIDE, target_aspect=TALL, energy="active"
        )
        assert not path.is_static
        assert path.travel() > 0.0

    def test_a_vertical_subject_trajectory_is_followed_on_the_available_axis(self) -> None:
        observations = [
            Observation(
                seconds=index * 0.5,
                centre_x=0.5,
                centre_y=0.20 + 0.10 * index,
                width=0.3,
                height=0.18,
            )
            for index in range(5)
        ]
        path = build_crop_path(
            observations,
            source_aspect=TALL,
            target_aspect=WIDE,
            energy="active",
        )
        assert not path.is_static
        assert path.keyframes[-1].crop.y > path.keyframes[0].crop.y

    def test_a_still_subject_holds_and_says_so(self) -> None:
        degradations: list = []
        path = build_crop_path(
            _still(),
            source_aspect=WIDE,
            target_aspect=TALL,
            energy="active",
            clip_id="k00",
            degradations=degradations,
        )
        assert path.is_static
        assert any(step.ladder == "static_on_subject" for step in degradations)

    def test_a_subject_that_returns_is_not_chased(self) -> None:
        """Out and back inside one shot reads as a wobble, not a move."""

        there_and_back = [
            Observation(seconds=index * 0.4, centre_x=x, centre_y=0.5, width=0.18, height=0.4)
            for index, x in enumerate([0.30, 0.25, 0.20, 0.21, 0.26, 0.30])
        ]
        degradations: list = []
        path = build_crop_path(
            there_and_back,
            source_aspect=WIDE,
            target_aspect=TALL,
            energy="active",
            clip_id="k01",
            degradations=degradations,
        )
        assert path.is_static
        assert degradations, "a substitution has to be recorded, not silent"

    def test_a_sweep_that_hesitates_still_follows(self) -> None:
        """One pause must not be mistaken for indecision."""

        hesitating = [
            Observation(seconds=index * 0.4, centre_x=x, centre_y=0.5, width=0.18, height=0.4)
            for index, x in enumerate([0.20, 0.30, 0.32, 0.44, 0.50, 0.60])
        ]
        path = build_crop_path(
            hesitating, source_aspect=WIDE, target_aspect=TALL, energy="active"
        )
        assert not path.is_static


class TestSweep:
    @pytest.mark.parametrize("direction", ["sweep_left", "sweep_right"])
    def test_a_sweep_moves_without_a_subject(self, direction: str) -> None:
        path = build_sweep_path(
            source_aspect=WIDE,
            target_aspect=TALL,
            duration_seconds=2.5,
            direction=direction,
            energy="active",
        )
        assert not path.is_static

    def test_the_two_directions_are_opposite(self) -> None:
        left = build_sweep_path(
            source_aspect=WIDE, target_aspect=TALL, duration_seconds=2.5,
            direction="sweep_left", energy="active",
        )
        right = build_sweep_path(
            source_aspect=WIDE, target_aspect=TALL, duration_seconds=2.5,
            direction="sweep_right", energy="active",
        )
        went_left = left.keyframes[-1].crop.x - left.keyframes[0].crop.x
        went_right = right.keyframes[-1].crop.x - right.keyframes[0].crop.x
        assert went_left < 0 < went_right


class TestZoom:
    @pytest.mark.parametrize("direction", ["push_in", "pull_out"])
    def test_a_zoom_changes_the_crop_size(self, direction: str) -> None:
        path = build_zoom_path(
            source_aspect=WIDE, target_aspect=TALL, duration_seconds=2.0,
            direction=direction, energy="dynamic", budget=0.5,
        )
        assert not path.is_static
        first, last = path.keyframes[0].crop, path.keyframes[-1].crop
        if direction == "push_in":
            assert last.width < first.width
        else:
            assert last.width > first.width

    def test_the_zoom_aims_at_the_subject(self) -> None:
        """Pushing at the centre of the frame wastes the only vertical move."""

        low = build_zoom_path(
            source_aspect=WIDE, target_aspect=TALL, duration_seconds=2.0,
            direction="push_in", centre_y=0.75, energy="dynamic",
            budget=0.5, framing="centre",
        )
        middle = build_zoom_path(
            source_aspect=WIDE, target_aspect=TALL, duration_seconds=2.0,
            direction="push_in", centre_y=0.5, energy="dynamic",
            budget=0.5, framing="centre",
        )
        assert low.keyframes[-1].crop.y > middle.keyframes[-1].crop.y

    def test_the_source_bounds_the_push(self) -> None:
        """A 4K source affords a push a 1080 one does not."""

        uhd = zoom_budget(
            source_width=3840, source_height=2160, source_aspect=WIDE,
            target_aspect=TALL, output_width=1080, output_height=1920,
        )
        hd = zoom_budget(
            source_width=1920, source_height=1080, source_aspect=WIDE,
            target_aspect=TALL, output_width=1080, output_height=1920,
        )
        assert uhd < hd, "more pixels should allow a tighter crop"

    def test_the_delivered_enlargement_is_reported(self) -> None:
        """Sharpness is measurable, so nobody should judge it from a preview."""

        budget = zoom_budget(
            source_width=3840, source_height=2160, source_aspect=WIDE,
            target_aspect=TALL, output_width=1080, output_height=1920,
        )
        path = build_zoom_path(
            source_aspect=WIDE, target_aspect=TALL, duration_seconds=2.0,
            direction="push_in", energy="dynamic", budget=budget,
        )
        tightest = min(path.keyframes, key=lambda k: k.crop.width).crop
        upscale = achieved_upscale(
            tightest, source_width=3840, source_height=2160,
            output_width=1080, output_height=1920,
        )
        assert upscale <= MAX_UPSCALE + 1e-6


class TestHandoff:
    def test_it_pans_between_measured_centres(self) -> None:
        """Not a cut, and not a guess at where the subjects are."""

        path = build_handoff_path(
            source_aspect=WIDE, target_aspect=TALL, duration_seconds=3.0,
            from_centre=0.359, to_centre=0.635, energy="calm",
        )
        assert not path.is_static
        first, last = path.keyframes[0].crop, path.keyframes[-1].crop
        assert first.x + first.width / 2 == pytest.approx(0.359, abs=0.02)
        assert last.x + last.width / 2 == pytest.approx(0.635, abs=0.02)

    def test_it_does_not_overshoot_the_subjects(self) -> None:
        """Aiming at nine-box extremes ran past both handsets into background."""

        path = build_handoff_path(
            source_aspect=WIDE, target_aspect=TALL, duration_seconds=3.0,
            from_centre=0.359, to_centre=0.635, energy="calm",
        )
        assert path.travel() < 0.4, "the move should be the gap, not the frame"


class TestFit:
    def test_a_subject_too_wide_is_reported(self) -> None:
        oversized = [
            Observation(seconds=i * 0.5, centre_x=0.5, centre_y=0.5, width=0.55, height=0.6)
            for i in range(4)
        ]
        degradations: list = []
        build_crop_path(
            oversized, source_aspect=WIDE, target_aspect=TALL,
            clip_id="k02", min_visible=0.85, degradations=degradations,
        )
        assert any(
            step.ladder_other == "subject_larger_than_crop"
            for step in degradations
        ), "a shot showing half its subject is a fact review needs"

    def test_a_subject_that_fits_is_not_reported(self) -> None:
        degradations: list = []
        build_crop_path(
            _moving(), source_aspect=WIDE, target_aspect=TALL,
            clip_id="k03", min_visible=0.85, degradations=degradations,
        )
        assert not [
            step for step in degradations
            if step.ladder_other == "subject_larger_than_crop"
        ]

    def test_visible_fraction_measures_both_axes(self) -> None:
        crop = CropBox(0.4, 0.0, 0.2, 1.0)
        inside = Observation(seconds=0, centre_x=0.5, centre_y=0.5, width=0.1, height=0.5)
        outside = Observation(seconds=0, centre_x=0.9, centre_y=0.5, width=0.1, height=0.5)
        assert visible_fraction(crop, inside) == pytest.approx(1.0)
        assert visible_fraction(crop, outside) == pytest.approx(0.0)


def test_framing_intents_are_all_known_to_the_builders() -> None:
    for intent in INTENT_NAMES:
        path = build_zoom_path(
            source_aspect=WIDE, target_aspect=TALL, duration_seconds=2.0,
            direction="push_in", energy="calm", budget=0.6, framing=intent,
        )
        assert path.keyframes


def test_a_subject_wider_than_the_delivery_is_recorded_under_every_move() -> None:
    """The fit check belongs to the clip, not to one branch of the dispatch.

    It lived inside the hold branch, so a wordmark 0.88 of a 16:9 frame wide
    was swept across when the planner asked for a hold and silently cropped to
    "Galaxy Unpac" when it asked for a push -- same material, same impossible
    promise, one of them unrecorded. None of the five path builders except
    build_crop_path reports fit, so nothing else caught it either.
    """

    import inspect

    from montagewright import pipeline

    source = inspect.getsource(pipeline.follow_subjects)
    fact_at = source.index("subject_wider_than_delivery")
    # Every branch that dispatches on what the shot is, whatever those
    # happen to be called today. This named the two-subject pan branch,
    # which has since been deleted -- the property is "before all of them",
    # not "before these two". Guards that only skip or warn do not count.
    # Indentation is not the property. The loop body has been wrapped in a
    # try since this was written -- everything moved four spaces right and
    # the check silently found nothing to enforce.
    branches = [
        line
        for line in source.splitlines()
        if line.strip().startswith("if ")
        and ("move ==" in line or "move in " in line or "reframe.looks" in line)
    ]
    assert len(branches) >= 3, branches
    for branch in branches:
        assert source.index(branch) > fact_at, (
            f"the fit check must run before {branch!r} dispatches, or that "
            "move delivers a clipped subject with nothing in the report"
        )


def test_a_subject_that_cannot_fit_says_so_before_it_is_chosen() -> None:
    """`must_be_whole` is only answerable if the ceiling travels with the subject."""

    from montagewright.cli import _subject_line
    from montagewright.clipcard import SubjectBox

    wide = SubjectBox(
        label="寫著 Galaxy Unpacked 的螢幕畫面",
        centre_x=0.52, centre_y=0.49, width=0.88, height=0.81, moves=False,
    )
    line = _subject_line(wide, WIDE, TALL)
    assert "36%" in line, line

    small = SubjectBox(
        label="硬幣", centre_x=0.5, centre_y=0.5, width=0.10, height=0.2,
        moves=False,
    )
    # No fraction, because this one fits whole. Whether it moves is still
    # said: that question has an answer for every subject, not only the ones
    # the delivery frame cannot hold.
    fits = _subject_line(small, WIDE, TALL)
    assert fits.startswith("硬幣（")
    assert "%" not in fits


def test_a_pan_onto_a_small_subject_centres_it_rather_than_hugging_an_edge() -> None:
    """Both edge bounds exist for a subject that fills the crop.

    A folded Flip 0.19 of the frame wide, panned to inside a 0.316 crop,
    cannot touch both edges: "do not travel past the far edge" and "do not
    start far from the near edge" describe an empty interval. Resolving that
    by taking one bound put the phone a tenth of the way in with a third of
    the frame wall behind it, which reads on screen as the pan overshooting.
    """

    path = build_handoff_path(
        source_aspect=WIDE,
        target_aspect=TALL,
        duration_seconds=4.0,
        from_centre=0.28,
        to_centre=0.69,
        from_width=0.20,
        to_width=0.19,
    )
    end = path.keyframes[-1].crop
    where = (0.69 - end.x) / end.width
    assert 0.4 <= where <= 0.6, (
        f"the destination sits at {where:.2f} of the frame, not centred"
    )


def test_rhythm_is_told_the_length_it_is_dividing_up() -> None:
    """Eight lengths decided in isolation summed to 26s of a 45s film."""

    import inspect

    from montagewright import planner

    source = inspect.getsource(planner.decide_rhythm)
    assert "target_seconds" in source
    assert "定調偏好" in source
    assert "duration_mode" in source


def test_selection_is_told_that_shot_count_is_a_length_decision() -> None:
    """Told only "35 seconds", selection picked sixteen shots.

    Every length downstream then had to be two seconds, which fits a static
    product view and does not fit a gesture playing out or a screen being
    read -- so the film hit its target duration by cutting away from more
    things sooner.
    """

    from montagewright.planner import PROMPTS

    prompt = (PROMPTS / "selection_zh-TW.txt").read_text(encoding="utf-8")
    assert "顆數與長度是同一個決定" in prompt
    assert "seconds_needed" in prompt


def test_the_layer_that_picks_a_shot_says_how_long_it_needs() -> None:
    """Length started from a constant, not from the shot.

    Every clip left selection with a flat four-second window, so the layer
    that knew what the shot was for had no say in how long it ran, and the
    layer that set the length began from a number nobody chose.
    """

    import inspect

    from montagewright import cli
    from montagewright.planner import _selection_schema

    shot = _selection_schema(["C1"])["properties"]["shots"]["items"]
    assert "seconds_needed" in shot["required"]

    source = inspect.getsource(cli._edl_from_selection)
    assert "seconds_needed" in source
    assert "start + 4.0" not in source


def test_a_move_floor_reports_rather_than_lengthens() -> None:
    """The length is the planner's; the floor says what could not happen.

    A flat 2.5s raised any pan to 2.5s, which reads as a safeguard and is a
    length decision made by a constant -- how long a sweep needs depends on
    how far it travels and what is on the way, and only the layer that
    watched the shot knows that. The menu now says local code will not add
    time, so it must not.
    """

    import inspect

    from montagewright import grounding

    source = inspect.getsource(grounding._requested_duration)
    assert "MOVE_FLOORS" not in source, (
        "the requested length must come back whole, floor applied nowhere"
    )
    assert "move_too_short" in inspect.getsource(grounding.ground_timeline)


def test_the_menu_hands_the_timing_judgement_to_the_planner() -> None:
    from montagewright.capabilities import describe_for_prompt

    menu = describe_for_prompt()
    assert "seconds_needed" in menu
    # The property, not the wording: length is the planner's call and the
    # executor reports a shortfall rather than quietly padding it. This
    # asserted one sentence verbatim and broke when the menu was rewritten
    # around looks, though nothing it guards had changed.
    assert "照實記一筆" in menu or "照實回報" in menu
    assert "只有看過這顆畫面的人知道" in menu
    assert "至少" not in menu


def test_the_shot_reviewer_sees_the_shot_and_settles_its_degradations() -> None:
    """Adjudication rested on a viewer who never saw the shot in question.

    "The subject is 0.88 of frame wide and can show 36% of itself" is not
    judgeable from a thirty-second film and a line of numbers. Whoever
    watched that one shot settles it.
    """

    from montagewright.review import adjudicate
    from montagewright.schema import DegradationStep, Issue, ReviewVerdict

    step = DegradationStep(
        clip_id="k00",
        ladder="other",
        ladder_other="subject_wider_than_delivery",
        trigger="wider than any crop at the delivery aspect",
        measured={"subject_width_vw": 0.88},
    )
    silent = ReviewVerdict(verdict="approve", overall="", issues=[])

    kept = adjudicate([step], silent, {"k00": {
        "degradation_verdict": "acceptable", "note": "字完整讀得到",
    }})
    assert kept[0].adjudication == "accept"
    assert "字完整讀得到" in kept[0].adjudication_reason

    sent_back = adjudicate([step], silent, {"k00": {
        "degradation_verdict": "replan", "note": "字被裁掉一角",
    }})
    assert sent_back[0].adjudication == "replan"

    # No shot verdict: the whole-cut reviewer's silence still decides.
    assert adjudicate([step], silent, {})[0].adjudication == "accept"


def test_segments_survive_the_render_so_they_can_be_reviewed() -> None:
    import inspect

    from montagewright import pipeline

    assert "keep_segments=True" in inspect.getsource(pipeline.run)


def test_the_report_says_why_a_degradation_was_settled() -> None:
    """"replan" with no grounds leaves the reader where the reviewer was."""

    import inspect

    from montagewright import cli

    assert "adjudication_reason" in inspect.getsource(cli._write_report)


def test_replanning_is_a_new_plan_rather_than_a_softer_fallback() -> None:
    """A ladder answers a failed shot with a less obvious version of itself.

    Push less far, sweep more slowly, crop a little wider -- none of those
    ask why the shot failed. A coin that fell outside the frame is not
    recovered by a gentler push; it wants a different take, a different
    subject, or the admission that the shot was about the handset edge.
    """

    from montagewright.planner import PROMPTS

    prompt = (PROMPTS / "replan_zh-TW.txt").read_text(encoding="utf-8")
    assert "不是把原本的做法縮水" in prompt
    assert "換成一顆可交付" in prompt
    # It must also be able to stand its ground: the shot reviewer sees one
    # shot with no context, and "swept past without stopping to be read" is
    # sometimes exactly what was wanted.
    assert "也可能是規劃本來就沒問題" in prompt


def test_a_sweep_is_judged_against_its_own_intent_not_legibility() -> None:
    """Reading the text is one valid outcome of a pan, not the only one.

    Leading the eye across a wall to open a scene is a different job from
    letting a viewer read the wall, and a reviewer holding every sweep to
    the second standard sends back shots that did what they meant to.
    """

    from montagewright.planner import PROMPTS

    prompt = (PROMPTS / "shotreview_zh-TW.txt").read_text(encoding="utf-8")
    assert "判準來自這顆自己的宣告，不是一套通用標準" in prompt
    assert "本來就不是問題" in prompt or "完全不是問題" in prompt


def test_the_executor_does_not_swap_the_move_it_was_given() -> None:
    """A substitution the planner cannot see is a decision it cannot argue with.

    A replan chose hold for a wide title, reasoning that travelling across it
    was what cut it in the first place -- and the executor swapped the hold
    for a sweep, so the next review described a sweep across a clipped title
    and the loop spent a round fighting itself.
    """

    import inspect

    from montagewright import pipeline

    source = inspect.getsource(pipeline.follow_subjects)
    assert "build_sweep_path" not in source.split('if move == "hold"')[1], (
        "the hold branch must hold; the fit is already recorded"
    )
    assert "subject_wider_than_delivery" in source


def test_must_be_whole_is_described_as_a_requirement_not_a_lever() -> None:
    """The planner replanned two shots by only setting this flag.

    Its stated reasoning was that the crop engine would scale to preserve
    the whole wordmark. Nothing does that -- shrinking a subject to fit is
    pillarboxing, which this pipeline will not do -- so the flag produced a
    record and an identical frame, and the loop spent a round on it twice.
    """

    from montagewright.planner import PROMPTS, _selection_schema

    # Lives on each look now rather than on the shot: a shot can settle on
    # a wordmark that must be whole and then on a face that need not be.
    described = _selection_schema(["C1"])["properties"]["shots"]["items"][
        "properties"
    ]["looks"]["items"]["properties"]["must_be_whole"]["description"]
    # States a requirement, does not fulfil one: the crop will not shrink a
    # subject to fit, so the flag is a declaration and not a lever.
    assert "宣告不是開關" in described

    # The declaration lives once, in the schema above. The selection prompt
    # no longer repeats it -- it carries the judgement the schema cannot: the
    # measured ceiling on how much of a subject this source can ever show.
    prompt = (PROMPTS / "selection_zh-TW.txt").read_text(encoding="utf-8")
    assert "只能露出" in prompt and "填 false" in prompt
    replan = (PROMPTS / "replan_zh-TW.txt").read_text(encoding="utf-8")
    assert "不是一個做法" in replan


def test_no_pass_rations_its_output_ceiling() -> None:
    """A flat ceiling truncates the whole pass, not its tail.

    Twenty-two shots stopped mid-token at 8192 and the run died: the answer
    is one decision plus one sentence per shot, so it scales with the cut.
    """

    import inspect

    from montagewright import planner

    source = inspect.getsource(planner.decide_rhythm)
    assert "MAX_OUTPUT_TOKENS" in source
    assert '"max_output_tokens": 8192' not in source
    # Billing is on tokens produced, so rationing the ceiling bought nothing
    # and cost a whole pass. One generous constant, everywhere.
    assert planner.MAX_OUTPUT_TOKENS >= 32768


def test_an_action_beat_has_to_be_long_enough_to_be_one() -> None:
    """Twenty-six of forty-one beats in one library were under 0.25s.

    "The models rotate their phones, 0.02 to 0.06s" is not a span, and the
    two fields carrying every timestamp in the schema were the only ones
    with no description telling the model what a good answer looks like.
    Downstream, in-points were snapped onto those numbers.
    """

    from montagewright.clipcard import action_beats, card_schema

    entry = card_schema()["properties"]["action"]["items"]["properties"]
    assert entry["to"].get("description"), (
        "the field that carries the timing must say what it wants"
    )
    # And it is a clock reading, like every other time on the card. One
    # notation, so `1:53` cannot arrive as 1.53 and have to be guessed at.
    assert entry["from"]["type"] == "string" and entry["to"]["type"] == "string"

    card = {"action": [
        {"what": "翻轉手機", "from": "0:00", "to": "0:00"},
        {"what": "手伸進畫面", "from": "0:02", "to": "0:03"},
    ]}
    kept = action_beats(card)
    assert [beat.what for beat in kept] == ["手伸進畫面"]


def test_a_library_that_wrote_nothing_stops_the_run() -> None:
    """An empty card library is missing input, not a degradation.

    A NameError in the request took all seventy-four cards down, was
    reported as the routine "74 failed ($0.0000)" line, and the run went on
    to pick an aspect, choose sixteen shots and spend $1.80 planning a film
    out of nothing -- every downstream layer reading the absence as "these
    clips have no description" rather than as a failure.
    """

    import tempfile
    from pathlib import Path

    from montagewright.clipcard import CardLibraryEmpty, build_library

    class Refuses:
        class files:
            @staticmethod
            def upload(**_):
                raise RuntimeError("no")

    with tempfile.TemporaryDirectory() as work:
        clip = Path(work) / "a.mp4"
        clip.write_bytes(b"not really a video")
        try:
            build_library(
                {"a": clip}, Path(work) / "cards", client=Refuses()
            )
        except CardLibraryEmpty as error:
            assert "RuntimeError" in str(error), "the reason has to survive"
        else:
            raise AssertionError("an empty library must not pass silently")


def test_money_running_out_stops_the_library_instead_of_truncating_it() -> None:
    """Twenty-one of seventy-four rushes is not a library, it is an accident.

    The credits ran out at clip twenty-two, the remaining fifty-three were
    recorded as ordinary per-clip failures, and the run planned from what it
    had -- freezing an inventory of twenty-one sources as revision zero of
    the planning authority. The next run, after a top-up, described all
    seventy-four and could no longer agree with what had been written down.
    Whose money ran out says nothing about the clip that was next in line.
    """

    import tempfile
    from pathlib import Path

    from montagewright.clipcard import build_library
    from montagewright.cost import BudgetSpent

    described: list[str] = []

    class RunsOutAfterOne:
        class files:
            @staticmethod
            def upload(**_):
                if described:
                    raise BudgetSpent("Gemini Prepay credits are depleted")
                described.append("one")
                raise RuntimeError("this one is merely unreadable")

    with tempfile.TemporaryDirectory() as work:
        clips = {}
        for name in ("a", "b", "c"):
            clip = Path(work) / f"{name}.mp4"
            clip.write_bytes(f"not really a video {name}".encode())
            clips[name] = clip
        with pytest.raises(BudgetSpent):
            build_library(clips, Path(work) / "cards", client=RunsOutAfterOne())


def test_clip_cards_can_be_written_at_all() -> None:
    """The request referenced a name the module never imported."""

    import inspect

    from montagewright import clipcard

    assert "MAX_OUTPUT_TOKENS" in dir(clipcard)
    compile(inspect.getsource(clipcard), "clipcard.py", "exec")


def test_the_card_says_what_the_source_camera_does() -> None:
    """"The camera moves" does not support the decision it feeds.

    A reference cut held a static frame on the left-hand handset and let the
    take's own move bring a third one in from the right. Choosing that needs
    to know what the move reveals; a boolean cannot say, and the planner was
    left to either ignore the movement or lay a digital one over it.
    """

    from montagewright.clipcard import card_schema
    from montagewright.planner import MaterialItem, _describe_material

    schema = card_schema()
    assert "camera_motion" in schema["required"]

    described = _describe_material([
        MaterialItem(
            source_id="C1",
            duration_seconds=9.0,
            summary="兩台摺疊機並排",
            camera_moves=True,
            camera_motion="往右平移，右邊會有第三台手機進畫面",
        )
    ])
    assert "第三台" in described, described


def test_a_card_box_is_only_reused_when_nothing_moved() -> None:
    """The box says where, and said nothing about when.

    `moves` was parsed off every subject and read by no one, so a held frame
    on a take whose camera pans was aimed at wherever the card happened to
    look -- and the subject walked out of it. The card knows which kind of
    shot this is; the reuse now depends on it.
    """

    import inspect

    from montagewright import pipeline
    from montagewright.clipcard import card_schema, subjects_from_card

    box = card_schema()["properties"]["subjects"]["items"]
    assert "seen_at" in box["required"], "a position needs its moment"

    parsed = subjects_from_card({"subjects": [{
        "label": "手機", "centre_x": 0.5, "centre_y": 0.5,
        "width": 0.2, "height": 0.4, "moves": True, "seen_at": "0:01",
    }]})
    assert parsed[0].at_seconds == 1.0
    assert parsed[0].moves

    source = inspect.getsource(pipeline.follow_subjects)
    assert "not box.moves" in source, (
        "a moving subject must be measured over the shot, not reused"
    )


def test_takes_set_aside_before_planning_say_why() -> None:
    """A run working from sixty-six of seventy-four looked like a full one.

    The card gives a reason a take failed and the filter dropped it, so
    "why wasn't the good coin shot used" had no answer anywhere in the
    output -- not in the report, not on the console.
    """

    import inspect

    from montagewright import cli

    source = inspect.getsource(cli.command_render)
    assert "unusable_reason" in source
    assert "set_aside" in inspect.getsource(cli._write_report)


def test_the_web_run_reports_what_it_decided_not_just_the_file() -> None:
    """The question after a run is which take that is and why it looks like that."""

    from montagewright.webapp import PAGE, create_app

    paths = {r.path for r in create_app().routes if hasattr(r, "path")}
    assert {"/api/runs", "/api/runs/{run_id}/video",
            "/api/runs/{run_id}/shot/{index}"} <= paths

    # What each shot is and why it looks like that. These used to be column
    # headings; the table became rows when it turned out seven columns of
    # very different lengths made every shot six hundred pixels tall.
    page = PAGE.read_text(encoding="utf-8")
    for shown in (
        "shotcard", "s.source_id", "b.camera_move", "s.subject", "s.why",
        "motion.source", "motion.digital", "tellDegradation", "實際做到什麼",
    ):
        assert shown in page, shown


def test_a_track_can_be_measured_without_a_reviewed_lock() -> None:
    """Requiring a lock file first turns "see what this does" into an errand.

    The lock proves which analysis a delivery was cut against, which is worth
    its ceremony for a delivery and is pure friction for a trial. Downbeats
    and section boundaries are derived while locking, so deriving them here
    too is what keeps "land on the chorus" resolvable.
    """

    import inspect

    from montagewright import cli, grounding

    source = inspect.getsource(grounding.analyse_track)
    assert "downbeat" in source and "section_boundary" in source
    assert "analyse_track" in inspect.getsource(cli.command_render)


def test_a_transcript_is_its_own_card() -> None:
    """Subtitling something finished has nothing to do with cutting it.

    A clip card describes what a take looks like and is cached because that
    stays true. Speech is only worth paying for when it matters, and it is
    useful with no edit at all -- so it is a separate artifact, not more
    fields on the clip card.
    """

    import inspect

    from montagewright import clipcard, transcript

    assert "transcript" not in clipcard.card_schema()["properties"]
    assert transcript.CARD_VERSION != clipcard.CARD_VERSION

    fields = transcript._schema()["properties"]
    assert "language" in fields, "the locale was a guess; this is the answer"
    assert "speaker" in fields["lines"]["items"]["required"], (
        "a talking shot framed on whoever is not talking is the fault this "
        "is here to make fixable"
    )
    # What the recogniser said still has to survive, or a correction is
    # invisible -- but it is not the model's to write. Asked to correct an
    # error and quote it unchanged in one breath it corrects both, and did:
    # it reported 髮 where the recogniser had said 發, erasing the only
    # evidence the field carries. It is filled locally from the stored words.
    written = fields["lines"]["items"]["properties"]
    assert "heard" not in written
    assert transcript.Line(
        text="x", heard="y", starts_seconds=0.0, ends_seconds=1.0
    ).corrected
    assert "what_was_heard(" in inspect.getsource(transcript.describe)

    # Nor are the times. `across_lines` puts the corrected text back on the
    # recogniser's own per-word clock and never reads a model timestamp, so
    # asking for one only makes it reconcile its text against a number it
    # invented -- and the text is the half being kept.
    assert "starts_seconds" not in written and "ends_seconds" not in written
    assert "across_lines(" in inspect.getsource(transcript.describe)


def test_spoken_boundaries_land_on_a_measured_break() -> None:
    """The model hears where a sentence ends; the recogniser marked where the
    sound broke. Neither alone puts the cut in the right place."""

    from montagewright.transcript import Word, gaps, snap

    words = [
        Word("溼", 0.0, 0.30), Word("了", 0.30, 0.62),
        Word("。", 0.62, 1.01),
        Word("回", 1.01, 1.66), Word("家", 1.66, 1.98),
    ]
    breaks = gaps(words)
    assert breaks == [1.01]
    assert snap(1.2, breaks) == 1.01
    # Too far to be the same boundary: a model second that lands nowhere near
    # a break is left alone rather than dragged across a word.
    assert snap(3.0, breaks) == 3.0


def test_every_upload_waits_until_the_file_can_be_used() -> None:
    """An upload returns before the service has finished with it.

    The cached path waited; the uncached branch in five other modules did
    not, so it worked on short clips and failed on the first long one with
    "not in an ACTIVE state".
    """

    import re
    from pathlib import Path

    for module in Path("src/montagewright").glob("*.py"):
        if module.name == "uploads.py":
            continue
        text = module.read_text(encoding="utf-8")
        assert not re.search(r"client\.files\.upload\(", text), (
            f"{module.name} uploads without waiting; use upload_now"
        )


def test_the_transcriber_is_reachable_as_its_own_command() -> None:
    from montagewright.cli import main

    try:
        main(["transcribe", "--help"])
    except SystemExit as exit_code:
        assert exit_code.code == 0


def test_music_goes_under_a_voice_rather_than_over_it() -> None:
    """Throwing the source audio away is right for b-roll and ruins an
    interview, where what was said is the whole content.

    A fixed lower level is not the answer either: quiet enough never to bury
    a sentence is too quiet to be doing anything in the gaps. Measured on one
    street interview, the voice as sidechain trigger pulled the bed down 8.2
    dB against the same bed with a silent trigger.
    """

    import inspect

    from montagewright import renderer

    source = inspect.getsource(renderer._mux_music)
    assert "sidechaincompress" in source
    assert "keep_voice" in inspect.signature(renderer.render).parameters
    # The voice has to reach the output, not only the compressor's key input.
    assert "asplit" in source and "[voice][ducked]amix" in source


def test_a_transcript_is_only_paid_for_where_speech_is_the_content() -> None:
    """A transcript costs a call and a minute a clip.

    On b-roll it answers a question nobody asked, and a flag somebody has to
    remember is a flag somebody forgets -- so the card, which already watched
    the clip with its audio, says which clips need one.
    """

    import inspect

    from montagewright import cli
    from montagewright.clipcard import card_schema

    speech = card_schema()["properties"]["speech"]
    assert speech["enum"] == ["none", "ambient", "content"]
    assert "speech" in card_schema()["required"]

    source = inspect.getsource(cli.command_render)
    assert '"speech") == "content"' in source
    assert "keep_voice=bool(transcripts)" in source


def test_the_planner_sees_the_sentences_it_is_choosing_between() -> None:
    """A window out of an interview is chosen because of a sentence."""

    from montagewright.cli import _speech_lines
    from montagewright.planner import MaterialItem, _describe_material
    from montagewright.transcript import CARD_VERSION

    lines = _speech_lines({
        "version": CARD_VERSION,
        "lines": [{
            "text": "夏天最崩潰的是流汗完又下雨",
            "speaker": "穿灰藍色T恤的受訪男子",
            "starts_seconds": 4.0, "ends_seconds": 9.2,
        }],
    })
    assert "穿灰藍色T恤的受訪男子" in lines[0]
    assert "4.0-9.2s" in lines[0]

    described = _describe_material([
        MaterialItem(source_id="S01", duration_seconds=70.0,
                     summary="街訪", speech=lines)
    ])
    assert "說了什麼" in described and "流汗完又下雨" in described


def test_adjacent_asr_lines_become_one_canonical_continuous_soundbite() -> None:
    from montagewright.cli import _audio_spans, _speech_lines
    from montagewright.transcript import CARD_VERSION

    card = {
        "version": CARD_VERSION,
        "lines": [
            {
                "text": "就流很多汗很麻煩，尤其是吹頭髮就很熱，",
                "speaker": "短髮受訪女子",
                "starts_seconds": 2.46, "ends_seconds": 8.52,
            },
            {
                "text": "然後還要吹那熱風會流汗。",
                "speaker": "短髮受訪女子",
                "starts_seconds": 8.70, "ends_seconds": 10.86,
            },
            {
                "text": "那還有其他的嗎？",
                "speaker": "主持人",
                "starts_seconds": 10.86, "ends_seconds": 12.30,
            },
        ],
    }

    spans = _audio_spans({"S01": card})
    grouped = spans["S01:t00-t01"]
    assert grouped["in_seconds"] == 2.46
    assert grouped["out_seconds"] == 10.86
    assert grouped["line_ids"] == ["S01:t00", "S01:t01"]
    assert grouped["kind"] == "continuous_turn"
    assert "S01:t00-t02" not in spans  # a speaker change cannot be spliced in
    described = _speech_lines("S01", card)
    assert any("`S01:t00-t01`" in one and "連續多行" in one for one in described)


def test_continuous_soundbites_split_at_long_pauses_and_duration_cap() -> None:
    from montagewright.cli import _audio_spans_for_source
    from montagewright.transcript import CARD_VERSION

    def line(text: str, start: float, end: float) -> dict:
        return {
            "text": text, "speaker": "受訪者",
            "starts_seconds": start, "ends_seconds": end,
        }

    card = {
        "version": CARD_VERSION,
        "lines": [
            line("一", 0.0, 4.0), line("二", 4.1, 8.0),
            line("三", 8.1, 15.0),  # exceeds the 14-second group cap
            line("四", 17.0, 19.0),  # a 2-second pause starts another run
            line("五", 19.1, 21.0),
        ],
    }
    spans = _audio_spans_for_source("S02", card)
    assert "S02:t00-t01" in spans
    assert "S02:t00-t02" not in spans
    assert "S02:t03-t04" in spans


def test_grouped_soundbite_resolves_to_one_source_window_in_the_edl(tmp_path) -> None:
    from montagewright.cli import _edl_from_selection
    from montagewright.transcript import CARD_VERSION

    transcript = {
        "version": CARD_VERSION,
        "lines": [
            {
                "text": "前半句，", "speaker": "受訪者",
                "starts_seconds": 4.0, "ends_seconds": 5.5,
            },
            {
                "text": "後半句。", "speaker": "受訪者",
                "starts_seconds": 5.6, "ends_seconds": 7.0,
            },
        ],
    }
    selection = {
        "shots": [{
            "span_id": "P:s00", "source_id": "P", "start_seconds": 0,
            "seconds_needed": 4, "frame": "settles", "energy": "medium",
            "why": "B-roll", "audio_role": "discard",
            "looks": [{"at": "centre", "seconds": 4, "framing": "thirds"}],
        }],
        "audio_assignments": [{
            "audio_span_id": "V:t00-t01", "starts_at_shot_index": 0,
            "offset_seconds": 0, "completion": "complete_thought",
            "gain_db": 0, "why": "完整回答",
        }],
    }

    edl, _ = _edl_from_selection(
        selection, tmp_path, cards={}, transcripts={"V": transcript}
    )
    audio = edl.audio_clips[0]
    assert (audio.source_id, audio.in_seconds, audio.out_seconds) == ("V", 4.0, 7.0)
    assert len(edl.audio_clips) == 1


def test_speaker_picture_is_locked_to_the_narrative_source_clock(tmp_path) -> None:
    from montagewright.cli import _edl_from_selection
    from montagewright.transcript import CARD_VERSION

    transcript = {
        "version": CARD_VERSION,
        "lines": [{
            "text": "嘴型必須對上這一句。", "speaker": "受訪者",
            "starts_seconds": 4.25, "ends_seconds": 7.25,
        }],
    }
    selection = {
        "shots": [{
            "span_id": "V:s00", "source_id": "V", "start_seconds": 20,
            "seconds_needed": 3, "frame": "settles", "energy": "medium",
            "why": "講者", "audio_role": "discard",
            "audio_completion": "none", "picture_role": "speaker",
            "looks": [{"at": "centre", "seconds": 3, "framing": "thirds"}],
        }],
        "audio_assignments": [{
            "audio_span_id": "V:t00", "starts_at_shot_index": 0,
            "offset_seconds": 0, "completion": "complete_thought",
            "gain_db": 0, "why": "完整回答",
        }],
    }

    edl, notes = _edl_from_selection(
        selection, tmp_path, cards={}, transcripts={"V": transcript}
    )

    assert edl.clips[0].approx_in_seconds == 4.25
    assert edl.clips[0].approx_out_seconds == 7.25
    assert "source clock aligned" in notes["k00"]


def test_speaker_picture_cannot_pretend_another_sources_voice_is_synced(tmp_path) -> None:
    import pytest
    from montagewright.cli import _edl_from_selection
    from montagewright.transcript import CARD_VERSION

    transcript = {
        "version": CARD_VERSION,
        "lines": [{
            "text": "另一個人說的話。", "speaker": "受訪者",
            "starts_seconds": 1, "ends_seconds": 3,
        }],
    }
    selection = {
        "shots": [{
            "span_id": "PICTURE:s00", "source_id": "PICTURE",
            "start_seconds": 0, "seconds_needed": 2,
            "frame": "settles", "energy": "medium", "why": "錯的人",
            "audio_role": "discard", "audio_completion": "none",
            "picture_role": "speaker",
            "looks": [{"at": "centre", "seconds": 2, "framing": "thirds"}],
        }],
        "audio_assignments": [{
            "audio_span_id": "VOICE:t00", "starts_at_shot_index": 0,
            "offset_seconds": 0, "completion": "complete_thought",
            "gain_db": 0, "why": "不相符",
        }],
    }
    with pytest.raises(ValueError, match="speaker picture.*narrative audio"):
        _edl_from_selection(
            selection, tmp_path, cards={}, transcripts={"VOICE": transcript}
        )


def test_speaker_can_return_after_broll_on_the_continuing_audio_clock() -> None:
    """A/B-roll/A does not restart or detach the speaker's lip sync."""

    from montagewright.pipeline import align_speaker_pictures_to_audio
    from montagewright.schema import AudioClip, Clip, EDL

    edl = EDL(
        project_id="answer-over-broll",
        clips=[
            Clip(
                clip_id="k00", source_id="VOICE", approx_in_seconds=2,
                approx_out_seconds=4, picture_role="speaker",
            ),
            Clip(
                clip_id="k01", source_id="BROLL", approx_in_seconds=20,
                approx_out_seconds=22, picture_role="illustrative_broll",
            ),
            Clip(
                clip_id="k02", source_id="VOICE", approx_in_seconds=40,
                approx_out_seconds=42, picture_role="speaker",
            ),
        ],
        audio_clips=[AudioClip(
            audio_id="a00", source_id="VOICE", in_seconds=10,
            out_seconds=16, starts_at_clip_id="k00", role="narrative",
            completion="complete_thought",
        )],
    )

    aligned, notes = align_speaker_pictures_to_audio(edl)

    assert aligned.clips[0].approx_in_seconds == pytest.approx(10)
    assert aligned.clips[1].approx_in_seconds == pytest.approx(20)
    assert aligned.clips[2].approx_in_seconds == pytest.approx(14)
    assert any("k02" in note and "14.000" in note for note in notes)


def test_transcribed_selection_requires_narrative_on_the_independent_track() -> None:
    from montagewright.planner import _selection_schema

    schema = _selection_schema(
        ["S:s00"], min_shots=1, max_shots=2,
        audio_span_ids=["S:t00-t01"],
    )
    shot = schema["properties"]["shots"]["items"]["properties"]
    assert "narrative" not in shot["audio_role"]["enum"]
    assignment = schema["properties"]["audio_assignments"]["items"]
    assert assignment["properties"]["audio_span_id"]["enum"] == ["S:t00-t01"]


def test_a_cut_that_never_asked_for_a_beat_is_not_a_missed_one() -> None:
    """A speech-led cut read as 0/13 aligned.

    Thirteen shots, every one deliberately off the grid so a sentence could
    finish, and the fallback for "no rhythm pass ran" turned that into total
    failure in the one line anyone reads.
    """

    from montagewright.pipeline import Report

    speech = Report(total_cuts=13, aligned_cuts=0)
    speech.rhythm_decisions = {
        f"k{i:02d}": {"cut_on_beat": False} for i in range(13)
    }
    assert "0/0 cuts on a musical event (13 content-led by choice)" in (
        speech.summary()
    )

    silent = Report(total_cuts=4, aligned_cuts=4)
    assert "4/4 cuts on a musical event" in silent.summary()


def test_an_already_cut_file_is_opened_along_its_own_boundaries() -> None:
    """One file holding many takes is not one take.

    Handed over whole it becomes one card describing five minutes, one
    transcript, and a planner choosing windows out of a single source as
    though the cuts inside it were not there. A continuous take comes back
    as itself, which is the honest answer for a locked-off interview.
    """

    import inspect

    from montagewright import cli, grounding

    assert "shots_in" in inspect.getsource(cli.command_render)
    source = inspect.getsource(grounding.shots_in)
    assert "scene" in source
    # The split pieces keep the name they came from, so a report traces back.
    assert '{path.stem}-{index:02d}' in inspect.getsource(cli.command_render)


def test_music_is_not_required_to_make_a_cut() -> None:
    """Refusing to run without a bed was the tool deciding on the way in.

    A cut carried by what people say does not need one, and without a grid
    every length is content-led -- which is what a speech cut wants.
    """

    import inspect

    from montagewright import cli

    source = inspect.getsource(cli.command_render)
    assert "--music or --music-map is required" not in source
    assert "lengths will be led by content" in source


def test_a_chinese_filename_can_be_uploaded() -> None:
    """The uploader puts the name in a header, and a header is latin-1.

    Every clip in a folder named in Chinese -- which is most of what this is
    pointed at -- failed with UnicodeEncodeError, and the material's own name
    is not part of the bytes being sent.
    """

    import tempfile
    from pathlib import Path

    from montagewright.uploads import _ascii_named

    work = Path(tempfile.mkdtemp())
    chinese = work / "夏日街訪_夏天最崩潰的事-00.mp4"
    chinese.write_bytes(b"x" * 64)
    with _ascii_named(chinese) as sendable:
        sendable.name.encode("ascii")
        assert sendable.suffix == ".mp4"
        assert sendable.read_bytes() == chinese.read_bytes()

    plain = work / "C8371.MP4"
    plain.write_bytes(b"y")
    with _ascii_named(plain) as sendable:
        assert sendable == plain, "an ASCII name needs no detour"


def test_a_timeline_is_written_only_when_asked_for() -> None:
    """Most runs want a file. A timeline is for the run where somebody
    intends to open it and disagree with one shot."""

    import inspect

    from montagewright import cli

    source = inspect.getsource(cli.command_render)
    assert 'args.timeline != "none"' in source
    parser_source = inspect.getsource(cli.main)
    assert '"--timeline"' in parser_source and 'default="none"' in parser_source


def test_a_timeline_carries_the_reasons_and_the_original_media() -> None:
    """Handles and reasons both existed and neither could be used.

    Every segment renders with half a second either side for exactly this
    and nothing consumed it; every shot carries why it was chosen, in a
    debugging artifact no editor reads. Referencing the original source is
    what makes the handle a thing you can drag.
    """

    from pathlib import Path

    from montagewright.executor import CropBox, RenderPlan, Segment, Source
    from montagewright.timeline import to_fcpxml, to_xmeml

    source = Source(
        source_id="S00", path=Path("/tmp/夏日街訪-00.mp4"),
        duration_seconds=48.0, width=1920, height=1080,
    )
    plan = RenderPlan(project_id="t", segments=[
        Segment(clip_id="k00", source=source, in_seconds=4.0,
                out_seconds=7.0, crop=CropBox(0.34, 0.0, 0.3164, 1.0))
    ])
    report = {
        "selection": {"shots": [{"why": "受訪者回答核心問題",
                                 "camera_move": "hold"}]},
        "rhythm": {"k00": {"why": "讓句子講完"}},
        "shots": {"k00": {"delivered": True, "note": "框住講話的人"}},
        "degradations": [],
    }
    for build in (to_xmeml, to_fcpxml):
        xml = build(plan, report, name="cut", width=1080, height=1920)
        assert "受訪者回答核心問題" in xml, "the reason has to travel"
        assert "讓句子講完" in xml
        # The original file, not the rendered segment: trimming outward is
        # only possible against material the timeline can still reach.
        assert "-00.mp4" in xml and "segments" not in xml


def test_rhythm_is_decided_whether_or_not_there_is_music() -> None:
    """This asserted the opposite, on reasoning that turned out to be wrong.

    "Its whole job is reconciling a length against a track" -- but what it
    reconciles is the sequence against itself. Gated on having a grid, a film
    with no music had nothing deciding its pacing at all: every length was
    whatever selection guessed for that shot alone, and nothing ever asked
    whether eight in a row had a shape. Speech-led cuts, which need shaping
    most, got none of it.
    """

    import inspect

    from montagewright import pipeline

    source = inspect.getsource(pipeline.run)

    assert "decide_rhythm_first and grid is not None" not in source
    assert "if decide_rhythm_first:" in source


def test_music_is_measured_before_direction_and_reused_after_selection() -> None:
    """Direction, selection and execution must share one measured clock."""

    import inspect

    from montagewright import cli, planner

    command = inspect.getsource(cli.command_render)
    measured = command.index("grid = analyse_track(args.music)")
    directed = command.index("direction, usage_direction = decide_direction(")
    selected = command.index("selection, usage_selection = select_shots(")
    assert measured < directed < selected
    assert command.count("music_grid=grid") >= 2
    assert "music_grid: BeatGrid | None = None" in inspect.getsource(
        planner.select_shots
    )


def test_a_film_with_no_track_is_told_so_rather_than_shown_an_empty_grid() -> None:
    from montagewright.planner import decide_rhythm
    import inspect

    said = inspect.getsource(decide_rhythm)

    assert "這支片沒有配樂" in said
    assert "沒有拍點要對" in said


def test_how_music_sits_under_speech_is_decided_not_computed() -> None:
    """A compressor ducks on signal and knows nothing about the film.

    On a cut that is speech end to end that means the bed climbs into every
    breath and is pushed down by the next line -- busier than sitting behind
    it steadily. Which of the two a film wants is an editorial call.
    """

    import inspect

    from montagewright import renderer
    from montagewright.planner import PROMPTS, _direction_schema

    field = _direction_schema()["properties"]["music_under_speech"]
    assert field["enum"] == ["bed", "duck", "none"]
    assert "music_under_speech" in _direction_schema()["required"]

    prompt = (PROMPTS / "direction_zh-TW.txt").read_text(encoding="utf-8")
    assert "music_under_speech" in prompt

    source = inspect.getsource(renderer._mux_music)
    assert 'under_speech == "bed"' in source
    assert "sidechaincompress" in source, "ducking is still available"


def test_the_bed_is_placed_under_the_voice_not_under_the_music() -> None:
    """"The music minus 12" lands wherever that track was mastered.

    A mastered track sits near -12 dBFS and a street interview averages
    around -22, so a fixed subtraction put the bed at exactly the level of
    the speech it was meant to be beneath, and the reviewer reported the
    voice as missing from a cut that contained it.
    """

    import inspect

    from montagewright import renderer

    assert not hasattr(renderer, "MUSIC_UNDER_VOICE_DB")
    assert renderer.BED_BELOW_VOICE_DB > 0
    source = inspect.getsource(renderer._mux_music)
    assert "_level(picture) - _level(music) - BED_BELOW_VOICE_DB" in source


def test_pauses_come_from_the_punctuation_the_recogniser_wrote() -> None:
    """The space between words is always zero; the punctuation is not.

    The transcriber segments a stream continuously, so each word's end is the
    next word's start -- ninety-five per cent of inter-word gaps in one
    interview were exactly 0.000s, and pause candidates built on them found
    four points in seventy seconds. A 。 is the recogniser saying it heard a
    break, and the token carries that break's span.
    """

    from montagewright.transcript import Word, gaps, snap_end

    words = [
        Word("行", 12.9, 13.14),
        Word("。", 13.14, 13.50),   # the break itself, 0.36s of it
        Word("對", 13.50, 13.68),
    ]
    assert gaps(words) == [13.5]
    # The out-point was landing on the last syllable's nominal end, which is
    # where the pause starts -- the word's decay is still to come.
    assert snap_end(13.14, gaps(words)) == 13.5
    # Never backwards: the pause before the final word is often the nearer one.
    assert snap_end(13.60, [13.5]) == 13.60


def test_the_direction_only_promises_what_the_tools_can_do() -> None:
    """It asked for keyword titles and jump cuts, and nothing downstream
    speaks either -- so the reviewer reported the film as failing to do what
    the film had asked of itself."""

    import inspect

    from montagewright import planner
    from montagewright.planner import PROMPTS

    assert "describe_for_prompt()" in inspect.getsource(planner.decide_direction)
    prompt = (PROMPTS / "direction_zh-TW.txt").read_text(encoding="utf-8")
    assert "只承諾做得到的事" in prompt


def test_cutting_before_a_sentence_ends_stays_available() -> None:
    """Snapping a card's line to the break is not a rule about the cut.

    The card records where the sentence ends, which is a fact. What the shot
    does with it is selection's -- `seconds_needed` is used as given, with no
    second snap on the way to the EDL -- so cutting away on the highest word
    and leaving the answer for the next shot is expressible. Only the prompt
    was talking anyone out of it, in words that could not tell a deliberate
    cliffhanger from a sentence hacked short to fit.
    """

    import inspect

    from montagewright import cli
    from montagewright.planner import PROMPTS

    source = inspect.getsource(cli._edl_from_selection)
    assert "snap_end" not in source, "the shot's out-point is selection's"

    prompt = (PROMPTS / "selection_zh-TW.txt").read_text(encoding="utf-8")
    assert "留懸念" in prompt
    assert "為了湊秒數砍掉半句是意外" in prompt


def test_the_voice_is_levelled_before_anything_goes_under_it() -> None:
    """One bed cannot sit under two speakers fourteen decibels apart.

    A street interview runs from a shouted answer to a mumbled one. Placed
    against the average, the bed sat comfortably under the loud speaker and
    five decibels under the quiet one -- close enough that the reviewer
    reported the voice as inaudible and, unable to hear the opening line,
    also reported the hook as missing.
    """

    import inspect

    from montagewright import renderer

    assert "speechnorm" in renderer.VOICE_LEVELLER
    source = inspect.getsource(renderer._mux_music)
    # Both paths: the steady bed and the ducked one.
    assert source.count("VOICE_LEVELLER") == 2


def test_a_replan_renders_the_same_way_the_first_pass_did() -> None:
    """The render call was written out twice and only one copy kept up.

    The first learned to keep the voice and lay the bed under it; the second,
    which runs after a replan, did not -- so any run that revised anything
    delivered the film with the speech thrown away and nothing but music
    left, having sounded correct in the round before.
    """

    import inspect

    from montagewright import cli

    source = inspect.getsource(cli.command_render)
    assert source.count("keep_voice=bool(transcripts)") == 1
    assert source.count("resolved = run(") == 0, (
        "both renders go through one definition"
    )
    # Initial render, picture-only replacement, and a joint audio+picture
    # reselection after an audio_content review all share the same helper.
    assert source.count("= cut(") == 3


def test_dropping_the_question_depends_on_something_carrying_the_premise() -> None:
    """"The answer implies the question" holds only when a title says it.

    Written as a flat rule it removed every host line from a street-interview
    Shorts that has no text cards, so the opening became a punchline with
    nothing to be the punchline of -- "來月經吧" is only startling after
    somebody asks what the worst thing about summer is.
    """

    from montagewright.planner import PROMPTS

    prompt = (PROMPTS / "selection_zh-TW.txt").read_text(encoding="utf-8")
    assert "問句通常不用剪進去" not in prompt
    assert "觀眾從哪裡知道題目" in prompt


def test_the_page_takes_a_path_and_remembers_what_it_ran() -> None:
    """Two things a local tool should not have made anyone do.

    Uploading material that is already on the disk beside the server copies
    it into the browser to write it back out a directory away -- 836 MB of it
    for one interview. And runs lived in a temp directory keyed by an
    in-memory dict, so closing the server threw away every finished cut,
    when comparing this one against the last is most of the work.
    """

    from montagewright.webapp import MAX_UPLOAD_BYTES, PAGE, RUNS_ROOT, create_app

    paths = {r.path for r in create_app().routes if hasattr(r, "path")}
    assert "/api/runs/{run_id}/transcripts" in paths
    assert MAX_UPLOAD_BYTES > 0, "an uncapped upload can fill the disk"
    assert "runs" in str(RUNS_ROOT)

    page = PAGE.read_text(encoding="utf-8")
    for control in ("source_path", "music_path", "speech", "locale"):
        assert control in page, control
    # The panes the browser column offers. They were drawer tabs with longer
    # names when the drawer ran the width of the screen.
    assert "pane-past" in page and "先前" in page
    assert "pane-tx" in page and "逐字稿" in page


def test_a_folder_can_be_clicked_instead_of_typed() -> None:
    """A browser will not hand over a real path.

    A directory picker gives relative names and nothing else, so pointing
    this at material sitting next to the server meant typing the path out.
    The server lists the folders instead -- it binds to localhost, and the
    person using it owns the disk.
    """

    from fastapi.testclient import TestClient

    from montagewright.webapp import PAGE, create_app

    listing = TestClient(create_app()).get("/api/browse").json()
    assert listing["here"], "somewhere to start from"
    assert "folders" in listing and "videos" in listing
    # The count is what makes a listing useful: it says which folder holds
    # the rushes without descending into every one of them.
    assert all("clips" in folder for folder in listing["folders"])

    page = PAGE.read_text(encoding="utf-8")
    assert "browseTo" in page and "用這個資料夾" in page


def test_every_paid_stage_checks_the_cap_before_spending() -> None:
    """"Call before dispatching, so the cap stops work rather than paying
    for it" -- and three stages did not.

    Transcription, replanning and the re-render after a replan all recorded
    against the ledger and none of them asked it first, so a run at $5.90 of
    a $6 cap would still fire a replan and land past it. The accounting was
    right; the stopping was not.
    """

    import inspect
    import re

    from montagewright import cli

    source = inspect.getsource(cli.command_render)
    # Each of these is a paid call site; each must be preceded by a check.
    for call in ("transcribe(", "replan_shots("):
        for match in re.finditer(re.escape(call), source):
            before = source[max(0, match.start() - 700):match.start()]
            assert "ledger.check()" in before, f"{call} spends unchecked"

    # Inside the render, the subject pass is one call per shot, so a cap read
    # only between stages lets a whole plan through after it is reached.
    from montagewright import pipeline

    inner = inspect.getsource(pipeline.follow_subjects)
    assert inner.count("_afford(report)") == inner.count("locate_subject(")


def test_a_pasted_path_is_taken_as_a_path() -> None:
    """Finder and browsers hand over a URL, terminals leave the quotes on.

    `Path("file:/Users/...")` is a relative directory called "file:", so a run
    started with one spent four minutes writing cards and only then failed on
    a track that had never been there.
    """

    from montagewright.webapp import _typed_path

    assert str(_typed_path("file:///Users/j/a%20b/%E5%A4%8F.mp3")) == (
        "/Users/j/a b/夏.mp3"
    )
    assert str(_typed_path("file:/Users/j/x.mp3")) == "/Users/j/x.mp3"
    assert str(_typed_path(' "/Users/j/y.mp3" ')) == "/Users/j/y.mp3"
    assert _typed_path("   ") is None


def test_the_long_silent_stage_reports_itself() -> None:
    """Seventy-four cards is four minutes with nothing on screen, which looks
    exactly like a hang."""

    import inspect

    from montagewright import cli, clipcard

    assert "progress" in inspect.signature(clipcard.build_library).parameters
    assert "card {index}/{total}" in inspect.getsource(cli.command_render)


def test_a_new_run_clears_the_last_one_from_the_page() -> None:
    """Leaving the previous result up while cards are written reads as though
    the new run had already finished."""

    from montagewright.webapp import PAGE

    page = PAGE.read_text(encoding="utf-8")
    start = page.index("runId = (await res.json()).run_id;")
    assert "classList.add('hide')" in page[start:start + 600]


def test_the_page_says_what_each_stage_is_doing() -> None:
    """"cards: 74 written" does not explain four minutes of nothing.

    The raw output is what the pipeline says to itself. Someone watching a
    run wants to know which part is happening and what that part is for --
    especially during the long silent one.
    """

    from montagewright.webapp import PAGE

    page = PAGE.read_text(encoding="utf-8")
    assert "const STEPS" in page and "paintSteps" in page
    for said in ("Apple 辨識器給逐字時間", "SAM 逐幀追出在哪裡",
                 "先逐顆對照它自己的計畫"):
        assert said in page, said
    # The machine output is still there, folded away.
    assert "原始輸出" in page


def test_music_can_be_picked_the_same_way_as_the_rushes() -> None:
    """Typing was the only way, and a path pasted from Finder is a URL."""

    from fastapi.testclient import TestClient

    from montagewright.webapp import PAGE, create_app

    audio = TestClient(create_app()).get(
        "/api/browse", params={"kind": "audio"}
    ).json()
    assert "videos" in audio  # the listing switches what it looks for
    page = PAGE.read_text(encoding="utf-8")
    assert "browse-music" in page and "openPicker" in page


def test_a_failed_run_says_so_where_it_can_be_seen() -> None:
    """A run died on the project spend cap and the page showed ticked steps.

    The traceback was in the raw output, folded away, and the stage list
    marked everything up to the failure as complete and everything after it
    as pending -- which is what a run still in progress looks like.
    """

    from montagewright.webapp import PAGE

    page = PAGE.read_text(encoding="utf-8")
    assert "showFault" in page
    assert "Google 專案的月度支出上限滿了" in page
    assert "ai.studio/spend" in page
    # The step it died on is marked, not ticked.
    assert "broke" in page and ".steps li.broke .name" in page


def test_a_run_that_died_can_be_picked_up_where_it_stopped() -> None:
    """The quota ran out after seventy-four cards and a selection.

    The cards survived because they are keyed by the content they describe.
    The direction and the selection were not kept at all, so the second
    attempt paid for them again -- and they are the same question whenever
    the material, the brief and the aspect are the same.
    """

    import inspect

    from montagewright import cli
    from montagewright.webapp import PAGE, create_app

    source = inspect.getsource(cli.command_render)
    assert '_decided(work, "direction"' in source
    assert '_decided(work, "selection"' in source
    # A different brief is a different question, not a stale answer.
    assert "brief, args.aspect" in source

    paths = {r.path for r in create_app().routes if hasattr(r, "path")}
    assert "/api/runs/{run_id}/resume" in paths
    assert "繼續跑" in PAGE.read_text(encoding="utf-8")


def test_cards_belong_to_the_material_not_to_one_attempt() -> None:
    """"Content-addressed" and "written once per output directory" at once.

    Cards and transcripts describe the clip, which is why they are worth
    keeping -- and they lived beside the output, so a second cut of the same
    rushes rewrote all seventy-four of them for forty-four cents before
    anything had been decided. They are named by the bytes now and live in
    one library, so any run over the same material finds them.
    """

    import inspect

    from montagewright import cli, clipcard
    from montagewright.uploads import default_library

    assert "library" in str(default_library())
    assert "content_hash(proxy)" in inspect.getsource(clipcard.build_library)

    source = inspect.getsource(cli.command_render)
    assert 'library / "cards"' in source
    assert 'library / "transcripts"' in source


def test_the_cut_can_be_adjusted_without_replanning_it() -> None:
    """The four things anyone wants after watching it once.

    A sentence cut short, a shot that runs long, an order that reads better
    the other way, one shot that should go. None of those need the film
    re-planned, and re-planning them costs money and changes everything
    else. The strip trims, reorders and drops; the recut renders from the
    amended order with no model calls, so it is free.
    """

    from montagewright.webapp import PAGE, create_app

    paths = {r.path for r in create_app().routes if hasattr(r, "path")}
    assert "/api/runs/{run_id}/recut" in paths
    assert "/api/runs/{run_id}/timeline-data" in paths

    page = PAGE.read_text(encoding="utf-8")
    for piece in ("paintStrip", "data-edge", "ondrop", "data-kill", "undo-cut"):
        assert piece in page, piece
    # Pulling the head earlier eats into the handle rather than sliding the
    # shot, so the out-point stays where the edit put it.
    assert "eats into the handle" in page
    # One shot at a time, beside the viewer, instead of a table to scroll.
    assert "function inspect" in page and 'id="inspector"' in page


def test_a_degradation_is_shown_in_words_with_its_number() -> None:
    """"static_on_subject　accept" names the code that raised it.

    A degradation is worth recording rather than hiding because it carries
    the measurement that forced it. Printing the enum and the verdict hides
    exactly that -- the reader learns there was a fallback and nothing about
    what happened or how far off it was.
    """

    from montagewright.schema import DegradationStep
    from montagewright.webapp import PAGE

    page = PAGE.read_text(encoding="utf-8")
    assert "tellDegradation" in page
    # Every ladder the schema can produce has something to say.
    ladders = DegradationStep.model_fields["ladder"].annotation.__args__
    for name in ladders:
        if name != "other":
            assert f"{name}:" in page, name
    for named in ("subject_wider_than_delivery", "tracking_lost_most_frames",
                  "trim_window_clamped_to_source", "subject_larger_than_crop"):
        assert f"{named}:" in page, named
    assert "measured" in page and "已改動" in page


def test_the_adjudication_says_who_looked_and_what_they_concluded() -> None:
    """"accept" is a word from the schema, not an account of anything.

    It means somebody watched that shot on its own, with the degradation
    beside it, and decided the picture still works. Which layer did the
    watching is the part that matters, and the part that changed: the
    whole-cut reviewer never saw the shot it was ruling on.
    """

    from montagewright.webapp import PAGE

    page = PAGE.read_text(encoding="utf-8")
    assert "看過畫面：可以這樣交" in page
    assert "逐顆驗收單獨看了這一顆" in page
    assert "沒有人看過" in page


def test_an_empty_panel_says_what_empty_means() -> None:
    """A panel showing only its subtitle reads as broken.

    Nothing set aside means every clip was usable; no transcript means the
    speech was not the content. Both are answers, and both looked like a
    failure to load.
    """

    from montagewright.webapp import PAGE

    page = PAGE.read_text(encoding="utf-8")
    assert "emptyRow" in page
    assert "每一支素材都用上了" in page
    assert "這一輪沒有做逐字稿" in page
    assert "還沒有跑過任何一輪" in page


def test_earlier_runs_are_grouped_by_the_material() -> None:
    """One folder of rushes, several cuts out of it.

    That is the unit of the work and the unit the cards are cached by, so
    runs in a group are the cheap ones -- which is the thing worth seeing
    when deciding whether to cut the same material again.
    """

    from montagewright.webapp import PAGE, create_app
    from fastapi.testclient import TestClient

    listed = TestClient(create_app()).get("/api/runs").json()["runs"]
    assert all("source_path" in row for row in listed)
    page = PAGE.read_text(encoding="utf-8")
    assert "bySource" in page and "支剪成" in page


def test_the_slowest_stage_reports_each_shot() -> None:
    """A grounding call and sometimes a propagation, per shot.

    SAM writes its progress with carriage returns that never reach a log, so
    minutes passed with the last visible line still being about the music --
    and a process at 0% CPU waiting on the network looks exactly like a hung
    one from the page.
    """

    import inspect

    from montagewright import pipeline
    from montagewright.webapp import PAGE

    assert "subject {index}/{total}" in inspect.getsource(
        pipeline.follow_subjects
    )
    page = PAGE.read_text(encoding="utf-8")
    # How long the current stage has been going, so waiting reads as waiting.
    assert "stepSince" in page and "已經 ${since(" in page
    assert "CPU 沒有動是正常的" in page


def test_every_way_a_clip_misses_the_film_is_shown() -> None:
    """Only the rarest of three was on screen.

    A card can call a take unusable, the direction can rule one out, and
    selection can simply pass one over -- and the last of those is how most
    of a folder does not reach the film. Showing only the first made a run
    that ruled out three takes and passed over sixty report that every clip
    had been usable.
    """

    import inspect

    from montagewright import cli
    from montagewright.webapp import PAGE

    page = PAGE.read_text(encoding="utf-8")
    for how in ("卡片判定不能用", "定調排除", "選片沒挑"):
        assert how in page, how
    # "Passed over" can only be told from what was on the table.
    assert "material_ids" in page
    assert "material_ids" in inspect.getsource(cli._write_report)


def test_the_timeline_has_tracks_on_one_time_scale() -> None:
    """A cut with speech under music is two things happening at once.

    The page could only play it. Seeing where the voice sits and where the
    bed steps back is the difference between trusting the mix and checking
    it -- and a strip laid out by proportion rather than by time cannot line
    up with anything.
    """

    from montagewright.webapp import PAGE, create_app

    paths = {r.path for r in create_app().routes if hasattr(r, "path")}
    assert "/api/runs/{run_id}/waveform/{which}" in paths

    page = PAGE.read_text(encoding="utf-8")
    for piece in ("paintRuler", "paintWaves", "playhead",
                  "lane-voice", "lane-music"):
        assert piece in page, piece
    # Clicking the reel seeks; a timeline that cannot be scrubbed is a chart.
    assert "$('reel').onclick" in page


def test_a_run_is_found_after_a_restart_without_asking_for_the_list() -> None:
    """Runs are picked up off disk lazily and only the listing did it.

    Every other endpoint answered "no such run" until something happened to
    fetch the list first, which is an ordering nobody could see.
    """

    import inspect

    from montagewright import webapp

    source = inspect.getsource(webapp.create_app)
    guts = source[source.index("def _run(run_id"):]
    assert "recall()" in guts[:400]


def test_the_crop_can_be_checked_against_the_take_it_came_from() -> None:
    """"push_in on (0.507, 0.484)" is a claim in a report.

    Playing the take underneath with the box drawn on it is that claim being
    checked, which is the whole job here. The rendered segment cannot show it
    -- it is the answer, not the working.
    """

    from montagewright.webapp import PAGE, create_app
    from fastapi.testclient import TestClient

    app = create_app()
    paths = {r.path for r in app.routes if hasattr(r, "path")}
    assert "/api/runs/{run_id}/source/{which}" in paths

    # The crop path travels with the timeline, keyframe by keyframe.
    page = PAGE.read_text(encoding="utf-8")
    assert "cropAt" in page and "drawCrop" in page
    assert "原素材＋裁切框" in page
    # A label built from the crop width said the same sentence on every
    # shot -- 9:16 out of 16:9 is 0.316 wide always. It has to say where the
    # box is pointed and whether it travelled.
    assert "cropSays" in page and "across(" in page
    assert "偏左" in page and "偏右" in page and "置中" in page


def test_a_card_is_found_by_what_it_describes_not_by_its_filename(
    tmp_path,
) -> None:
    """The rebuild used to key cards by filename and miss every one.

    A card is named for the hash of the proxy it describes. Taking that name
    to be the source id meant the map was empty in a way nothing could see:
    every lookup missed, every shot reframed with no subject, every crop dead
    centre -- and the report still described the subject it had followed.
    """

    from montagewright.clipcard import card_map
    from montagewright.uploads import content_hash

    proxies = tmp_path / "proxies"
    proxies.mkdir()
    cards = tmp_path / "cards"
    cards.mkdir()

    proxy = proxies / "C8329.mp4"
    proxy.write_bytes(b"not really a video, but it hashes")
    named_for_its_bytes = cards / f"{content_hash(proxy)[:20]}.json"
    named_for_its_bytes.write_text("{}", encoding="utf-8")

    found = card_map(proxies, cards)
    assert found == {"C8329": named_for_its_bytes}
    # The old way. It is what the bug looked like from the inside.
    assert "C8329" not in {path.stem: path for path in cards.glob("*.json")}


def test_a_proxy_with_no_card_is_left_out_rather_than_guessed_at(
    tmp_path,
) -> None:
    from montagewright.clipcard import card_map

    (tmp_path / "proxies").mkdir()
    (tmp_path / "cards").mkdir()
    (tmp_path / "proxies" / "C0001.mp4").write_bytes(b"unanalysed")
    assert card_map(tmp_path / "proxies", tmp_path / "cards") == {}


def test_the_planner_is_told_to_name_a_subject_the_card_already_measured(
) -> None:
    """A reworded subject is a subject that has to be located again.

    The listing hands the planner every subject the card measured, with its
    box. Free-form naming meant six shots in nine described theirs in wording
    the card never used -- sometimes in another language entirely -- and each
    miss fell through to a paid grounding call for a position already sitting
    in the library.
    """

    from montagewright.planner import _selection_schema

    schema = _selection_schema(["C8330", "C8332"])
    said = str(schema)
    assert "可框住的主體" in said
    assert "copy" in said and "exactly" in said


def test_a_block_carries_the_shot_it_came_from() -> None:
    """Peeking at the take needs the shot's index, not the reel's.

    The two are the same number until a recut drops or reorders anything, and
    the block never carried either -- so the take never loaded, and the crop
    box had no picture to be drawn on.
    """

    from pathlib import Path

    from montagewright.webapp import PAGE

    page = PAGE.read_text(encoding="utf-8")
    # Addressed by the take now. The position was the bug this test was
    # written for and then became one of its own: it changes at every cut
    # even when the next shot comes out of the same file, so the browser
    # dropped a take it already had and fetched it again under a new name.
    assert "/source/${encodeURIComponent(b.source_id)}" in page
    # The endpoint still answers to a position, because a block carries one
    # and an older page may still ask that way.
    from montagewright import webapp

    import inspect

    resolve = inspect.getsource(webapp.create_app)
    assert "which.isdigit()" in resolve
    source = (
        Path(__file__).resolve().parents[1]
        / "src" / "montagewright" / "webapp.py"
    ).read_text(encoding="utf-8")
    assert 'current_blocks[index]["selection_index"]' in source


def test_the_reel_says_whether_its_boxes_are_the_render_s_or_a_rebuild(
) -> None:
    """A guess shaped like evidence is worse than no evidence.

    Only a held frame can be re-derived after the fact, and only when a card
    can name the subject. Where it cannot, the rebuild centres -- and a
    centred box drawn over the take, unlabelled, says the render centred
    too. This view exists to be checkable, so it has to say which it holds.
    """

    from montagewright.webapp import PAGE

    page = PAGE.read_text(encoding="utf-8")
    assert "crops_are" in page and "cropsAre" in page
    assert "推測" in page
    assert "不等於實際裁切" in page


def test_peeking_does_not_seek_the_take_every_frame() -> None:
    """A seek is a decode from the nearest keyframe.

    Doing one per frame to keep two players aligned pinned a core and played
    like a slideshow, when a playing video keeps its own time for free.
    """

    from montagewright.webapp import PAGE

    page = PAGE.read_text(encoding="utf-8")
    assert "raw.play()" in page
    assert "cut.paused ? 0.08 : 0.35" in page


def test_the_take_is_served_as_a_proxy_when_there_is_one() -> None:
    """128MB of 4K to draw a rectangle on, where 256KB says the same thing."""

    from pathlib import Path

    source = (
        Path(__file__).resolve().parents[1]
        / "src" / "montagewright" / "webapp.py"
    ).read_text(encoding="utf-8")
    assert 'proxy = run.output / "work" / "proxies" / f"{source_id}.mp4"' in source
    assert "if proxy.exists():" in source


def test_one_reframe_builder_for_the_run_and_for_the_rebuild() -> None:
    """The rebuild's copy was missing then_subject.

    A pan between two subjects in one frame is the only move that reads it,
    so the handoff branch was never reached and a pan measured at
    0.278 -> 0.696 was rebuilt as a held centre crop.
    """

    from montagewright.schema import reframe_of

    both = reframe_of({
        "subject": "the left, black smartphone",
        "then_subject": "the right, white smartphone",
        "camera_move": "pan",
    })
    assert both.then_subject is not None
    assert both.then_subject.description == "the right, white smartphone"
    assert reframe_of({"subject": "a phone"}).then_subject is None


def test_a_proxy_is_kept_where_the_cards_it_feeds_are_kept(tmp_path) -> None:
    """Re-encoding seventy-four 4K files to ask the first question.

    A proxy is a pure function of the bytes it was made from -- the same
    reason the card built from it is content-addressed and shared. Keeping it
    in the output directory meant a second cut of the same rushes paid the
    whole encode again before anything was decided.
    """

    from montagewright.cli import _make_proxy
    from montagewright.uploads import content_hash

    library = tmp_path / "library"
    source = tmp_path / "C0001.MP4"
    source.write_bytes(b"pretend this is a take")

    kept = library / "proxies" / f"{content_hash(source)[:20]}.mp4"
    kept.parent.mkdir(parents=True)
    kept.write_bytes(b"already encoded once")

    first = _make_proxy(source, tmp_path / "a" / "C0001.mp4", library=library)
    assert first.read_bytes() == b"already encoded once"

    # A second run over the same rushes finds it too, under its own name.
    second = _make_proxy(source, tmp_path / "b" / "C0001.mp4", library=library)
    assert second.read_bytes() == b"already encoded once"
    assert second.name == "C0001.mp4"


def test_the_zoom_slider_does_not_redraw_the_waveform_per_pixel() -> None:
    """Drawing a waveform decodes the whole cut.

    It is cached by the width asked for, and the slider asked at a width
    derived from its exact position -- a miss every time, two ffmpeg passes
    over the film per pixel of drag. The width is rounded to something the
    cache can hold, and the request waits for the drag to settle.
    """

    from montagewright.webapp import PAGE

    page = PAGE.read_text(encoding="utf-8")
    assert "Math.ceil(asked / 500) * 500" in page
    assert "paintWavesWhenSettled" in page
    assert "clearTimeout(wavesSoon)" in page


def test_the_run_can_write_down_the_crops_it_used(tmp_path) -> None:
    """The record has to survive a real path, not a plausible one.

    It was written against a field name Keyframe does not have, which no
    test touched -- so it raised only after twelve paid grounding calls, at
    the moment the run had everything it needed and was about to render.
    """

    import json

    from montagewright.executor import CropBox
    from montagewright.pipeline import write_crops
    from montagewright.reframe import CropPath, Keyframe

    path = CropPath(keyframes=[
        Keyframe(seconds=0.0, crop=CropBox(x=0.34, y=0.0, width=0.32, height=1.0)),
        Keyframe(seconds=2.5, crop=CropBox(x=0.41, y=0.05, width=0.26, height=0.82)),
    ])
    out = tmp_path / "deep" / "crops.json"
    write_crops({"k00": path}, out)

    back = json.loads(out.read_text(encoding="utf-8"))
    assert [k["at"] for k in back["k00"]] == [0.0, 2.5]
    assert back["k00"][1]["w"] == 0.26


def test_the_selection_becomes_an_edl_without_reaching_for_a_missing_name(
    tmp_path,
) -> None:
    """Two runs died here on names that were not defined.

    Both were a moved import, and both raised only after the run had paid for
    cards, direction and selection -- the point where an editing tool has
    spent everything and delivered nothing. Nothing exercised this function,
    so nothing said so until it was expensive.
    """

    from montagewright.cli import _edl_from_selection

    selection = {
        "shots": [
            {
                "source_id": "C0001",
                "subject": "the left, black smartphone",
                "then_subject": "the right, white smartphone",
                "camera_move": "pan",
                "start_seconds": 1.5,
                "seconds_needed": 3.0,
                "why": "hand off between the two handsets",
                "energy": "medium",
            },
            {
                "source_id": "C0002",
                "subject": "the coin beside the hinge",
                "camera_move": "hold",
                "start_seconds": 0.0,
                "must_be_whole": True,
            },
        ]
    }

    edl, snaps = _edl_from_selection(selection, tmp_path, cards={})

    assert [clip.clip_id for clip in edl.clips] == ["k00", "k01"]
    first, second = edl.clips
    assert first.reframe.camera_move == "pan"
    assert first.reframe.then_subject is not None
    assert first.approx_out_seconds - first.approx_in_seconds == 3.0
    # No seconds_needed means the fallback length, not zero.
    assert second.approx_out_seconds > second.approx_in_seconds
    assert second.reframe.subject.min_visible == 1.0
    assert snaps == {}


def test_the_overlay_eases_the_way_the_render_does() -> None:
    """The box is checked against the picture, so it has to move like it.

    The crop expression ramps on smoothstep -- the camera takes up the move
    and sets it down. Interpolating the drawn box linearly put it in the
    wrong place through the middle of every move, which is the part of a
    move anyone is checking.
    """

    from montagewright.webapp import PAGE

    page = PAGE.read_text(encoding="utf-8")
    assert "u * u * (3 - 2 * u)" in page

    # The same curve the expression builder writes.
    from montagewright.reframe import _eased

    assert "*(3-2*" in _eased(0.0, 1.0, 0.0, 1.0)


def test_the_reel_moves_without_relaying_itself_out_every_frame() -> None:
    """A one-pixel line, sixty times a second, at the cost of a full layout.

    Writing `left` on the playhead relayouts the reel under it; a transform
    is composited. Reading the scroller's geometry after writing that style
    forces the browser to flush the layout it was told to do. And the clock
    reads in seconds, so writing its text every frame is fifty-nine text
    measurements a second that change nothing.
    """

    from montagewright.webapp import PAGE

    page = PAGE.read_text(encoding="utf-8")
    assert "translateX(${x}px)" in page
    assert "if (now !== clockSaid)" in page
    # Every measurement read before anything is written.
    move = page[page.index("function movePlayhead()"):]
    move = move[:move.index("\n}")]
    assert move.index("box.scrollLeft, wide") < move.index("style.transform")


def test_the_crop_overlay_does_not_measure_the_page_every_frame() -> None:
    """Two getBoundingClientRect calls a frame, to move one rectangle.

    Where the take sits inside the frame changes when the window changes or
    another take loads, and at no other time.
    """

    from montagewright.webapp import PAGE

    page = PAGE.read_text(encoding="utf-8")
    assert "let placed = null" in page
    assert "function forgetPlacement()" in page
    assert "loadedmetadata" in page


def test_the_reviewer_is_told_what_this_tool_cannot_do() -> None:
    """A reviewer cannot tell "done badly" from "cannot be done here".

    A brief asked for title cards. There is no text layer, so the reviewer
    reported their absence as a fault, returned "revise" on a cut that was
    exactly as planned, and sent a paid round after something no replan can
    ever fix. The list of what the executor can do already lives in one
    place and is rendered into the prompt that chooses; the list of what it
    cannot now sits beside it, rendered into the prompt that judges.
    """

    from montagewright.capabilities import CANNOT, describe_limits_for_prompt
    from montagewright.review import PROMPTS

    said = describe_limits_for_prompt()
    assert "字卡" in said and "轉場" in said
    assert len(CANNOT) == said.count("\n") + 1

    prompt = (PROMPTS / "review_zh-TW.txt").read_text(encoding="utf-8")
    assert "{limits}" in prompt
    assert "不要把它們當成缺點寫進 issues" in prompt

    # And it is actually substituted, not left as a literal brace.
    filled = prompt.replace("{limits}", said)
    assert "{limits}" not in filled and "字卡" in filled


def test_a_run_made_from_the_command_line_is_openable(tmp_path) -> None:
    """The interface listed only the cuts it had started itself.

    A run from the command line left a report, a film and two timelines, and
    nothing that could open them -- so the one made to check the crops with
    could not be looked at.
    """

    import json

    import montagewright.webapp as web

    folder = tmp_path / "check-0805"
    (folder / "out").mkdir(parents=True)
    (folder / "out" / "command.json").write_text(
        json.dumps({"source": "/rushes", "command": ["render", "/rushes"]}),
        encoding="utf-8",
    )
    (folder / "out" / "report.json").write_text("{}", encoding="utf-8")

    was_root, was_runs = web.RUNS_ROOT, dict(web.RUNS)
    try:
        web.RUNS_ROOT = tmp_path
        web.RUNS.clear()
        web.recall()
        assert "check-0805" in web.RUNS
        found = web.RUNS["check-0805"]
        assert found.source == "/rushes"
        assert found.state == "done"
        assert found.started_at > 0
    finally:
        web.RUNS_ROOT = was_root
        web.RUNS.clear()
        web.RUNS.update(was_runs)


def test_a_cut_written_anywhere_can_still_be_listed(tmp_path) -> None:
    """The interface only scans its own runs folder.

    `--output ~/cut` is what the README tells people to type, and it produced
    a film, a report and two timelines that nothing could open. The runs
    folder gets a link to wherever the cut actually went, so every path in
    the interface carries on believing the layout it already believes.
    """

    import montagewright.cli as command
    import montagewright.webapp as web

    was = web.RUNS_ROOT
    try:
        web.RUNS_ROOT = tmp_path / "runs"
        elsewhere = tmp_path / "somewhere" / "mycut"
        elsewhere.mkdir(parents=True)
        (elsewhere / "report.json").write_text("{}", encoding="utf-8")

        command._make_findable(elsewhere)

        link = tmp_path / "runs" / "mycut" / "out"
        assert link.is_symlink()
        assert link.resolve() == elsewhere.resolve()
        assert (link / "report.json").exists()

        # Twice is not two links to the same place, nor a crash.
        command._make_findable(elsewhere)
        assert sorted(p.name for p in (tmp_path / "runs").iterdir()) == ["mycut"]
    finally:
        web.RUNS_ROOT = was


def test_a_cut_already_in_the_runs_folder_is_left_alone(tmp_path) -> None:
    import montagewright.cli as command
    import montagewright.webapp as web

    was = web.RUNS_ROOT
    try:
        web.RUNS_ROOT = tmp_path / "runs"
        inside = tmp_path / "runs" / "abc123" / "out"
        inside.mkdir(parents=True)
        command._make_findable(inside)
        assert sorted(p.name for p in (tmp_path / "runs").iterdir()) == ["abc123"]
    finally:
        web.RUNS_ROOT = was


def test_a_line_lands_where_the_cut_put_the_take_it_came_from() -> None:
    """A line is timed against its take; the cut kept part of that take.

    This lived inside the SRT endpoint because that was the only thing that
    needed it. The track on the timeline needs it, and burning it into the
    picture will need it, and three copies of "where does this line land" is
    three answers to it.
    """

    from montagewright.transcript import against_cut

    lines = against_cut(
        [
            {"source_id": "A", "start_seconds": 2.0},
            {"source_id": "B", "start_seconds": 0.0},
        ],
        {"k00": {"seconds": 3.0}, "k01": {"seconds": 2.0}},
        {
            "A": {"lines": [
                {"text": "早安", "starts_seconds": 2.5, "ends_seconds": 4.0,
                 "speaker": "主持人"},
                # Spoken in the take, after the part that was used.
                {"text": "太早了", "starts_seconds": 9.0, "ends_seconds": 10.0},
            ]},
            "B": {"lines": [
                {"text": "第二顆", "starts_seconds": 0.2, "ends_seconds": 1.4},
            ]},
        },
    )

    assert [line.text for line in lines] == ["早安", "第二顆"]
    assert lines[0].starts_seconds == 0.5      # 2.5 in a take entered at 2.0
    assert lines[0].speaker == "主持人"
    assert lines[1].starts_seconds == 3.2      # after a three-second shot


def test_who_said_it_is_a_decision_rather_than_a_default() -> None:
    """Two callers had already drifted -- one prefixed, one did not."""

    from montagewright.transcript import Line, to_srt

    lines = [
        Line(text="早安", starts_seconds=0.5, ends_seconds=2.0,
             speaker="主持人"),
        Line(text="你好", starts_seconds=2.0, ends_seconds=3.0),
    ]
    named = to_srt(lines, with_speaker=True)
    plain = to_srt(lines)

    assert "主持人：早安" in named and "主持人" not in plain
    # And neither loses the numbering or the timestamps.
    for made in (named, plain):
        assert made.startswith("1\n00:00:00,500 --> 00:00:02,000\n")
        assert "\n2\n00:00:02,000 --> 00:00:03,000\n" in made


def test_an_edited_line_is_what_appears_and_the_transcript_stays_true(
    tmp_path, monkeypatch,
) -> None:
    """Gemini fixes most of what the recogniser mishears, not all of it.

    A product name is exactly the kind of word it gets wrong. The correction
    is kept beside the transcript rather than written over it: the transcript
    is what was heard, which stays true, and this is what should be on
    screen.

    Built the way a run builds it -- a proxy per source, and a transcript in
    the shared library named for that proxy's bytes. Two readers were looking
    in the output directory and keying by filename, so every run had an empty
    transcript tab and no subtitles, and both read as "no speech here".
    """

    import json
    from dataclasses import dataclass

    from montagewright.transcript import save
    from montagewright.uploads import content_hash
    from montagewright.webapp import _subtitle_lines

    monkeypatch.setenv("MONTAGEWRIGHT_LIBRARY", str(tmp_path / "library"))

    @dataclass
    class Pretend:
        output: Path

        def report(self):
            return {
                "selection": {"shots": [{"source_id": "A",
                                         "start_seconds": 0.0}]},
                "rhythm": {"k00": {"seconds": 3.0}},
            }

    proxies = tmp_path / "out" / "work" / "proxies"
    proxies.mkdir(parents=True)
    proxy = proxies / "A.mp4"
    proxy.write_bytes(b"a take with someone talking in it")

    save({"lines": [
        {"text": "Galaxy Z 佛的", "starts_seconds": 0.0,
         "ends_seconds": 2.0, "heard": "Galaxy Z 佛的"},
    ]}, tmp_path / "library" / "transcripts"
        / f"{content_hash(proxy)[:20]}.json")

    run = Pretend(output=tmp_path / "out")
    assert [line.text for line in _subtitle_lines(run)] == ["Galaxy Z 佛的"]

    (tmp_path / "out" / "work" / "subtitles.json").write_text(
        json.dumps([{"at": 0.0, "until": 2.0, "text": "Galaxy Z Fold",
                     "heard": "Galaxy Z 佛的"}], ensure_ascii=False),
        encoding="utf-8",
    )
    fixed = _subtitle_lines(run)
    assert [line.text for line in fixed] == ["Galaxy Z Fold"]
    # What was actually heard survives the correction.
    assert fixed[0].heard == "Galaxy Z 佛的"


def test_the_safe_area_is_a_property_of_where_the_film_is_going() -> None:
    """A 9:16 cut is watched inside an app that draws over its own bottom.

    The handle, the caption and the button rail sit in the lower fifth, so a
    subtitle where a subtitle traditionally goes is a subtitle nobody reads.
    A 16:9 cut has none of that. One number for both would be the execution
    layer deciding something about distribution.
    """

    from montagewright.subtitles import SAFE_AREAS, safe_area

    tall = safe_area("9:16")
    wide = safe_area("16:9")
    assert tall.up_from_bottom > wide.up_from_bottom * 2
    assert tall.side_margin > wide.side_margin
    # Every aspect the render flag offers has one.
    assert set(SAFE_AREAS) == {"9:16", "4:5", "1:1", "16:9"}
    # An aspect nobody planned for gets the most constrained band, not none.
    assert safe_area("21:9") == SAFE_AREAS["9:16"]


def test_a_subtitle_never_loses_a_word_to_fit(tmp_path) -> None:
    """Cutting the overflow at max_lines dropped the end of a sentence and
    left a cut that looked finished.

    Fitting is tried in this order: split the sentence into separate cues,
    then set it smaller, and only then let it take a third row. What is
    never tried is saying less.
    """

    from montagewright.subtitles import _face, draw_line, safe_area, wrap

    area = safe_area("9:16")
    room = round(1080 * (1 - area.side_margin * 2))
    # Nowhere to break it into cues -- no punctuation anywhere -- and too
    # long for two rows even at the smallest size this will set.
    said = (
        "夏天最崩潰的事情就是流汗然後又曬傷然後又中暑然後還要擠捷運"
        "然後回到家發現冷氣壞掉真的是非常非常痛苦的一件事情啊"
        "而且隔天起來還要再經歷一次一模一樣的事情"
    )
    asked = round(1920 * area.text_height)
    assert len(wrap(said, _face(asked), room)) > area.max_lines

    made = draw_line(
        said, width=1080, height=1920, area=area, into=tmp_path / "one.png",
    )
    assert made is not None
    drawn, left, top = made
    assert drawn.exists() and top > 0

    # Whatever size it settled on, every character is still on the picture.
    for attempt in range(6):
        size = max(12, round(asked * (1 - attempt * 0.06)))
        rows = wrap(said, _face(size), room)
        if len(rows) <= area.max_lines:
            break
    assert "".join(rows) == said


def test_no_line_begins_with_a_mark_that_closes_one() -> None:
    """Breaking before a comma hangs it under the line above.

    It is the one typographic mistake in Chinese that everybody notices, and
    it was in the first frame that came out of this.
    """

    from montagewright.subtitles import NEVER_STARTS, _face, safe_area, wrap

    area = safe_area("9:16")
    room = round(360 * (1 - area.side_margin * 2))
    for said in (
        "對，然後你就…你就已經濕漉漉了，然後慢慢被自己蒸乾",
        "在夏天讓你最崩潰的事情是什麼？我覺得是流汗，還有曬傷。",
    ):
        for line in wrap(said, _face(round(640 * area.text_height)), room)[1:]:
            assert line[0] not in NEVER_STARTS, line


def test_two_lines_are_balanced_rather_than_filled() -> None:
    """A full line and an orphan reads as a mistake on screen."""

    from montagewright.subtitles import _face, wrap

    face = _face(40)
    said = "對，然後你就已經濕漉漉了，然後慢慢被自己蒸乾"
    wide = face.getbbox(said)[2]
    lines = wrap(said, face, round(wide * 0.6))

    assert len(lines) == 2
    shorter = min(face.getbbox(one)[2] for one in lines)
    longer = max(face.getbbox(one)[2] for one in lines)
    assert shorter > longer * 0.6, lines


def test_a_long_sentence_becomes_several_cues_not_more_rows() -> None:
    """The transcript's idea of a line is a sentence.

    The median is thirteen characters and the tail runs to fifty-six.
    Wrapping the long ones put fifty characters of Chinese over somebody's
    face, which is not a subtitle, it is a paragraph -- and setting it
    smaller only made it a smaller paragraph.
    """

    from montagewright.subtitles import _face, safe_area, split_cues, wrap
    from montagewright.transcript import Line

    area = safe_area("9:16")
    face = _face(round(1920 * area.text_height))
    room = round(1080 * (1 - area.side_margin * 2))
    said = (
        "哦，如果是這種…這種就是如果今天洗完澡，然後出來又是剛好冷氣"
        "又壞掉的話，應該就是會蠻…蠻不開心、蠻不爽的呀，對。"
    )

    cues = split_cues([Line(text=said, starts_seconds=10.0,
                            ends_seconds=17.0)], face, room)

    assert len(cues) > 1
    # Every one of them fits on a single row.
    for cue in cues:
        assert len(wrap(cue.text, face, room)) == 1, cue.text
    # Nothing said twice, nothing lost, and the window is the one it had.
    assert "".join(cue.text for cue in cues) == said
    assert cues[0].starts_seconds == 10.0
    assert abs(cues[-1].ends_seconds - 17.0) < 1e-6
    # And they run in order, without gaps or overlaps.
    for before, after in zip(cues, cues[1:]):
        assert abs(before.ends_seconds - after.starts_seconds) < 1e-6


def test_a_split_cue_uses_measured_words_instead_of_character_pace() -> None:
    """A pause before the next phrase must not put that phrase on screen early."""

    from montagewright.subtitles import _face, _width, split_cues
    from montagewright.transcript import Line, Word

    face = _face(40)
    text = "前半句說得很快，後半句停一下才開始"
    room = _width("前半句說得很快，", face) + 2
    words = []
    at = 0.0
    for character in "前半句說得很快":
        words.append(Word(character, at, at + 0.1))
        at += 0.1
    # A real pause. Character-proportional timing would start the next cue
    # around the middle of the 4s line rather than at this measured boundary.
    at = 2.4
    for character in "後半句停一下才開始":
        words.append(Word(character, at, at + 0.2))
        at += 0.2

    cues = split_cues(
        [Line(text=text, starts_seconds=0.0, ends_seconds=4.0)],
        face, room, words=words,
    )

    assert len(cues) >= 2
    assert cues[1].starts_seconds == pytest.approx(2.4)
    assert cues[0].ends_seconds == cues[1].starts_seconds


def test_a_split_cue_prefers_the_corrected_character_clock() -> None:
    """Gemini spelling and Apple timing remain joined through cue layout."""

    from montagewright.subtitles import _face, _width, split_cues
    from montagewright.transcript import CharacterTiming, Line, Word

    text = "前半句說得很快，後半句停一下才開始"
    clock, at = [], 0.0
    for character in text:
        if character == "，":
            clock.append(CharacterTiming(character, at, at, False))
            at = 2.4
            continue
        clock.append(CharacterTiming(character, at, at + 0.1, True))
        at += 0.1
    line = Line(
        text=text, starts_seconds=0.0, ends_seconds=3.3,
        timed_text=tuple(clock),
    )
    # Deliberately misleading legacy word evidence. It must not override the
    # corrected character clock carried by the line.
    words = [Word(character, index * .05, index * .05 + .05)
             for index, character in enumerate(text) if character != "，"]
    cues = split_cues(
        [line], _face(40), _width("前半句說得很快，", _face(40)) + 2,
        words=words,
    )

    assert len(cues) >= 2
    assert cues[1].starts_seconds == pytest.approx(2.4)
    assert cues[1].timed_text


def test_cue_layout_does_not_leave_one_character_of_a_phrase_orphaned() -> None:
    """Visual width may split a quote, but not as a lone final character."""

    from montagewright.subtitles import _by_sense, _face, _width

    face = _face(40)
    text = "「喔，你身上有梅雨的味道。」這樣子。"
    room = round(_width("「喔，你身上有梅雨的味", face) + 1)
    pieces = _by_sense(text, face, room)

    assert "".join(pieces) == text
    meaningful = [one.strip("，。、！？：；,.!?;:「」") for one in pieces]
    assert all(len(one) >= 2 for one in meaningful), pieces
    assert not any(one.startswith("道") for one in pieces[1:])
    assert any("梅雨的味道" in one for one in pieces), pieces


def test_a_brief_cue_is_joined_only_while_it_still_fits_one_row() -> None:
    """Two rules pull against each other, and one of them wins.

    A cue too short to read is a flicker, so it gets joined to the one
    before. But joining up to two rows traded that fault for a worse one --
    a wall of text where a quick line was wanted. So the join happens only
    while the result still fits on a single row, and a short cue that
    cannot be absorbed stays as it is.
    """

    from montagewright.subtitles import _face, _width, split_cues
    from montagewright.transcript import Line

    face = _face(40)
    room = _width("十二個字的一行寬度啊啊", face)
    said = "好，對，是，嗯，然後呢，就這樣，真的很誇張啊我跟你講"
    cues = split_cues(
        [Line(text=said, starts_seconds=0.0, ends_seconds=1.2)], face, room,
        least=0.7,
    )

    # Some joining happened: fewer cues than there are places to break.
    assert 1 < len(cues) < said.count("，") + 1
    # None of them overflows the row.
    for cue in cues:
        assert _width(cue.text, face) <= room, cue.text
    assert "".join(cue.text for cue in cues) == said


def test_a_shot_that_catches_part_of_a_sentence_shows_that_part() -> None:
    """The window was clipped to the shot and the words were not.

    So a shot holding one second of a ten-second sentence put the whole
    sentence on screen for one second. There are no word timings kept, so
    the share of the window stands in for the share of the words, and the
    ends are nudged to where the sentence pauses.
    """

    from montagewright.transcript import Line, _within

    line = Line(
        text="對，然後你就…你就已經濕漉漉了，然後慢慢被自己…被下午的太陽弄乾",
        starts_seconds=63.95,
        ends_seconds=74.03,
    )

    # The shot catches only the first second of it.
    opening = _within(line, from_seconds=63.95, to_seconds=64.95)
    assert opening and len(opening) < len(line.text) / 3
    assert line.text.startswith(opening.rstrip("…，"))

    # Wholly inside the shot: untouched, not re-cut.
    assert _within(line, from_seconds=60.0, to_seconds=80.0) == line.text

    # A sliver too short to read is nothing, rather than two characters.
    assert _within(line, from_seconds=63.95, to_seconds=64.05) == ""


def test_clipping_a_line_never_inverts_the_slice() -> None:
    """Snapping the head past the tail produced an empty string.

    rfind counts from the end when given a negative start, so an unclamped
    search window looked at the wrong part of the line -- and the subtitle
    it produced simply vanished, which nothing would have reported.
    """

    from montagewright.transcript import Line, _within

    line = Line(
        text="對，然後你就…你就已經濕漉漉了，然後慢慢被自己…被慢慢被下午的太陽弄乾了",
        starts_seconds=0.0,
        ends_seconds=10.0,
    )
    # Every window of a reasonable size gives something or nothing on
    # purpose; none of them gives an accidental empty string.
    for at in range(0, 9):
        got = _within(line, from_seconds=float(at), to_seconds=at + 2.0)
        assert got == "" or len(got) >= 3, (at, got)
        assert got in line.text or got.strip("…，。") in line.text, (at, got)


def test_clipping_without_a_nearby_joint_does_not_restore_the_whole_line() -> None:
    """A missing punctuation joint is -1, not an instruction to use index 0."""

    from montagewright.transcript import Line, _within

    line = Line(
        text="夏天最讓我崩潰哦",
        starts_seconds=1.44,
        ends_seconds=4.80,
    )
    clipped = _within(line, from_seconds=4.0, to_seconds=9.0)
    assert clipped != line.text
    assert clipped in line.text


def test_corrected_character_clock_keeps_only_the_audible_tail() -> None:
    """Corrected text follows Apple's character clock, not string proportion."""

    from montagewright.transcript import (
        CharacterTiming, CutWindow, Line, _portion_within, against_windows,
    )

    line = Line(
        text="夏天最讓我崩潰哦？",
        starts_seconds=1.44,
        ends_seconds=4.80,
        timed_text=tuple(
            CharacterTiming(letter, start, end, measured)
            for letter, start, end, measured in [
                ("夏", 1.44, 1.80, True), ("天", 1.80, 2.16, True),
                ("最", 2.16, 2.52, True), ("讓", 2.52, 2.88, True),
                ("我", 2.88, 3.24, True), ("崩", 3.24, 4.02, True),
                ("潰", 4.02, 4.80, False), ("哦", 4.80, 4.80, False),
                ("？", 4.80, 4.80, False),
            ]
        ),
    )
    said, starts, ends = _portion_within(
        line, from_seconds=4.0, to_seconds=9.0
    )
    assert said == "崩潰哦？"
    assert starts == pytest.approx(4.0)
    assert ends == pytest.approx(4.8)

    card = {
        "lines": [{
            "text": line.text,
            "starts_seconds": line.starts_seconds,
            "ends_seconds": line.ends_seconds,
            "timed_text": [one.__dict__ for one in line.timed_text],
        }]
    }
    moved = against_windows([CutWindow("A", 4.0, 5.0)], {"A": card})
    assert [one.text for one in moved] == ["崩潰哦？"]
    assert moved[0].starts_seconds == pytest.approx(0.0)
    assert moved[0].ends_seconds == pytest.approx(0.8)


def test_final_dialogue_audit_snaps_to_a_measured_pause() -> None:
    """Action/music grounding may move a safe proposal back into speech."""

    from montagewright.schema import Clip, EDL
    from montagewright.transcript import snap_edl_to_dialogue

    edl = EDL(project_id="p", clips=[Clip(
        clip_id="k00", source_id="A",
        approx_in_seconds=1.0, approx_out_seconds=4.05,
    )])
    card = {
        "lines": [{
            "text": "前半句，後半句。",
            "starts_seconds": 1.0,
            "ends_seconds": 5.0,
            "timed_text": [
                {"text": "前", "starts_seconds": 1.0, "ends_seconds": 1.5},
                {"text": "半", "starts_seconds": 1.5, "ends_seconds": 2.0},
                {"text": "句", "starts_seconds": 2.0, "ends_seconds": 2.5},
                {"text": "，", "starts_seconds": 2.5, "ends_seconds": 2.5},
                {"text": "後", "starts_seconds": 2.8, "ends_seconds": 3.3},
                {"text": "半", "starts_seconds": 3.3, "ends_seconds": 3.8},
                {"text": "句", "starts_seconds": 3.8, "ends_seconds": 4.3},
                {"text": "。", "starts_seconds": 4.3, "ends_seconds": 4.3},
            ],
        }],
        "words": [
            {"text": "前半句", "starts_seconds": 1.0, "ends_seconds": 2.5},
            {"text": "後半句", "starts_seconds": 2.8, "ends_seconds": 4.3},
        ],
    }
    snapped, notes, faults = snap_edl_to_dialogue(edl, {"A": card})
    assert faults == []
    assert notes
    assert snapped.clips[0].approx_out_seconds == pytest.approx(4.3)


def test_final_dialogue_audit_refuses_an_unfixable_mid_sentence_cut() -> None:
    from montagewright.schema import Clip, EDL
    from montagewright.transcript import snap_edl_to_dialogue

    edl = EDL(project_id="p", clips=[Clip(
        clip_id="k00", source_id="A",
        approx_in_seconds=1.0, approx_out_seconds=5.0,
    )])
    card = {"lines": [{
        "text": "這是一整段沒有停頓而且還沒有講完的話",
        "starts_seconds": 1.0,
        "ends_seconds": 9.0,
    }]}
    _, _, faults = snap_edl_to_dialogue(
        edl, {"A": card}, max_snap_seconds=0.3
    )
    assert faults and "cuts active dialogue" in faults[0]


def test_discarded_source_audio_does_not_constrain_a_visual_cut() -> None:
    """A product B-roll shot may come from a file containing irrelevant talk."""

    from montagewright.schema import Clip, EDL
    from montagewright.transcript import snap_edl_to_dialogue

    edl = EDL(project_id="p", clips=[Clip(
        clip_id="k00", source_id="A",
        approx_in_seconds=3.0, approx_out_seconds=5.0,
        audio_role="discard", audio_completion="none",
        picture_role="primary_action",
    )])
    card = {"lines": [{
        "text": "現場有人一直講話但這顆只使用產品畫面",
        "starts_seconds": 0.0, "ends_seconds": 10.0,
    }]}
    resolved, notes, faults = snap_edl_to_dialogue(edl, {"A": card})
    assert faults == []
    assert notes == []
    assert resolved.clips[0].approx_in_seconds == 3.0


def test_audio_and_picture_roles_survive_into_the_render_segment(tmp_path) -> None:
    from montagewright.executor import Source, plan_render
    from montagewright.schema import Clip, EDL

    source = Source("A", tmp_path / "a.mp4", 10.0, 1920, 1080)
    edl = EDL(project_id="p", clips=[Clip(
        clip_id="k00", source_id="A",
        approx_in_seconds=1.0, approx_out_seconds=3.0,
        audio_role="discard", audio_completion="none",
        picture_role="illustrative_broll",
    )])
    segment = plan_render(edl, {"A": source}).segments[0]
    assert segment.audio_role == "discard"
    assert segment.audio_completion == "none"
    assert segment.picture_role == "illustrative_broll"


def test_audio_assignment_contract_rejects_narration_without_speech() -> None:
    from montagewright.planner import MaterialItem, audio_assignment_disagreements

    material = [MaterialItem(source_id="A", duration_seconds=10.0, summary="")]
    faults = audio_assignment_disagreements([{
        "source_id": "A",
        "audio_role": "narrative",
        "audio_completion": "complete_thought",
    }], material)
    assert faults and "no transcribed content speech" in faults[0]


def test_discarded_audio_is_silenced_before_segment_concat() -> None:
    import inspect
    from montagewright import renderer

    source = inspect.getsource(renderer._render_segment)
    assert 'segment.audio_role == "discard"' in source
    assert '["-af", "volume=0"]' in source


def test_discarded_audio_renders_as_silence_with_the_same_stream_layout(
    tmp_path,
) -> None:
    import subprocess

    from montagewright.executor import Segment, Source
    from montagewright.renderer import _level, _render_segment

    source_path = tmp_path / "source.mp4"
    subprocess.run([
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
        "-f", "lavfi", "-i", "color=c=blue:s=64x64:r=30:d=1",
        "-f", "lavfi", "-i", "sine=frequency=440:duration=1",
        "-shortest", "-c:v", "libx264", "-c:a", "aac", str(source_path),
    ], check=True)
    segment = Segment(
        "k00", Source("A", source_path, 1.0, 64, 64), 0.0, 1.0,
        audio_role="discard",
    )
    rendered, _ = _render_segment(
        segment, tmp_path / "muted.mp4", video_encoder="libx264",
        output_size=(64, 64), output_fps=30, output_frames=30,
    )
    assert _level(rendered) < -80.0


def test_a_refused_run_leaves_nothing_behind(tmp_path) -> None:
    """The folder was made before the input was checked.

    So every mistyped path left an empty directory in the runs folder that
    nothing would ever open, list or clean up.
    """

    import montagewright.webapp as web
    from fastapi.testclient import TestClient

    was = web.RUNS_ROOT
    try:
        web.RUNS_ROOT = tmp_path / "runs"
        client = TestClient(web.create_app())

        gone = client.post("/api/runs", data={"source_path": "/no/such/one"})
        assert gone.status_code == 400

        empty = tmp_path / "no-footage"
        empty.mkdir()
        (empty / "notes.txt").write_text("not a video", encoding="utf-8")
        barren = client.post("/api/runs", data={"source_path": str(empty)})
        assert barren.status_code == 400

        wrong = client.post(
            "/api/runs", data={"source_path": "/tmp", "aspect": "3:2"}
        )
        assert wrong.status_code == 400

        made = list((tmp_path / "runs").iterdir()) if (
            tmp_path / "runs"
        ).exists() else []
        assert made == [], made
    finally:
        web.RUNS_ROOT = was


def test_nothing_cut_yet_is_an_invitation_not_an_empty_editor() -> None:
    """A black rectangle, empty tracks and an empty inspector.

    That is what a first run saw, and it reads as broken rather than as new.
    The only readable thing on the page was a line at the bottom of a drawer.
    """

    from montagewright.webapp import PAGE

    page = PAGE.read_text(encoding="utf-8")
    assert 'id="nothing-yet"' in page
    assert "還沒有剪過東西" in page
    assert "function showWorkspace(" in page
    # Off at startup, on only when a run is opened.
    assert "showWorkspace(false);" in page
    assert "showWorkspace(true);" in page


def test_the_setup_offers_what_the_command_line_offers() -> None:
    """A flag the interface cannot set is a flag most people never find."""

    from pathlib import Path

    from montagewright.webapp import PAGE

    page = PAGE.read_text(encoding="utf-8")
    assert 'id="subtitles"' in page
    assert "body.append('subtitles'" in page

    source = (
        Path(__file__).resolve().parents[1]
        / "src" / "montagewright" / "webapp.py"
    ).read_text(encoding="utf-8")
    assert 'subtitles: str = Form("sidecar")' in source
    assert 'command += ["--subtitles", subtitles]' in source


def test_the_system_is_asked_for_a_font_before_paths_are_guessed() -> None:
    """A hard-coded list of paths is a guess about somebody else's machine.

    It was wrong on this one in both directions: it missed the font the
    system would have named, and the font the system names first is one
    FreeType cannot open. So asking is the start of the search, not the end.
    """

    from montagewright import subtitles as typeset

    face = typeset._face(40, text="夏天最崩潰的事")
    assert face is not None
    assert typeset._can_draw(face, "夏天最崩潰的事")

    # An explicit choice wins over anything found.
    was = typeset.CHOSEN
    try:
        typeset.CHOSEN = "/System/Library/Fonts/STHeiti Medium.ttc"
        assert typeset._face(40, text="夏天").path.endswith("STHeiti Medium.ttc")
    finally:
        typeset.CHOSEN = was


def test_a_font_is_chosen_for_the_language_not_for_one_stray_character(
) -> None:
    """One emoji in a line means no Chinese font draws everything.

    Taking the first candidate that did set a whole street interview in a
    maths font, which had the emoji and not the language.
    """

    from montagewright import subtitles as typeset

    with_emoji = typeset._face(40, text="好熱😀真的")
    plain = typeset._face(40, text="好熱真的")
    assert with_emoji.path == plain.path
    assert typeset._can_draw(with_emoji, "好熱真的")


def test_characters_the_font_cannot_spell_are_named() -> None:
    """Pillow draws a missing glyph as an empty box and says nothing, so a
    name it cannot spell reaches a finished film."""

    from montagewright.subtitles import cannot_spell

    assert cannot_spell("夏天最崩潰的事") == ""
    assert "😀" in cannot_spell("好熱😀真的")


def test_one_shot_can_be_turned_down_without_re_planning_anything() -> None:
    """Levelling makes every speaker the same loudness.

    That is not the same as every speaker being right: one of them stood
    next to a road. The gain belongs to the segment, survives a re-cut, and
    costs nothing because nothing has to be decided again.
    """

    from pathlib import Path

    from montagewright.executor import Segment, Source
    from montagewright.webapp import PAGE

    made = Segment(
        clip_id="k00",
        source=Source(source_id="A", path=Path("/nowhere.mp4"), width=1920,
                      height=1080, duration_seconds=10.0),
        in_seconds=0.0, out_seconds=2.0,
    )
    assert made.gain_db == 0.0

    renderer = (
        Path(__file__).resolve().parents[1]
        / "src" / "montagewright" / "renderer.py"
    ).read_text(encoding="utf-8")
    assert 'f"volume={segment.gain_db:.2f}dB"' in renderer
    # Silence costs a filter nobody needs.
    assert "abs(segment.gain_db) > 0.01" in renderer

    web = (
        Path(__file__).resolve().parents[1]
        / "src" / "montagewright" / "webapp.py"
    ).read_text(encoding="utf-8")
    assert 'gains[f"k{index:02d}"]' in web
    assert "segment.gain_db = gains.get(segment.clip_id, 0.0)" in web

    page = PAGE.read_text(encoding="utf-8")
    assert 'id="gain"' in page and "gain_db: b.gain_db || 0" in page


def test_the_regional_face_and_a_weight_that_reads_over_a_picture() -> None:
    """A .ttc holds several faces and index 0 is whatever it lists first.

    For PingFang that is Hong Kong Regular, so a Taiwanese cut was being set
    in the wrong regional character forms, in a weight too thin to hold up
    against a moving picture.
    """

    from montagewright import subtitles as typeset

    for lang, region in (("zh-tw", "TC"), ("zh-cn", "SC"), ("zh-hk", "HK")):
        family, style = typeset._face(40, text="夏天", lang=lang).getname()
        if not family.startswith("PingFang"):
            continue  # another machine, another font; nothing to assert
        assert family.endswith(region), (lang, family)
        assert style in ("Medium", "Semibold"), (lang, style)


def test_the_words_move_onto_the_cut_with_the_lines() -> None:
    """Otherwise a line asked to fill as it is said fills to the rhythm of
    a completely different part of the interview."""

    from montagewright.transcript import words_against_cut

    cards = {"A": {"words": [
        {"text": "在", "starts_seconds": 10.0, "ends_seconds": 10.3},
        {"text": "夏", "starts_seconds": 10.3, "ends_seconds": 10.6},
        # Spoken in the take, outside the part that was used.
        {"text": "後", "starts_seconds": 30.0, "ends_seconds": 30.4},
    ]}}
    moved = words_against_cut(
        [{"source_id": "A", "start_seconds": 10.0}],
        {"k00": {"seconds": 2.0}},
        cards,
    )
    assert [word.text for word in moved] == ["在", "夏"]
    assert moved[0].starts_seconds == 0.0
    assert abs(moved[1].starts_seconds - 0.3) < 1e-6


def test_subtitles_follow_resolved_windows_after_action_snapping() -> None:
    """The rendered in-point wins over the earlier selection proposal."""

    from types import SimpleNamespace

    from montagewright.transcript import (
        against_windows, windows_against_segments, words_against_windows,
    )

    cards = {"A": {
        "lines": [
            {"text": "提早保留", "starts_seconds": 0.2, "ends_seconds": 1.2},
            {"text": "原本入點", "starts_seconds": 2.2, "ends_seconds": 3.2},
        ],
        "words": [
            {"text": "提", "starts_seconds": 0.2, "ends_seconds": 0.4},
            {"text": "原", "starts_seconds": 2.2, "ends_seconds": 2.4},
        ],
    }}
    segment = SimpleNamespace(
        source=SimpleNamespace(source_id="A"),
        in_seconds=0.0,
        duration_seconds=3.5,
    )
    windows = windows_against_segments([segment])

    lines = against_windows(windows, cards)
    words = words_against_windows(windows, cards)

    assert [line.text for line in lines] == ["提早保留", "原本入點"]
    assert lines[0].starts_seconds == pytest.approx(0.2)
    assert [word.text for word in words] == ["提", "原"]
    assert words[0].starts_seconds == pytest.approx(0.2)


def test_resolved_subtitle_windows_follow_late_snap_and_final_durations() -> None:
    """A later snap drops old speech and advances the next shot by real frames."""

    from types import SimpleNamespace

    from montagewright.transcript import against_windows, windows_against_segments

    cards = {
        "A": {"lines": [
            {"text": "已剪掉", "starts_seconds": 4.2, "ends_seconds": 5.2},
            {"text": "真正留下", "starts_seconds": 6.2, "ends_seconds": 7.2},
        ]},
        "B": {"lines": [
            {"text": "下一顆", "starts_seconds": 1.5, "ends_seconds": 2.5},
        ]},
    }
    segments = [
        SimpleNamespace(
            source=SimpleNamespace(source_id="A"),
            in_seconds=6.0,
            duration_seconds=2.0,
        ),
        SimpleNamespace(
            source=SimpleNamespace(source_id="B"),
            in_seconds=1.0,
            duration_seconds=3.0,
        ),
    ]

    lines = against_windows(windows_against_segments(segments), cards)

    assert [line.text for line in lines] == ["真正留下", "下一顆"]
    assert lines[0].starts_seconds == pytest.approx(0.2)
    assert lines[1].starts_seconds == pytest.approx(2.5)


def test_speech_detector_intervals_remain_separate_measured_evidence() -> None:
    """VAD evidence is preserved, but never fabricated from token lengths."""

    from montagewright.transcript import detector_silences

    assert detector_silences({"silences": []}) == []
    assert detector_silences({
        "silences": [
            {"starts_seconds": 1.2344, "ends_seconds": 2.3456},
            # SpeechDetector can finish with an open interval represented as
            # a zero-length marker.  It is not a measured silence interval.
            {"starts_seconds": 8.0, "ends_seconds": 8.0},
            {"starts_seconds": "bad", "ends_seconds": 9.0},
        ],
    }) == [{"starts_seconds": 1.234, "ends_seconds": 2.346}]


def test_transcript_cache_identity_includes_the_apple_speech_helper(
    tmp_path, monkeypatch,
) -> None:
    """Changing the local timestamp contract cannot reuse an old card."""

    import montagewright.transcript as transcript

    executable = tmp_path / "Transcribe"
    source = executable.with_suffix(".swift")
    source.write_text("let detector = SpeechDetector(.high)\n", encoding="utf-8")
    monkeypatch.setattr(transcript, "TOOL", executable)
    before = transcript._transcript_version()

    source.write_text("let detector = SpeechDetector(.medium)\n", encoding="utf-8")
    after = transcript._transcript_version()

    assert before != after


def test_an_anomalous_apple_word_range_is_not_rewritten_without_evidence() -> None:
    """A suspicious duration alone is not permission to alter Apple's clock."""

    from montagewright.transcript import CutWindow, words_against_windows

    card = {"words": [
        {"text": "要", "starts_seconds": 2.46, "ends_seconds": 4.38},
        {"text": "流", "starts_seconds": 4.38, "ends_seconds": 4.56},
    ]}
    moved = words_against_windows(
        [CutWindow("A", 0.0, 6.0)], {"A": card},
    )

    assert moved[0].starts_seconds == pytest.approx(2.46)
    assert moved[0].ends_seconds == pytest.approx(4.38)


def test_a_cue_fills_at_the_speed_it_was_actually_said() -> None:
    """Character by character, against measured timings rather than a pace.

    That is the difference between reading along and watching a progress
    bar. When the two cannot be lined up, the cue is drawn whole rather
    than guessed at.
    """

    from montagewright.subtitles import spans_in
    from montagewright.transcript import Word

    said = "在夏天讓你最崩潰的事情是什麼？"
    words = [
        Word(text=ch, starts_seconds=i * 0.3, ends_seconds=(i + 1) * 0.3)
        for i, ch in enumerate("在夏天讓你最崩潰的事情是什麼")
    ]
    marks = spans_in(said, words, 0.0, 5.0)

    assert len(marks) == len(words)
    # Each mark says how much of the line has been said by when.
    assert marks[0] == (1, 0.3)
    # The question mark is revealed with the character before it.
    assert marks[-1][0] == len(said)
    # Nothing measurable in this window: draw it in one piece.
    assert spans_in(said, words, 90.0, 95.0) == []


def test_the_interface_offers_the_fonts_this_machine_has() -> None:
    """A flag on the command line is a flag most people never find, and
    typing a path to a font file is worse than that."""

    from pathlib import Path

    from montagewright.webapp import PAGE

    page = PAGE.read_text(encoding="utf-8")
    # Both places a subtitle gets set: before a run, and when burning one.
    assert 'id="setup-font"' in page and 'id="font"' in page
    assert 'id="setup-look"' in page and 'id="look"' in page
    assert "'/api/fonts'" in page and "function loadFonts(" in page
    assert "body.append('subtitle_font'" in page

    source = (
        Path(__file__).resolve().parents[1]
        / "src" / "montagewright" / "webapp.py"
    ).read_text(encoding="utf-8")
    assert '@app.get("/api/fonts")' in source
    assert 'command += ["--subtitle-font", subtitle_font]' in source
    # A chosen font applies to one render and does not leak into the next.
    assert "was, typeset.CHOSEN = typeset.CHOSEN, (font or None)" in source
    assert "typeset.CHOSEN = was" in source


def test_the_font_list_leaves_out_what_nobody_should_pick() -> None:
    """LastResort is the font that draws the boxes, and the dotted names
    are interface variants the system keeps for itself."""

    from montagewright.subtitles import fonts_here

    found = fonts_here()
    if not found:
        return  # no fontconfig on this machine; nothing to check
    assert all(not one["family"].startswith(".") for one in found)
    assert all("LastResort" not in one["file"] for one in found)
    assert len({one["family"] for one in found}) == len(found)


def test_the_preview_places_subtitles_where_the_burn_will() -> None:
    """Correcting wording without seeing it in place is guessing.

    So the player draws each cue over the picture as it plays -- and it is
    driven by the band the render actually uses, sent with the timeline. A
    second opinion about where the words sit would make the preview a lie
    about the thing it exists to preview.
    """

    from pathlib import Path

    from montagewright.subtitles import safe_area
    from montagewright.webapp import PAGE

    page = PAGE.read_text(encoding="utf-8")
    assert 'id="burnt"' in page and "function showBurnt(" in page
    # The client reads the server's numbers rather than keeping its own.
    assert "safeArea = data.safe_area" in page
    for field in ("side_margin", "text_height", "up_from_bottom"):
        assert f"safeArea.{field}" in page
    # Nothing is placed against an element that has not loaded: before
    # metadata it is 300x150, which set a whole line at four pixels.
    assert "if (!video.videoWidth) { box.classList.add('hide'); return; }" in page

    source = (
        Path(__file__).resolve().parents[1]
        / "src" / "montagewright" / "webapp.py"
    ).read_text(encoding="utf-8")
    assert '"safe_area": _safe_area_of(report)' in source

    # And the numbers it sends are the ones the burn is built from.
    from montagewright.webapp import _safe_area_of

    for aspect in ("9:16", "16:9", "1:1", "4:5"):
        sent = _safe_area_of({"direction": {"aspect": aspect}})
        band = safe_area(aspect)
        assert sent["up_from_bottom"] == band.up_from_bottom
        assert sent["side_margin"] == band.side_margin
        assert sent["text_height"] == band.text_height


def test_the_font_list_follows_the_language_of_the_cut() -> None:
    """Offering Chinese faces to somebody cutting a Japanese interview is a
    list with nothing they want in it."""

    from montagewright.subtitles import fonts_here

    chinese = {one["family"] for one in fonts_here("zh-tw")}
    japanese = {one["family"] for one in fonts_here("ja")}
    if not chinese or not japanese:
        return  # no fontconfig here
    assert chinese != japanese


def test_the_zoom_budget_guards_the_size_that_is_actually_delivered() -> None:
    """It was calibrated against an output that never existed.

    zoom_budget assumed 1080x1920 while the renderer scaled every segment to
    whatever the opening crop happened to measure -- 1214x2160 off a 4K
    source at 9:16. So a push the report called 1.35x enlargement was 1.52x
    on disk, and the one number deciding how far a shot may push was
    protecting a file nobody was making.
    """

    from montagewright.executor import delivery_size
    from montagewright.reframe import MAX_UPSCALE, zoom_budget

    for aspect, expected in (
        (9 / 16, (1080, 1920)), (16 / 9, (1920, 1080)),
        (1.0, (1080, 1080)), (0.8, (1080, 1350)),
    ):
        assert delivery_size(aspect) == expected

    wide, tall = delivery_size(9 / 16)
    budget = zoom_budget(
        source_width=3840, source_height=2160, source_aspect=3840 / 2160,
        target_aspect=9 / 16, output_width=wide, output_height=tall,
    )
    # The tightest crop this allows, delivered at that size, enlarges by
    # exactly the budget -- not by half as much again.
    base_w = (9 / 16) / (3840 / 2160) * 3840
    tightest = base_w * budget
    assert abs((wide / tightest) - MAX_UPSCALE) < 0.01


def test_a_segment_is_scaled_to_the_delivery_not_to_its_own_crop() -> None:
    from pathlib import Path
    from montagewright.executor import CropBox
    from montagewright.reframe import (
        CropPath, Keyframe, ffmpeg_crop_filters,
    )

    renderer = (
        Path(__file__).resolve().parents[1]
        / "src" / "montagewright" / "renderer.py"
    ).read_text(encoding="utf-8")
    # The still path is explicit in renderer; moving paths are now compiled
    # by the shared filter builder because zoom needs perspective rather than
    # crop's init-only w/h options. Both still land on one delivery size.
    assert renderer.count('f"scale={output_size[0]}:{output_size[1]}"') == 2
    moving = CropPath([
        Keyframe(0, CropBox(.34, 0, .32, 1)),
        Keyframe(2, CropBox(.40, .17, .21, .66)),
    ])
    assert ffmpeg_crop_filters(
        moving, 3840, 2160, (1080, 1920)
    )[-1] == "scale=1080:1920"
    assert "keyframes[0].crop.to_pixels(" not in renderer


def test_the_planner_is_told_how_far_each_clip_can_be_pushed() -> None:
    """The execution layer was answering an editorial question alone.

    How far a shot may push before it softens is measured from that file's
    own dimensions -- it is a fact about the clip, not a rule about clips.
    Nobody was telling the planner, so it asked for pushes that could not be
    given and found out afterwards, in a degradation, with a constant in the
    executor deciding how much softness was acceptable.
    """

    from montagewright.planner import MaterialItem, _describe_material

    said = _describe_material([
        MaterialItem(source_id="BIG", duration_seconds=6.2, summary="4K",
                     push_room=1.52),
        MaterialItem(source_id="SMALL", duration_seconds=4.0, summary="1080p",
                     push_room=1.0),
    ])
    assert "最多推近 1.52×" in said
    assert "推近沒有空間" in said

    # And the prompt says what to do about it, or the number is decoration.
    from montagewright.planner import PROMPTS

    prompt = (PROMPTS / "selection_zh-TW.txt").read_text(encoding="utf-8")
    assert "最多推近" in prompt
    # What has to survive rewording: that "no room" is a fact about the file
    # rather than a caution, and that the answer is a different take rather
    # than a smaller version of the same move.
    assert "沒有空間" in prompt and "糊" in prompt
    assert "別把計畫打折" in prompt or "不要把原本的計畫打折" in prompt


def test_push_room_is_read_from_the_file_that_gets_cut(tmp_path) -> None:
    """The proxy is 640 pixels wide and would report that nothing anywhere
    can be pushed into."""

    from pathlib import Path

    source = (
        Path(__file__).resolve().parents[1]
        / "src" / "montagewright" / "cli.py"
    ).read_text(encoding="utf-8")
    assert "originals = {path.stem: path for path in sources_paths}" in source
    assert "originals.get(source_id, proxy)" in source


def test_a_finished_cut_can_be_taken_away() -> None:
    """The interface could make a film, show it, prove the crop followed
    something -- and offer no way to get any of it out.

    The download links lived in the drawer along the bottom, and the drawer
    was deleted when the layout became columns. Nothing failed, nothing was
    reported: the routes stayed, and the only thing that went was every way
    of reaching them.
    """

    from montagewright.webapp import PAGE, create_app

    paths = {r.path for r in create_app().routes if hasattr(r, "path")}
    assert {
        "/api/runs/{run_id}/deliverable",
        "/api/runs/{run_id}/timeline/{flavour}",
        "/api/runs/{run_id}/subtitles",
        "/api/runs/{run_id}/burned",
    } <= paths

    page = PAGE.read_text(encoding="utf-8")
    for reached in ("/deliverable", "/timeline/premiere", "/timeline/finalcut",
                    "/subtitles", "/burned"):
        assert reached in page, reached
    # Offered only when there is one -- a link that answers 404 reads as a
    # fault rather than as an absence.
    assert "$('dl-srt').classList.toggle('hide', !spoken)" in page


def test_final_cut_will_parse_what_we_write() -> None:
    """It refused the whole file and imported nothing.

    "No declaration for attribute time of element param" -- because the
    keyframes were written as <param time="..."/>, which is not a thing
    FCPXML has. A still frame is two attributes on adjust-transform; a move
    is keyframes inside a keyframeAnimation inside the param they belong to.
    """

    import xml.etree.ElementTree as ET
    from pathlib import Path

    from montagewright.executor import CropBox, RenderPlan, Segment, Source
    from montagewright.reframe import CropPath, Keyframe
    from montagewright.timeline import to_fcpxml

    where = Source(source_id="A", path=Path("/rushes/A.mp4"),
                   duration_seconds=20.0, width=3840, height=2160)
    still = Segment(clip_id="k00", source=where, in_seconds=0.0,
                    out_seconds=2.0,
                    crop=CropBox(x=0.34, y=0.0, width=0.32, height=1.0))
    moving = Segment(
        clip_id="k01", source=where, in_seconds=3.0, out_seconds=6.0,
        crop=CropBox(x=0.10, y=0.0, width=0.32, height=1.0),
        crop_path=CropPath(keyframes=[
            Keyframe(seconds=0.0,
                     crop=CropBox(x=0.10, y=0.0, width=0.32, height=1.0)),
            Keyframe(seconds=3.0,
                     crop=CropBox(x=0.55, y=0.0, width=0.32, height=1.0)),
        ]),
    )
    made = to_fcpxml(
        RenderPlan(project_id="p", segments=[still, moving]), {},
        name="p", width=1080, height=1920,
    )

    root = ET.fromstring(made)
    assert not [
        one for one in root.iter("param") if "time" in one.attrib
    ], "param carries no time attribute in FCPXML"

    adjusts = list(root.iter("adjust-transform"))
    assert len(adjusts) == 2
    # The still one says it in attributes.
    assert adjusts[0].get("position") and adjusts[0].get("scale")
    assert list(adjusts[0]) == []
    # The moving one wraps its keyframes.
    named = {one.get("name") for one in adjusts[1].iter("param")}
    assert named == {"position", "scale"}
    for one in adjusts[1].iter("param"):
        frames = list(one.iter("keyframe"))
        # Production uses smoothstep. Dense output-frame keys preserve that
        # curve in an NLE instead of delegating to its different interpolation.
        assert len(frames) == 91
        assert all(f.get("time") and f.get("value") for f in frames)


def test_the_sequence_format_is_a_shape_not_a_preset_name() -> None:
    """Final Cut warned that the sequence's format was an unexpected value.

    FFVideoFormat is the prefix Apple gives its built-in presets --
    FFVideoFormat1080p30 and the like -- so a bare "FFVideoFormat" sent it
    looking for a preset that does not exist. A custom size does not claim
    to be a preset; it states its dimensions. And every asset names a format
    of its own, or Final Cut is left to work out the shape of the media by
    opening it.
    """

    import xml.etree.ElementTree as ET
    from pathlib import Path

    from montagewright.executor import CropBox, RenderPlan, Segment, Source
    from montagewright.timeline import to_fcpxml

    where = Source(source_id="A", path=Path("/rushes/A.mp4"),
                   duration_seconds=20.0, width=3840, height=2160)
    made = to_fcpxml(
        RenderPlan(project_id="p", segments=[
            Segment(clip_id="k00", source=where, in_seconds=0.0,
                    out_seconds=2.0,
                    crop=CropBox(x=0.34, y=0.0, width=0.32, height=1.0)),
        ]),
        {}, name="p", width=1080, height=1920,
    )
    root = ET.fromstring(made)

    shapes = {one.get("id"): one for one in root.iter("format")}
    assert all(one.get("name") is None for one in shapes.values())
    # The sequence's own shape, and one for the media it cuts from.
    assert shapes["r1"].get("width") == "1080"
    assert any(one.get("width") == "3840" for one in shapes.values())

    for asset in root.iter("asset"):
        assert asset.get("format") in shapes, asset.get("id")
    assert root.find(".//sequence").get("format") in shapes


def test_keyframes_are_on_the_clip_s_own_clock() -> None:
    """A clip's clock starts at its source in-point, not at zero.

    Written from zero, a move began before the shot did and ended before it
    ended -- so the head and tail of every moving shot rendered with no
    transform, which for a 16:9 source in a 9:16 sequence is the picture
    letterboxed in black. That is the black somebody saw.
    """

    import xml.etree.ElementTree as ET
    from pathlib import Path

    from montagewright.executor import CropBox, RenderPlan, Segment, Source
    from montagewright.reframe import CropPath, Keyframe
    from montagewright.timeline import to_fcpxml

    where = Source(source_id="A", path=Path("/rushes/A.mp4"),
                   duration_seconds=30.0, width=3840, height=2160)
    moving = Segment(
        clip_id="k00", source=where, in_seconds=4.0, out_seconds=7.0,
        crop=CropBox(x=0.10, y=0.0, width=0.32, height=1.0),
        crop_path=CropPath(keyframes=[
            Keyframe(seconds=0.0,
                     crop=CropBox(x=0.10, y=0.0, width=0.32, height=1.0)),
            Keyframe(seconds=3.0,
                     crop=CropBox(x=0.55, y=0.0, width=0.32, height=1.0)),
        ]),
    )
    root = ET.fromstring(to_fcpxml(
        RenderPlan(project_id="p", segments=[moving]), {},
        name="p", width=1080, height=1920,
    ))

    def ticks(stamp: str) -> float:
        top, _, bottom = stamp.rstrip("s").partition("/")
        return float(top) / float(bottom or 1)

    clip = root.find(".//asset-clip")
    began, ran = ticks(clip.get("start")), ticks(clip.get("duration"))
    for frame in root.iter("keyframe"):
        at = ticks(frame.get("time"))
        assert began - 1e-6 <= at <= began + ran + 1e-6, (
            f"keyframe at {at} is outside the clip's {began}..{began + ran}"
        )
    # And they span it, rather than sitting in a corner of it.
    times = sorted({ticks(f.get("time")) for f in root.iter("keyframe")})
    assert abs(times[0] - began) < 1e-6
    assert abs(times[-1] - (began + ran)) < 1e-6


def test_the_timeline_carries_the_bed(tmp_path) -> None:
    """It opened as a silent film with no sign there had been a track."""

    import xml.etree.ElementTree as ET
    from pathlib import Path

    from montagewright.executor import CropBox, RenderPlan, Segment, Source
    from montagewright.timeline import to_fcpxml

    track = tmp_path / "bed.m4a"
    track.write_bytes(b"pretend this is music")
    where = Source(source_id="A", path=Path("/rushes/A.mp4"),
                   duration_seconds=20.0, width=3840, height=2160)
    root = ET.fromstring(to_fcpxml(
        RenderPlan(project_id="p", segments=[
            Segment(clip_id="k00", source=where, in_seconds=0.0,
                    out_seconds=3.0,
                    crop=CropBox(x=0.34, y=0.0, width=0.32, height=1.0)),
        ]),
        {}, name="p", width=1080, height=1920, music=track,
    ))

    bed = [
        one for one in root.iter("asset-clip")
        if one.get("audioRole") == "music"
    ]
    assert len(bed) == 1
    assert bed[0].get("lane") == "-1"       # under the picture
    assert bed[0].get("offset") == "0s"
    assert bed[0].get("ref") in {a.get("id") for a in root.iter("asset")}


def test_frames_are_not_pulled_when_there_is_nothing_to_ask() -> None:
    """Sampling frames runs ffmpeg over the take.

    Three branches did it and then checked whether a client existed to send
    them to -- so rebuilding a plan, which never has a client, extracted
    frames for every shot and discarded all of them. Opening a finished cut
    took eight and a half seconds of that before anything appeared.
    """

    from pathlib import Path

    source = (
        Path(__file__).resolve().parents[1]
        / "src" / "montagewright" / "pipeline.py"
    ).read_text(encoding="utf-8")

    lines = source.splitlines()
    # Every place frames are pulled has a check that asking is possible
    # close above it -- never only below it.
    sites = [
        i for i, line in enumerate(lines)
        if "_sample_frames(" in line and not line.lstrip().startswith("def ")
    ]
    assert sites, "the sampling calls moved; this guard needs rewriting"
    for at in sites:
        # Above for a guard clause or the branch it sits in, just below for
        # the conditional form -- `_sample_frames(...) if _may_ask(...)`.
        near = "\n".join(lines[max(0, at - 30):at + 8])
        assert "_may_ask(client)" in near, (
            f"line {at + 1} pulls frames with no check for a client:"
            f"\n{near}"
        )


def test_a_file_is_measured_once(tmp_path) -> None:
    """Reading a file's shape costs an ffprobe -- a whole process.

    Drawing the timeline asked for the same dozen sources every time, which
    was most of the second and a half before a cut appeared.
    """

    import montagewright.pipeline as works

    calls = []
    real = works.subprocess.run

    def counted(command, *args, **kw):
        if command and command[0] == "ffprobe":
            calls.append(command[-1])
        return real(command, *args, **kw)

    made = tmp_path / "one.mp4"
    import subprocess as sp
    sp.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
            "-f", "lavfi", "-i", "testsrc=size=64x64:rate=10:duration=1",
            str(made)], check=True)

    works.subprocess.run = counted
    try:
        first = works.probe("A", made)
        again = works.probe("A", made)
        # A second id for the same bytes is still not a second ffprobe.
        other = works.probe("B", made)
    finally:
        works.subprocess.run = real

    assert len(calls) == 1, calls
    assert first.duration_seconds == again.duration_seconds
    assert other.source_id == "B"
    assert other.duration_seconds == first.duration_seconds


def test_the_subtitle_panel_has_the_operations_captioning_needs() -> None:
    """A lane you can drag is not an editor.

    Four things come up over and over when captioning: put a line in, cut
    one in two where the speaker paused, join two that were split too
    finely, and take one out. Everything else is typing.
    """

    from montagewright.webapp import PAGE

    page = PAGE.read_text(encoding="utf-8")
    assert 'id="pane-subs"' in page
    for control in ("cue-add", "cue-split", "cue-join", "cue-del"):
        assert f"$('{control}')" in page, control

    # Timecodes are editable, not only draggable.
    assert "function unstamp(" in page and "function stamp(" in page
    # A cue may not end before it starts.
    assert "Math.min(want, subs[i].until - 0.1)" in page
    assert "Math.max(want, subs[i].at + 0.1)" in page
    # Splitting lands on a pause, near by, and never leaves a sliver.
    assert "Math.max(2, Math.min(cut, line.text.length - 2))" in page
    # The line being spoken is marked without rebuilding rows being typed in.
    assert "function followCue()" in page


def test_a_still_subject_says_so_before_the_shot_is_planned():
    """The executor was answering this after the plan was already spent.

    A frame told to follow something that stands still becomes a hold, and
    that substitution was happening in `reframe`, one pass too late to change
    anything, and recorded as a degradation -- forty-five of them in one run.
    The card has always known; the listing simply did not say.
    """

    from montagewright.cli import _subject_line
    from montagewright.clipcard import SubjectBox

    still = SubjectBox(
        label="the row of watches", centre_x=0.5, centre_y=0.5,
        width=0.56, height=0.3, moves=False, at_seconds=1.0,
    )
    walking = SubjectBox(
        label="the model", centre_x=0.5, centre_y=0.5,
        width=0.2, height=0.8, moves=True, at_seconds=1.0,
    )
    said = _subject_line(still, 16 / 9, 9 / 16)
    assert "定鏡" in said
    # And the fraction it can show is still there: two facts, one line.
    assert "57%" in said
    assert "移動" in _subject_line(walking, 16 / 9, 9 / 16)


def test_the_quoted_travel_time_is_the_one_the_render_will_take():
    """A price the planner budgets against, read from the executor's table.

    Selection is told `seconds_needed` must cover "the travel between" the
    looks and was never told the speed, so it could not price a move and
    stopped asking for them -- twenty-three shots, twenty-three single looks.
    The quote has to be the real number or it is worse than none.
    """

    from montagewright.planner import _travel_seconds
    from montagewright.reframe import ENERGY_LIMITS, seconds_needed_for
    from montagewright.schema import LOOK_ENERGIES

    room = 0.684
    quoted = _travel_seconds(room)
    for label, energy in LOOK_ENERGIES.items():
        # What the executor will actually charge for the same distance, with
        # no dwell at either end, is travel alone.
        stops = [(0.0, 0.0, 0.0, 0.3164), (0.0, room, 0.0, 0.3164)]
        charged = seconds_needed_for(stops, energy) - 2 * 0.35
        assert f"{label} {charged:.1f}s" in quoted, (label, quoted, charged)

    # All three, because the speed follows the energy the same answer picks.
    assert set(LOOK_ENERGIES) == {"low", "medium", "high"}
    assert len(set(ENERGY_LIMITS[e]["max_speed"] for e in LOOK_ENERGIES.values())) == 3


def test_the_energy_a_shot_asks_for_reaches_the_camera():
    """Two vocabularies, and until now nothing joined them.

    The shot says low/medium/high; the speed table is keyed calm/active/
    dynamic. `reframe_of` hard-coded "active", so a shot's own label had no
    effect on anything and a calm passage swept at the same rate as a frantic
    one.
    """

    from montagewright.reframe import ENERGY_LIMITS
    from montagewright.schema import look_energy, reframe_of

    for asked, expected in (("low", "calm"), ("medium", "active"), ("high", "dynamic")):
        assert look_energy(asked) == expected
        shot = {"looks": [{"at": "a", "seconds": 1.0}], "energy": asked}
        assert reframe_of(shot).camera_energy == expected

    # An absent or unknown label still has to land on a real speed.
    for junk in (None, "", "brisk"):
        assert look_energy(junk) in ENERGY_LIMITS


def test_the_planner_has_to_choose_a_camera_intent_before_looks():
    """A binary movement question still hid most of the available grammar.

    `camera_move` was a required enum, so every shot answered "does this one
    move?" before it could be written down. An array with a minimum length of
    one turned that into an option rather than a question, and the cheapest
    valid answer is a single look -- twenty-three shots, no move anywhere.
    """

    from montagewright.planner import _selection_schema

    shot = _selection_schema(["A"])["properties"]["shots"]["items"]
    assert "camera_intent" in shot["required"]
    assert shot["properties"]["camera_intent"]["enum"] == [
        "hold", "use_source_motion", "follow_subject", "reveal", "compare",
        "push_in", "pull_out", "multi_stop",
    ]

    # Asked before the looks are written, which is the working part -- a
    # model that has just written "travels" writes what follows in the
    # presence of that word.
    order = shot["required"]
    assert order.index("camera_intent") < order.index("looks")
    assert list(shot["properties"]).index("camera_intent") < list(
        shot["properties"]
    ).index("looks")


def test_a_plan_that_says_one_thing_and_describes_another_is_reported():
    """Prose and structure disagreed and nothing anywhere compared them.

    One shot's `why` said the frame sweeps across a row; its looks named a
    single place; the rhythm pass then repeated the sweep in its own
    reasoning; the film held still.
    """

    from montagewright.planner import frame_disagreements

    one = {"at": "a"}
    assert frame_disagreements([
        {"frame": "travels", "looks": [one]},
        {"frame": "settles", "looks": [one, one]},
    ]) == [
        "k00 said travels and gave one look",
        "k01 said settles and gave 2 looks",
    ]

    # Agreement is silent, in both directions.
    assert frame_disagreements([
        {"frame": "settles", "looks": [one]},
        {"frame": "travels", "looks": [one, one, one]},
    ]) == []

    # Legacy `travels` with one look meant follow-subject before the explicit
    # intent field existed; preserving that meaning keeps old cached plans
    # moving instead of silently turning them into holds.
    from montagewright.schema import move_of_shot

    assert move_of_shot({"frame": "travels", "looks": [one]}) == "follow_subject"


def test_the_speed_budget_covers_every_axis_not_just_across():
    """`_limit_speed` read `x` and copied the rest through unchanged.

    So a tilt arrived at whatever speed the keyframes asked for and a push
    changed size instantly, while the report said the energy budget had been
    applied. It was named for the budget and enforced it on the one move that
    existed when it was written.
    """

    from montagewright.reframe import ENERGY_LIMITS, Keyframe, _limit_speed
    from montagewright.executor import CropBox

    ceiling = ENERGY_LIMITS["calm"]["max_speed"]

    def travelled(start: CropBox, end: CropBox) -> tuple[float, float, float]:
        limited, _ = _limit_speed(
            [Keyframe(0.0, start), Keyframe(1.0, end)], ENERGY_LIMITS["calm"]
        )
        last = limited[-1].crop
        return (
            abs(last.x - start.x), abs(last.y - start.y),
            abs(last.width - start.width),
        )

    # Straight down, far further than a second of calm allows.
    base = CropBox(0.0, 0.0, 0.5, 0.5)
    _, down, _ = travelled(base, CropBox(0.0, 0.5, 0.5, 0.5))
    assert down <= ceiling + 1e-6, down

    # Closing in, ditto. The width may not collapse in one step.
    _, _, closed = travelled(base, CropBox(0.0, 0.0, 0.1, 0.1))
    assert closed <= ceiling + 1e-6, closed

    # A diagonal keeps its direction: clamping each axis on its own would
    # cut the longer component and leave the shorter one, bending the path.
    across, down, _ = travelled(base, CropBox(0.4, 0.2, 0.5, 0.5))
    assert across > 1e-6 and down > 1e-6
    assert abs(across / down - 2.0) < 0.05, (across, down)


def test_a_multi_look_path_cannot_crop_past_the_resolution_budget():
    """The looks builder was given two aspect ratios and no pixels.

    So it cropped as tightly as a framing asked and the pipeline measured the
    upscale afterwards, which is a record of a soft shot rather than a
    prevention of one -- and the shot has been spent by the time it is read.
    """

    from montagewright.reframe import (
        MAX_UPSCALE,
        achieved_upscale,
        build_look_path,
    )

    # A framing asking for a tenth of the frame width, on 4K delivered
    # 1080x1920 -- room to push, but nothing like this much.
    stops = [(0.4, 0.3, 0.5, 0.10), (0.4, 0.7, 0.5, 0.10)]
    common = dict(
        source_aspect=16 / 9, target_aspect=9 / 16, duration_seconds=3.0,
        energy="active",
    )
    pixels = dict(
        source_width=3840, source_height=2160,
        output_width=1080, output_height=1920,
    )

    degradations: list = []
    guarded = build_look_path(
        stops, clip_id="k00", degradations=degradations, **pixels, **common
    )
    tightest = min(guarded.keyframes, key=lambda one: one.crop.width).crop
    assert achieved_upscale(tightest, **pixels) <= MAX_UPSCALE + 1e-3
    assert [one.ladder for one in degradations] == ["reduced_zoom"]

    # And the same call without the pixel dimensions is the old behaviour,
    # which is what this guards against coming back.
    loose = build_look_path(stops, clip_id="k00", **common)
    assert achieved_upscale(
        min(loose.keyframes, key=lambda one: one.crop.width).crop, **pixels
    ) > MAX_UPSCALE

    # A source is not opened out further than its own pixels require: the
    # same framing on the same file delivered smaller keeps more of the push.
    smaller = build_look_path(
        stops, clip_id="k01", degradations=[],
        source_width=3840, source_height=2160,
        output_width=540, output_height=960, **common,
    )
    assert (
        min(one.crop.width for one in smaller.keyframes)
        < min(one.crop.width for one in guarded.keyframes)
    )


def test_the_production_path_passes_the_output_size_to_the_looks_builder():
    """A guard that only covers the builder guards the wrong caller.

    The last refactor's own test named a branch by string and went green
    when the branch was deleted; this asserts the call the pipeline makes.
    """

    import inspect

    from montagewright import pipeline

    source = inspect.getsource(pipeline.follow_subjects)
    call = source[source.index("build_look_path("):]
    call = call[: call.index(")\n")]
    for given in ("source_width=", "source_height=", "output_width=", "output_height="):
        assert given in call, given


def test_an_approving_film_review_does_not_bury_a_shot_that_missed():
    """Two reviewers, two questions, and only one verdict was being read.

    A run finished with three shots that did not do what they planned,
    forty-five degradations and five seconds missing -- and the whole-film
    reviewer said "approve (0 issues)", which returned before the shot
    reviewer's findings were even collected. The replan loop those findings
    exist to drive had never run.
    """

    from montagewright.review import Round, ReviewVerdict, should_continue

    approved = Round(
        index=1,
        verdict=ReviewVerdict(verdict="approve", overall="looks good", issues=[]),
        actionable=[],
    )

    # Nothing missed: an approval still ends the loop.
    assert should_continue([approved]) == (False, "approved")

    # Something missed: the film reviewer cannot see a promise it was never
    # told about, so its approval is not a veto over the shot reviewer.
    keep, why = should_continue([approved], undelivered=3)
    assert keep and "3" in why

    # And the limits still win, or a shot nobody can fix spends the budget
    # one replan at a time.
    from montagewright.review import MAX_ROUNDS

    capped = [approved] * MAX_ROUNDS
    assert should_continue(capped, undelivered=3)[0] is False


def test_the_shot_verdicts_are_collected_before_the_gate_that_reads_them():
    """The ordering is the bug, so the ordering is what is asserted."""

    import inspect

    from montagewright import cli

    source = inspect.getsource(cli.command_render)
    assert source.index("failing = [") < source.rindex("should_continue(")
    # Both gates read it. The one at the top of the loop decides whether a
    # replanned cut is looked at again, and reading a pre-replan verdict
    # there would stop on it.
    assert source.count("undelivered=undelivered") == 2


def test_an_informational_degradation_does_not_replan_a_delivered_shot():
    """Measurements inform review; they do not all declare failure."""

    from montagewright.cli import _replan_diagnostics
    from montagewright.schema import DegradationStep

    mild = DegradationStep(
        clip_id="k04",
        ladder="other",
        ladder_other="subject_wider_than_delivery",
        trigger="the crop can show 90.4% of the subject",
        measured={"most_visible_fraction": 0.904,
                  "requested_min_visible": 0.85},
    )
    said, mandatory = _replan_diagnostics(
        [], [mild], {
            "k04": {
                "delivered": True,
                "degradation_verdict": "acceptable",
                "note": "the intended detail is readable",
            }
        },
    )
    assert "k04" in said
    assert "k04" not in mandatory


def test_shot_failure_and_structural_faults_still_require_replan():
    from montagewright.cli import _replan_diagnostics

    _, mandatory = _replan_diagnostics(
        ["k02 static hold exceeds the maximum"],
        [],
        {"k07": {"delivered": False, "degradation_verdict": "none_recorded"}},
    )
    assert mandatory == {"k02", "k07"}


def test_silence_about_cropping_does_not_read_as_permission():
    """An optional boolean whose absent value is the permissive one.

    `must_be_whole` says partial cropping destroys this subject. It was not
    required, and an absent boolean reads false, so saying nothing meant
    cropping was fine -- the opposite of the safe answer for a field that
    exists to flag danger. In one cut seven of eight looks answered it and
    the eighth was a wordmark; its silence dropped the fraction that had to
    stay visible from all of it to 85%, and the title shipped as "y Unpacke"
    after three rounds of review.

    This is the same shape as the camera-move regression: a required enum
    became an optional array, and a question nobody is made to answer stops
    being answered.
    """

    from montagewright.planner import _selection_schema

    look = (
        _selection_schema(["C1:s00"])
        ["properties"]["shots"]["items"]["properties"]["looks"]["items"]
    )
    assert "must_be_whole" in look["required"]
    assert set(look["required"]) == set(look["properties"])


def test_selection_audits_cross_field_look_contract_before_edl():
    from montagewright.planner import look_contract_disagreements

    faults = look_contract_disagreements([{
        "looks": [{
            "at": "the complete product", "seconds": 2.0,
            "framing": "centre", "composition": "object_priority",
            "energy": "low", "must_be_whole": False,
            "entity_id": "device.fold",
            "presentation_intent": "complete_hold",
        }],
    }])
    assert len(faults) == 1
    assert "complete_hold requires must_be_whole=true" in faults[0]


def test_movement_that_reveals_nothing_is_not_the_source_doing_the_work():
    """Texture was described to the planner as a camera move.

    The listing labelled any recorded `camera_motion` as 素材自己的運鏡, and
    the selection prompt reads that label as a reason to hold: the source
    camera will bring the subject in, so a digital move on top would fight
    it. True of a reveal or a follow. False of handheld drift, which brings
    nothing in -- and a wordmark wider than any vertical crop sat on a take
    whose card said 微幅手持飄移, measured at 0.06 frame widths over three
    and a half seconds. Held twice, cropped both times.

    The distinction was already measured and already on the span.
    """

    from montagewright.planner import MaterialItem, _describe_material
    from montagewright.spans import Span

    def take(role):
        return MaterialItem(
            source_id="C1", duration_seconds=3.5, summary="標題牆",
            camera_motion="微幅手持飄移，鏡頭始終鎖定並呈現標題字樣。",
            spans=(Span(
                span_id="C1:s00", source_id="C1",
                starts_seconds=0.0, ends_seconds=3.5, motion_role=role,
            ),),
        )

    drift = _describe_material([take("handheld_texture")])
    assert "素材自己的運鏡" not in drift
    assert "這是質感，不是運鏡" in drift

    # A move that reveals still counts, because there the prompt is right.
    reveal = _describe_material([take("authored")])
    assert "素材自己的運鏡" in reveal


def test_every_thing_the_report_records_reaches_the_report():
    """A field added to the account and not to the file it writes.

    `plan_disagreements` was added to `Report`, filled during the run, read
    during replanning, and never serialised -- the edit that was supposed to
    put it in the payload matched nothing and changed nothing, silently,
    because a string replace that finds no target is not an error. Every
    test passed. The key was simply absent from report.json, which reads
    exactly like a run that had nothing to say.

    So omission has to be a decision. A field either appears in the payload
    or is named here as deliberately left out, and adding one without doing
    either fails.
    """

    import dataclasses
    import inspect

    from montagewright import cli
    from montagewright.pipeline import Report

    # Written under another name, or not written on purpose.
    elsewhere = {
        "aligned_cuts": "cuts_on_music",
        "total_cuts": "cuts_on_music",
        "following_shots": "shots_following",
        "static_shots": "shots_held",
        "rhythm_decisions": "rhythm",
        "delivered_seconds": "duration_seconds",
        "usages": "tokens",
        "ledger": "spend",
    }
    written = inspect.getsource(cli._write_report)
    missing = [
        one.name
        for one in dataclasses.fields(Report)
        if f'"{one.name}"' not in written
        and f'"{elsewhere.get(one.name, one.name)}"' not in written
    ]
    assert not missing, missing


def test_a_take_that_was_beaten_is_not_a_take_that_was_removed():
    """Naming a better attempt is a comparison, not a verdict on the file.

    Every entry in the direction's `unusable` list removed a whole source
    from selection, and the span contract exists precisely because a take is
    usually not all one thing. Across one cut that filter took eight sources
    and sixteen otherwise usable spans with them, and all eight had named a
    better take rather than called anything broken -- the worst was a
    forty-three second underwater run binned for how it ended.

    So the two are separated by what the answer itself already said. An
    entry naming `superseded_by` is advice and travels to selection beside
    the footage; an entry naming nothing is a verdict and removes the take.
    """

    import inspect

    from montagewright import cli
    from montagewright.planner import _beaten_and_broken

    ruled = {"unusable": [
        {"source_id": "C8400", "reason": "手錶在末段脫落",
         "superseded_by": "C8398"},
        {"source_id": "C8383", "reason": "誤按錄影，畫面沒有內容"},
    ]}
    beaten, broken = _beaten_and_broken(ruled)
    assert broken == {"C8383"}
    assert set(beaten) == {"C8400"}
    assert "C8398" in beaten["C8400"] and "脫落" in beaten["C8400"]

    # Selection keeps the beaten take and is told what the direction thought.
    source = inspect.getsource(cli.command_render)
    ruling = source[source.index('for entry in direction.get("unusable"'):]
    ruling = ruling[: ruling.index("chose = ")]
    assert "set_aside[source_id]" in ruling
    assert "in broken" in ruling

    picking = inspect.getsource(__import__(
        "montagewright.planner", fromlist=["select_shots"]
    ).select_shots)
    assert "not in broken" in picking
    assert "beaten" in picking
    # And it happens after the direction exists, not before it.
    assert source.index("set_aside: dict[str, str] = {}") < source.index(ruling[:40])


def test_the_side_by_side_pane_keeps_its_layout_while_a_take_reloads():
    """The flash was the page relaying out, not the video going black.

    `width:auto` lays a video out from its own intrinsic size, and a video
    whose src is being swapped has none -- it falls back to 300x150 until
    metadata arrives. In a centred flex row that resized both halves and
    re-centred the pair, so the whole picture area jumped and jumped back at
    every cut. It never happened on the finished cut because that is one
    element whose src never changes.
    """

    from montagewright.webapp import PAGE

    page = PAGE.read_text(encoding="utf-8")

    # Each half owns its width whatever the element inside it currently
    # knows about itself.
    assert ".frame.both > .half {" in page
    assert "flex: 1 1 0" in page
    # And is invisible to layout otherwise, so single-pane mode is unchanged.
    assert ".frame > .half { display: contents; }" in page

    # A second element to swap to, so crossing a cut does not call load().
    assert 'id="raw-next"' in page
    assert "function showTake(" in page and "function warmNext(" in page
    # Swapped by exchanging which one answers to the visible id, because
    # everything else on the page looks the take up that way.
    swap = page[page.index("function showTake("):]
    swap = swap[: swap.index("\n}\n")]
    assert "spare.id = 'raw-video'" in swap
    assert "raw.id = 'raw-next'" in swap
    # Only reloads when nothing already has it.
    assert swap.index("spare.getAttribute('src') === want") < swap.index("raw.load()")

    # Listeners are bound to the pane, not to one element: the two swap, and
    # a listener attached to the element would follow the wrong one.
    assert "$('raw-video').addEventListener" not in page
    assert "$('frame').addEventListener" in page


def test_a_crop_path_survives_a_round_trip_through_disk():
    """`write_crops` had no reader, so the record was written and ignored."""

    import tempfile
    from pathlib import Path

    from montagewright.executor import CropBox
    from montagewright.pipeline import read_crops, write_crops
    from montagewright.reframe import CropPath, Keyframe

    was = {
        "k00": CropPath([
            Keyframe(0.0, CropBox(0.10, 0.0, 0.3164, 1.0)),
            Keyframe(1.5, CropBox(0.42, 0.0, 0.3164, 1.0)),
            Keyframe(3.0, CropBox(0.68, 0.0, 0.3164, 1.0)),
        ]),
        "k01": CropPath([Keyframe(0.0, CropBox(0.0, 0.0, 0.5, 0.5))]),
    }
    with tempfile.TemporaryDirectory() as work:
        where = Path(work) / "crops.json"
        write_crops(was, where)
        back = read_crops(where)

    assert sorted(back) == ["k00", "k01"]
    assert [round(k.seconds, 3) for k in back["k00"].keyframes] == [0.0, 1.5, 3.0]
    assert [round(k.crop.x, 5) for k in back["k00"].keyframes] == [0.1, 0.42, 0.68]

    # A run that kept no record says so rather than raising: those exist.
    assert read_crops(Path(work) / "gone.json") == {}


def test_the_timeline_is_written_from_what_the_render_did():
    """FCPXML that disagrees with the film opens as a different cut.

    The exports rebuilt the crop paths from the cards with no client and no
    checkpoint. For a held frame that is the same arithmetic; for anything
    that followed a subject it is a guess, because that path came out of a
    mask propagation nothing there can repeat.
    """

    import inspect

    from montagewright import cli, webapp

    exporting = inspect.getsource(cli.command_timeline)
    assert exporting.index("read_crops(") < exporting.index("follow_subjects(")
    # And says so when there is no record, rather than quietly guessing.
    assert "will differ from the film" in exporting

    # A recut preserves the recorded source-time curve, shifting its original
    # interpolation domain rather than rebuilding without SAM.
    rebuilding = inspect.getsource(webapp.create_app)
    rebuilding = rebuilding[rebuilding.index("stored = read_crops("):]
    rebuilding = rebuilding[: rebuilding.index("plan = plan_render(")]
    assert "retime_crop_path(" in rebuilding
    assert "if stale:" in rebuilding


def test_a_walking_subject_is_followed_rather_than_averaged():
    """The looks path collapsed every observation into a mean.

    Frames are pulled across the shot and the boxes come back one per frame,
    and all of them were reduced to one point before anything downstream saw
    them -- so a subject that walked across the frame was handed on as a
    place in the middle of its own path, and a shot planned to follow it held
    there while the subject left.
    """

    from montagewright.reframe import build_look_path

    walked = [(0.0, 0.20, 0.5), (1.0, 0.50, 0.5), (2.0, 0.80, 0.5)]
    common = dict(
        source_aspect=16 / 9, target_aspect=9 / 16, duration_seconds=2.0,
        energy="dynamic",
    )
    # One look, resting the whole shot, on something that does not stay put.
    stops = [(2.0, 0.50, 0.5, 0.3164)]

    followed = build_look_path(stops, tracks=[walked], **common)
    centres = [k.crop.x + k.crop.width / 2 for k in followed.keyframes]
    assert len(followed.keyframes) >= 3, followed.keyframes
    assert centres == sorted(centres)
    assert max(centres) - min(centres) > 0.25, centres

    # Without the track it is the old behaviour: one place, held.
    held = build_look_path(stops, **common)
    still = {round(k.crop.x, 4) for k in held.keyframes}
    assert len(still) == 1

    # A subject that barely moved is not chased -- below the deadband the
    # frame would only jitter, and holding is what it should look like.
    from montagewright.reframe import DEADBAND

    twitch = [(0.0, 0.50, 0.5), (1.0, 0.50 + DEADBAND / 4, 0.5)]
    steady = build_look_path(stops, tracks=[twitch], **common)
    assert max(k.crop.x for k in steady.keyframes) - min(
        k.crop.x for k in steady.keyframes
    ) < DEADBAND


def test_a_track_is_read_where_the_frame_is_actually_looking():
    """The window a stop occupies is not the whole shot.

    The frame arrives at a subject partway through and leaves before the end,
    so sampling the track into the window by position would run the subject's
    movement at the wrong speed.
    """

    from montagewright.reframe import _across

    track = [(0.0, 0.0, 0.5), (1.0, 0.5, 0.5), (2.0, 1.0, 0.5)]

    # A window over the second half sees the second half of the walk.
    across = _across(track, 1.0, 2.0)
    assert [round(one[0], 3) for one in across] == [1.0, 2.0]
    assert [round(one[1], 3) for one in across] == [0.5, 1.0]

    # Edges are pinned and interior samples kept, so the shape survives.
    whole = _across(track, 0.0, 2.0)
    assert [round(one[0], 3) for one in whole] == [0.0, 1.0, 2.0]

    # A window past either end clamps rather than extrapolating.
    assert round(_across(track, 3.0, 4.0)[0][1], 3) == 1.0


def test_the_production_path_hands_the_tracks_to_the_builder():
    """A guard on the builder alone guards the wrong caller."""

    import inspect

    from montagewright import pipeline

    source = inspect.getsource(pipeline.follow_subjects)
    call = source[source.index("build_look_path("):]
    assert "tracks=tracks" in call[: call.index(")\n")]

    # And the sampler's timestamps reach the measurement, which is what
    # makes a box tie to a moment. They were being dropped by the caller.
    measuring = inspect.getsource(pipeline._measure_looks)
    assert "frames, times = _sample_frames(" in measuring
    assert "frame_index" in measuring


def test_every_look_that_promised_whole_is_checked_not_only_the_first():
    """`must_be_whole` moved onto the look and the check did not follow.

    `reframe.subject` is built from `looks[0]`, so a shot that settles on a
    face and then on a wordmark that has to be whole made its second promise
    to nobody -- which is the case the field was moved onto the look for.
    """

    from montagewright.schema import Look, reframe_of

    shot = {
        "looks": [
            {"at": "the presenter", "seconds": 1.0, "must_be_whole": False},
            {"at": "the wordmark", "seconds": 1.0, "must_be_whole": True},
        ],
        "energy": "medium",
    }
    reframe = reframe_of(shot)
    # The promise survives onto the look even though the first subject is
    # what `reframe.subject` is built from.
    assert reframe.subject is not None
    assert reframe.subject.min_visible < 0.99
    assert [one.must_be_whole for one in reframe.looks] == [False, True]

    import inspect

    from montagewright import pipeline

    checking = inspect.getsource(pipeline.follow_subjects)
    later = checking[checking.index("for index, look in enumerate(reframe.looks[1:]"):]
    later = later[: later.index("if reframe.subject is not None")]
    assert "look.must_be_whole" in later
    assert "find_subject(" in later
    assert "entity_id=look.entity_id" in later
    assert "box.width <= crop_width" in later


def test_the_usable_window_is_a_constraint_not_a_hint():
    """The card said where a take is worth cutting into and nothing read it.

    Outside the card itself, `usable_from_seconds` reached exactly two
    places: a dataclass field, and the line of prompt text built from it. A
    planner told "可用區間 4.2–8.5s" and choosing 2.0 was contradicted by
    nothing, so the shot opened on the camera still swinging -- the case the
    field was added for.
    """

    from montagewright.cli import _edl_from_selection
    from pathlib import Path
    import json
    import tempfile

    card = {
        "version": "x", "summary": "", "usable": True,
        "segments": [{"from": 4.2, "to": 8.5, "status": "eligible", "why": ""}],
        "subjects": [], "action": [],
    }
    with tempfile.TemporaryDirectory() as work:
        where = Path(work) / "c.json"
        where.write_text(json.dumps(card), encoding="utf-8")
        # As `expand_spans` leaves it: the span the planner named, already
        # resolved to a file and a second, with its edges carried along.
        selection = {"shots": [{
            "span_id": "C017:s00", "source_id": "C017",
            "start_seconds": 4.2, "seconds_needed": 3.0,
            "usable_from_seconds": 4.2, "usable_to_seconds": 8.5,
            "frame": "settles", "energy": "medium", "why": "",
            "looks": [{"at": "a", "seconds": 1.0, "framing": "thirds"}],
        }]}
        # load_card checks the version, so point the reader at a real one.
        from montagewright import clipcard

        was, clipcard.CARD_VERSION = clipcard.CARD_VERSION, "x"
        try:
            edl, _ = _edl_from_selection(selection, Path(work), {"C017": where})
        finally:
            clipcard.CARD_VERSION = was

    clip = edl.clips[0]
    assert clip.approx_in_seconds >= 4.2, clip.approx_in_seconds
    assert clip.approx_out_seconds <= 8.5 + 1e-6, clip.approx_out_seconds
    # And it travels with the clip, because the layers below decide lengths.
    assert clip.usable_window == (4.2, 8.5)


def test_reaching_a_beat_is_not_a_reason_to_run_into_the_reset():
    """Rhythm stretches shots to land on music and knew only the file length."""

    from montagewright.grounding import apply_to_edl
    from montagewright.schema import EDL, Clip

    clip = Clip(
        clip_id="k00", source_id="C017",
        approx_in_seconds=4.2, approx_out_seconds=6.0,
        usable_from_seconds=4.2, usable_to_seconds=8.5,
    )

    class _Entry:
        def __init__(self, clip, seconds):
            self.clip, self.duration_seconds = clip, seconds

    class _Timeline:
        def __init__(self, clips):
            self.clips = clips

    # Asked for six seconds from 4.2, which ends at 10.2 -- past the take.
    stretched = apply_to_edl(
        EDL(project_id="p", clips=[clip]), _Timeline([_Entry(clip, 6.0)])
    )
    assert stretched.clips[0].approx_out_seconds == 8.5

    # Inside the window it is left alone.
    kept = apply_to_edl(
        EDL(project_id="p", clips=[clip]), _Timeline([_Entry(clip, 2.0)])
    )
    assert kept.clips[0].approx_out_seconds == 6.2


def test_an_overrun_is_recorded_rather_than_silently_delivered():
    """The executor's only question was whether the time existed in the file.

    A second of a reset and a second of a take are the same second to it.
    """

    from montagewright.executor import _resolve_times
    from montagewright.schema import Clip

    clip = Clip(
        clip_id="k00", source_id="C017",
        approx_in_seconds=4.0, approx_out_seconds=9.5,
        usable_from_seconds=4.0, usable_to_seconds=8.5,
    )

    class _Source:
        duration_seconds = 30.0

    found: list = []
    _resolve_times(clip, _Source(), found, [])
    assert [one.ladder_other for one in found] == ["ran_past_the_usable_take"]
    assert found[0].measured["overrun_seconds"] == 1.0

    # No window, no complaint: most material has never been given one.
    plain = clip.model_copy(update={"usable_to_seconds": 0.0})
    quiet: list = []
    _resolve_times(plain, _Source(), quiet, [])
    assert quiet == []


def test_a_handle_does_not_open_onto_the_part_nobody_should_see():
    """Handles were bounded by the file, which is the wrong boundary.

    Half a second before a take is the camera being aimed; half a second
    after it is often somebody saying "again". A handle exists to be pulled,
    so one that opens onto a reset is worse than none.
    """

    import inspect

    from montagewright import renderer

    source = inspect.getsource(renderer)
    cutting = source[source.index("first = segment.usable_from_seconds"):]
    cutting = cutting[: cutting.index("command = [")]
    assert "segment.in_seconds - first" in cutting
    assert "last - segment.out_seconds" in cutting
    # Falls back to the file when nothing said otherwise.
    assert "or source.duration_seconds" in cutting


def test_an_action_snap_cannot_land_outside_the_take():
    """Actions are recorded across the whole clip, resets included.

    The camera being repositioned is a movement, and so is somebody walking
    in to reset a prop -- so the correction that exists to land a cut on a
    gesture could move it over the take's own boundary, on purpose.
    """

    from montagewright.clipcard import snap_to_action

    card = {"action": [
        {"what": "the hand reaches in", "from": "0:05", "to": "0:06"},
        {"what": "somebody resets the prop", "from": "0:09", "to": "0:10"},
    ]}

    # Unbounded, the nearest action to 8.6 is the reset.
    loose, _ = snap_to_action(card, 8.6, 2.0)
    assert loose == 9.0

    # Bounded by the take, the reset is not a candidate at all.
    held, note = snap_to_action(card, 8.6, 2.0, within=(4.2, 8.5))
    assert held == 8.6 and note is None

    # And a gesture inside the window is still snapped to.
    moved, note = snap_to_action(card, 5.4, 2.0, within=(4.2, 8.5))
    assert moved == 5.0 and note


def test_a_rejected_stretch_has_no_name_to_be_chosen_by():
    """The whole argument, in one assertion.

    A file and a second is always well formed: `C8330` plus 9.8 is a valid
    plan even when 9.8 lands in the middle of somebody saying "again". A span
    either exists or it does not, and the stretches that failed are simply
    not in the vocabulary the answer is written in.
    """

    from montagewright.spans import spans_of

    card = {"usable": True, "segments": [
        {"from": "0:00", "to": "0:03", "status": "reject", "why": "還在甩"},
        {"from": "0:03", "to": "0:09", "status": "eligible", "why": "第一次"},
        {"from": "0:09", "to": "0:11", "status": "reject", "why": "有人喊卡"},
        {"from": "0:11", "to": "0:18", "status": "eligible", "why": "第二次"},
        {"from": "0:18", "to": "0:22", "status": "reject", "why": "收器材"},
    ]}
    found = spans_of(card, "C8330", 22.0)

    # Two islands, not one window swallowing the water between them.
    assert [one.span_id for one in found] == ["C8330:s01", "C8330:s03"]
    assert [(one.starts_seconds, one.ends_seconds) for one in found] == [
        (3.0, 9.0), (11.0, 18.0)
    ]
    # The rejected stretches are absent, not marked.
    assert not [one for one in found if "喊卡" in one.why]

    # A take nobody segmented offers itself whole, which is exactly as much
    # as was known before any of this existed.
    assert [one.span_id for one in spans_of({"usable": True}, "C1", 8.0)] == ["C1:s00"]
    # A take the card called unusable offers nothing at all.
    assert spans_of({"usable": False}, "C2", 8.0) == []


def test_an_offset_cannot_walk_out_of_its_span():
    """The number the planner still writes is answered against the span."""

    from montagewright.spans import Span

    span = Span("C1:s01", "C1", 11.0, 18.0)

    assert span.at(0.0, 3.0) == (11.0, 14.0)
    assert span.at(2.0, 3.0) == (13.0, 16.0)
    # Past the end, pulled back so the whole shot still fits inside.
    assert span.at(99.0, 3.0) == (15.0, 18.0)
    # Longer than the span, shortened to it rather than running over.
    assert span.at(0.0, 99.0) == (11.0, 18.0)
    # Negative is not a way out either.
    assert span.at(-5.0, 2.0) == (11.0, 13.0)


def test_a_span_is_written_back_out_as_a_file_and_a_second():
    """Fourteen readers ask a shot for `source_id` and `start_seconds`.

    None of them needs to learn about spans to stay correct; what they needed
    was for those two fields to stop being the model's to invent. One place
    knows both shapes, which is the only way this project has survived
    changing one before.
    """

    from montagewright.planner import expand_spans
    from montagewright.spans import Span

    offered = [Span("C1:s00", "C1", 0.0, 5.0), Span("C1:s01", "C1", 11.0, 18.0)]
    chosen = {"shots": [
        {"span_id": "C1:s01", "start_offset_seconds": 2.0, "seconds_needed": 3.0},
        {"span_id": "C1:s00", "start_offset_seconds": 0.0, "seconds_needed": 2.0},
    ]}
    expand_spans(chosen, offered)

    assert [s["source_id"] for s in chosen["shots"]] == ["C1", "C1"]
    assert [s["start_seconds"] for s in chosen["shots"]] == [13.0, 0.0]
    # And the edges travel too, because rhythm and the executor decide
    # lengths after this and only know how long the file is.
    assert chosen["shots"][0]["usable_from_seconds"] == 11.0
    assert chosen["shots"][0]["usable_to_seconds"] == 18.0

    # A name nobody offered is left alone rather than invented around.
    stray = {"shots": [{"span_id": "C9:s07", "start_offset_seconds": 0.0}]}
    expand_spans(stray, offered)
    assert "source_id" not in stray["shots"][0]


def test_the_planner_is_offered_spans_and_not_files():
    """The schema's enum is the list of what exists."""

    from montagewright.planner import _selection_schema

    shot = _selection_schema(["C1:s00", "C1:s01"])["properties"]["shots"]["items"]
    assert "span_id" in shot["required"]
    assert shot["properties"]["span_id"]["enum"] == ["C1:s00", "C1:s01"]
    # The two fields it used to be free to write are gone from the contract.
    assert "source_id" not in shot["properties"]
    assert "start_seconds" not in shot["properties"]
    # And every time it writes is a clock reading, like every other time
    # that goes through a model here.
    for field in ("start_offset_seconds", "seconds_needed"):
        assert shot["properties"][field]["type"] == "string"
    assert shot["properties"]["looks"]["items"]["properties"]["seconds"]["type"] == "string"


def test_a_call_waits_as_long_as_that_call_is_worth():
    """One ceiling for every call is the largest call's ceiling.

    Twenty-five minutes was sized for a planning pass carrying seventy-four
    proxies, measured at six hundred seconds. The card pass inherited it and
    then wedged at clip sixty of seventy-four -- an open connection with no
    bytes moving, still waiting half an hour later, because the ceiling had
    not been reached rather than because nothing was wrong.
    """

    import inspect

    from montagewright import clipcard, planner

    assert "patience_seconds" in inspect.signature(planner.ask).parameters
    passed = inspect.getsource(planner.ask)
    assert 'request["timeout"] = float(patience_seconds)' in passed

    # A call about one short clip does not wait as long as one about the
    # whole shoot.
    card = inspect.getsource(clipcard.describe_clip)
    assert "patience_seconds=" in card
    seconds = float(card.split("patience_seconds=")[1].split(",")[0])
    assert seconds < planner.REQUEST_TIMEOUT_MS / 1000.0

    # And the default is unchanged, so the long calls it was sized for keep it.
    assert planner.ask.__defaults__ is None
    assert inspect.signature(planner.ask).parameters[
        "patience_seconds"
    ].default is None


def test_the_inspector_names_the_move_the_shot_actually_makes():
    """A shot has not carried `camera_move` since the looks refactor.

    The panel read it off the plan, where it stopped existing, and printed an
    em dash -- so a shot that pushed in reported no move at all, in the panel
    whose job is saying what was planned. The server had the answer the whole
    time: the block carries it, worked out by the one reader that knows how
    to turn a list of looks into a move name.
    """

    from montagewright.webapp import PAGE

    page = PAGE.read_text(encoding="utf-8")
    said = page[page.index("<span>運鏡</span>"):]
    said = said[: said.index("</span>", said.index("${")) + 7]
    assert "b.camera_move" in said
    assert "plan.camera_move" not in page

    # And the declaration sits beside it, so a plan that said it would move
    # and did not is visible without opening the report.
    assert "plan.frame" in said


def test_a_shot_that_planned_to_hold_is_not_a_downgraded_follow():
    """Fourteen of sixteen shots were recorded as substitutions.

    One look means "stay on this", which for something that walks is a follow
    and for something standing still is a held frame. Both are the plan being
    carried out, and until the plan said so out loud there was no way to tell
    them apart -- so every settled shot came back carrying a note saying the
    frame held still on a shot whose plan was to hold still.

    That is not free. Each degradation is a question the shot reviewer has to
    adjudicate, and the shot reviewer is a paid call per shot.
    """

    from montagewright.reframe import Observation, build_crop_path
    from montagewright.schema import reframe_of

    still = [Observation(seconds=0.0, centre_x=0.5, centre_y=0.5,
                         width=0.2, height=0.4)]
    common = dict(source_aspect=16 / 9, target_aspect=9 / 16, clip_id="k00")

    asked_to_hold: list = []
    build_crop_path(still, degradations=asked_to_hold,
                    planned_to_move=False, **common)
    assert [one.ladder for one in asked_to_hold] == []

    asked_to_move: list = []
    build_crop_path(still, degradations=asked_to_move,
                    planned_to_move=True, **common)
    assert [one.ladder for one in asked_to_move] == ["static_on_subject"]

    # The declaration travels on the reframe, from the field the planner
    # answers before it writes a single look.
    assert reframe_of({"looks": [{"at": "a"}], "frame": "settles"}).planned_to_move is False
    assert reframe_of({"looks": [{"at": "a"}], "frame": "travels"}).planned_to_move is True
    # Two looks is a move whatever the declaration says, because it is one.
    assert reframe_of({"looks": [{"at": "a"}, {"at": "b"}]}).planned_to_move is True


def test_the_spend_cap_reads_the_same_on_an_upload():
    """A 429 on the other API surface was a crash rather than an ending.

    `ask` has translated this since the first time it happened. Uploads went
    straight past it, so a run with a finished film, a written report and
    every card paid for died on `files.upload` with a raw traceback --
    recorded as broken rather than as out of money, which are two different
    things to do next.
    """

    import inspect

    from montagewright import uploads
    from montagewright.cost import BudgetSpent

    source = inspect.getsource(uploads.upload_now)
    assert "_is_spend_cap(error)" in source
    assert "BudgetSpent(" in source
    assert "ai.studio/spend" in source

    class _Capped:
        class files:
            @staticmethod
            def upload(**_):
                raise RuntimeError(
                    "429 RESOURCE_EXHAUSTED: Your project has exceeded its "
                    "monthly spending cap."
                )

    import tempfile
    from pathlib import Path

    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as handle:
        handle.write(b"x")
        where = Path(handle.name)
    with pytest.raises(BudgetSpent):
        uploads.upload_now(where, _Capped())


def test_a_cloud_service_account_key_explains_why_file_upload_cannot_use_it():
    """An Enterprise key is valid, but not for the Developer Files API."""

    from montagewright import uploads

    class _EnterpriseKey:
        class files:
            @staticmethod
            def upload(**_):
                raise RuntimeError(
                    "401 UNAUTHENTICATED: ACCESS_TOKEN_TYPE_UNSUPPORTED "
                    "google.ai.generativelanguage.v1beta.FileService.CreateFile"
                )

    import tempfile
    from pathlib import Path

    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as handle:
        handle.write(b"x")
        where = Path(handle.name)
    with pytest.raises(RuntimeError, match="service-account-backed"):
        uploads.upload_now(where, _EnterpriseKey())


def _grid(bpm=120.0, bars=8):
    """A clean grid: downbeat every four beats, accents on the third."""

    from montagewright.grounding import BeatGrid, Cue

    period = 60.0 / bpm
    cues = []
    for beat in range(bars * 4):
        at = round(beat * period, 6)
        kind = "downbeat" if beat % 4 == 0 else (
            "accent" if beat % 4 == 2 else "beat"
        )
        cues.append(Cue(cue_id=f"c{beat:03d}", time_seconds=at, kind=kind))
    return BeatGrid(bpm=bpm, meter=4, cues=tuple(cues),
                    duration_seconds=bars * 4 * period)


def test_a_cut_lands_on_the_downbeat_rather_than_the_nearest_beat():
    """Nearest by distance is almost never the right musical event.

    The cue list runs three hundred beats against seventy-six downbeats, so
    the closest one is usually an ordinary beat. On a finished cut that put
    every one of the first six shots on the fourth beat of its bar -- one
    beat before the bar turned over, every time -- and the report called it
    sixteen cuts out of sixteen landed on a musical event, which was true and
    said nothing.
    """

    grid = _grid()
    period = 60.0 / 120.0

    # Just past the third beat of a bar. The nearest cue is that beat; the
    # downbeat is half a beat further and is what an editor would cut on.
    wanted = 3 * period + 0.05
    landed = grid.nearest_cue(wanted)
    assert landed.kind == "downbeat"
    assert landed.time_seconds == round(4 * period, 6)

    # An accent beats a plain beat by the same rule.
    landed = grid.nearest_cue(1 * period + 0.05)
    assert landed.kind == "accent"

    # Nothing is dragged. Halfway between the second and third beats of a
    # bar, every downbeat is more than a beat away, so none is reached for --
    # a downbeat two beats off is a different edit, not this edit placed
    # better. The accent inside the window still beats the plain beat.
    landed = grid.nearest_cue(2.5 * period)
    assert landed.kind == "accent"
    assert landed.time_seconds == round(2 * period, 6)

    # And a section boundary outranks a downbeat, because that is where the
    # music itself changes.
    from montagewright.grounding import BeatGrid, Cue

    both = BeatGrid(bpm=120.0, meter=4, duration_seconds=10.0, cues=(
        Cue(cue_id="d", time_seconds=1.00, kind="downbeat"),
        Cue(cue_id="s", time_seconds=1.10, kind="section_boundary"),
    ))
    assert both.nearest_cue(1.02).kind == "section_boundary"


def test_a_cut_review_note_at_a_timecode_names_the_shot_it_lands_in():
    """A fault the finished-film pass found, dropped for want of a clip_id.

    The whole-cut reviewer reports a problem at a time -- dust on a screen at
    0:10, a coin cropped at 0:10 -- and often names no shot, so clip_id is
    null. The replan loop acted only on shots the per-shot pass marked
    undelivered, so the run stopped with "revision asked for, but no shot was
    named" and shipped the fault. The timeline knows what is on screen at
    0:10; the note maps to it.
    """

    from montagewright.cli import _shot_at_second

    # Three shots of 4, 3 and 5 seconds: windows 0-4, 4-7, 7-12.
    rhythm = {
        "k00": {"seconds": 4.0},
        "k01": {"seconds": 3.0},
        "k02": {"seconds": 5.0},
    }
    assert _shot_at_second(0.0, rhythm) == "k00"
    assert _shot_at_second(5.0, rhythm) == "k01"
    assert _shot_at_second(10.0, rhythm) == "k02"
    # Past the last cut belongs to the last shot, not to nothing.
    assert _shot_at_second(99.0, rhythm) == "k02"
    # No timeline is the only case that names nothing.
    assert _shot_at_second(1.0, {}) is None


def test_a_look_at_something_outside_this_window_is_a_disagreement():
    """Two facts on record, never compared, and a pan that was a hold.

    A shot asked to sweep across a row of watches took two seconds of a
    seven-second pan, and named as the far end of the sweep a subject the
    take does not reach until six seconds after this shot cuts away. The
    card had measured when that subject was seen; the motion pass had
    measured what the frame did in between. Nothing put the two together,
    so the grounding pass went looking for the watches in frames they are
    not in, found the nearest thing resembling them, and the frame travelled
    0.013 widths -- delivered, reported, and wrong until somebody watched.

    The numbers here are that run's. Every shot that delivered what it
    planned sat at 0.035 widths or below.
    """

    from montagewright.motion import MotionInterval
    from montagewright.planner import MaterialItem, frame_disagreements

    def moving(starts, ends, travel):
        return MotionInterval(
            event_id="m00", starts_seconds=starts, ends_seconds=ends,
            state="moving", peak_vw_s=0.06, travel_vw=travel, settles=True,
        )

    material = [MaterialItem(
        source_id="C8329", duration_seconds=16.5, summary="",
        sightings=(("右前方的綠色錶帶智慧手錶", 2.0), ("左側平移後出現的手錶", 12.0)),
        motion=(moving(4.0, 11.0, 0.34),),
        # What a 16:9 source is delivered through at 9:16.
        crop_width=0.3164,
    )]
    shot = {
        "source_id": "C8329", "frame": "travels",
        "start_seconds": 4.0, "seconds_needed": 2.0,
        "looks": [
            {"at": "右前方的綠色錶帶智慧手錶"},
            {"at": "左側平移後出現的手錶"},
        ],
    }

    off = frame_disagreements([shot], material)
    assert len(off) == 1
    assert "左側平移後出現的手錶" in off[0]
    assert "0:12.0" in off[0]

    # The near end of the same sweep is inside the window and is not flagged:
    # this asks whether the subject is reachable, not whether the take moves.
    assert "右前方" not in off[0]

    # And on a take whose camera never moves, a position measured at any
    # moment describes every moment, which is why most shots pass.
    still = [MaterialItem(
        source_id="C8329", duration_seconds=16.5, summary="",
        sightings=material[0].sightings, motion=(), crop_width=0.3164,
    )]
    assert frame_disagreements([shot], still) == []

    # Without any material the older checks still run on their own.
    assert frame_disagreements([shot]) == []

    # A stretch in between that no camera movement accounts for -- a lens
    # uncovered, a subject filling the frame -- contributes no offset, and
    # summing offsets would certify it as never having moved. That is the
    # error this module was rewritten to stop making, so it says unknown.
    unreadable = [MaterialItem(
        source_id="C8329", duration_seconds=16.5, summary="",
        sightings=material[0].sightings, crop_width=0.3164,
        motion=(MotionInterval(
            event_id="m00", starts_seconds=4.0, ends_seconds=11.0,
            state="not_a_shift", peak_vw_s=0.0, travel_vw=0.0, settles=False,
        ),),
    )]
    # Both looks, because the stretch nothing explains covers the reach from
    # either sighting to this window -- including the near one that a
    # measured pan would have carried.
    unknown = frame_disagreements([shot], unreadable)
    assert len(unknown) == 2
    assert all("unknown" in one for one in unknown)

    # The same drift on a wider delivery is not a problem, because the
    # number is half the crop rather than a constant. Delivered at its own
    # aspect this source is not cropped at all, so a subject that would have
    # left a vertical frame is still comfortably inside this one -- which is
    # why the first version of this check, a flat 0.15, was a rule about one
    # shoot wearing the clothes of a general one.
    roomy = [MaterialItem(
        source_id="C8329", duration_seconds=16.5, summary="",
        sightings=material[0].sightings, motion=material[0].motion,
        crop_width=1.0,
    )]
    assert frame_disagreements([shot], roomy) == []


def test_a_cut_records_what_kind_of_event_it_landed_on():
    """The count was true and could not be checked, which is worse than false.

    A cue id is a position in a sorted list of candidates of every kind, so
    `mc-00040` reads back as nothing. Six cuts running one beat ahead of the
    bar were reported as six cuts on the music, and the only way anyone found
    out was by listening. Ranking cues by strength fixed the cutting; this is
    what makes the claim answerable next time without listening first.
    """

    from montagewright.grounding import ground_timeline
    from montagewright.schema import EDL, Clip, MusicSync

    grid = _grid()
    period = 60.0 / 120.0
    asked = EDL(project_id="p", clips=[Clip(
        clip_id="k00", source_id="C1",
        approx_in_seconds=0.0, approx_out_seconds=4 * period + 0.05,
        music_sync=MusicSync(cut_on_beat=True),
    )])
    landed = ground_timeline(asked, grid).clips[0]
    assert landed.landed_on is not None
    assert landed.landed_kind == "downbeat"

    # And the grid can be asked about a cue by name, which is what a shot
    # naming its own sync point needs.
    assert grid.cue(landed.landed_on).kind == "downbeat"
    assert grid.cue("nothing-by-this-name") is None


def test_the_preference_covers_every_cuttable_kind():
    """A kind nobody ranked would sort last by accident rather than by choice."""

    from montagewright.grounding import CUTTABLE, CUT_PREFERENCE

    assert set(CUT_PREFERENCE) == set(CUTTABLE)
    assert CUT_PREFERENCE["downbeat"] < CUT_PREFERENCE["beat"]
    assert CUT_PREFERENCE["section_boundary"] < CUT_PREFERENCE["downbeat"]


def test_the_replan_is_checked_for_the_same_disagreement_as_the_selection():
    """One check, two paths, and only one of them had it.

    A replan is where a shot that failed for want of a move is most likely to
    be answered with the word rather than the thing. It happened on the first
    round of the first film that reached this loop: the reasoning said the
    frame would sweep across the wordmark, the looks named one place, and the
    executor held -- so the next review reported the same clipped title and
    the round was spent.
    """

    import inspect

    from montagewright import cli, planner

    replanning = inspect.getsource(planner.replan_shots)
    assert "frame_disagreements(" in replanning
    # After the spans are expanded, so a shot is in its final shape.
    assert replanning.index("expand_spans(") < replanning.index(
        "frame_disagreements("
    )

    # And the caller prints them, next to the replan lines they belong to.
    rendering = inspect.getsource(cli.command_render)
    assert 'replanned.get("frame_disagreements")' in rendering


def test_two_looks_that_landed_in_the_same_place_are_reported():
    """A move that goes nowhere reads as a plan being carried out.

    There are two looks, the frame travels, the report says `pan`. On screen
    it is a hold. It happens when both looks describe the same thing instead
    of its two ends -- a wordmark asked to be read across came back as a
    frame drifting from "y Unpa" to "acked", a fifth of the title in shot
    throughout, and every layer agreed it had swept.

    Not catchable by comparing the plan against itself: it has the two looks
    it promised. Only the measurement knows they are the same place.
    """

    from montagewright.reframe import DEADBAND, build_look_path

    common = dict(source_aspect=16 / 9, target_aspect=9 / 16,
                  duration_seconds=4.0, energy="active", clip_id="k15")

    nowhere: list = []
    build_look_path(
        [(0.5, 0.50, 0.5, 0.3164), (0.5, 0.505, 0.5, 0.3164)],
        degradations=nowhere, **common,
    )
    assert [one.ladder_other for one in nowhere] == ["looks_landed_on_the_same_place"]
    assert nowhere[0].measured["total_travel_vw"] < DEADBAND

    # A real journey says nothing.
    somewhere: list = []
    build_look_path(
        [(0.5, 0.20, 0.5, 0.3164), (0.5, 0.80, 0.5, 0.3164)],
        degradations=somewhere, **common,
    )
    assert "looks_landed_on_the_same_place" not in [
        one.ladder_other for one in somewhere
    ]

    # And a shot that only ever asked for one look is not a failed move.
    held: list = []
    build_look_path([(1.0, 0.5, 0.5, 0.3164)], degradations=held, **common)
    assert held == []


def test_a_zoom_leg_follows_the_subject_in_delivered_screen_space():
    """A small source drift must not reverse when a crop closes around it.

    Two looks at one subject are how the planner expresses a push.  The
    multi-look path used to follow the measured track only during the rests,
    then aim the whole zoom leg at the track's mean.  A subject that moved a
    little to the right before the push was consequently magnified to the
    left: both motions were individually smooth and their composite visibly
    reversed.
    """

    from montagewright.reframe import build_look_path

    track = [
        (0.0, 0.552, 0.50),
        (0.8, 0.564, 0.50),
        (1.6, 0.565, 0.50),
        (2.4, 0.560, 0.50),
        (3.2, 0.558, 0.50),
        (4.0, 0.558, 0.50),
    ]
    path = build_look_path(
        [(0.35, 0.5595, 0.50, 0.3164),
         (0.35, 0.5595, 0.50, 0.2083)],
        source_aspect=16 / 9,
        target_aspect=9 / 16,
        duration_seconds=4.0,
        energy="active",
        tracks=[track, track],
    )

    def subject_at(when: float) -> float:
        for before, after in zip(track, track[1:]):
            if before[0] <= when <= after[0]:
                share = (when - before[0]) / (after[0] - before[0])
                return before[1] + (after[1] - before[1]) * share
        return track[-1][1]

    positions = [
        (subject_at(frame.seconds) - frame.crop.x) / frame.crop.width
        for frame in path.keyframes
        if 0.7 <= frame.seconds <= 3.3
    ]
    assert len(positions) >= 4
    assert max(positions) - min(positions) < 0.02


def test_a_named_music_point_is_actually_resolved():
    """`resolve_sync_point` existed from the first day and nothing called it.

    A shot asked to land on a named section was placed by `nearest_cue` like
    every other -- which finds the event nearest the requested length, and
    the requested length is the thing being overridden. The planner named a
    moment in the music and got the moment nearest to where it would have cut
    anyway.
    """

    from montagewright.grounding import (
        BeatGrid, Cue, apply_to_edl, ground_timeline,
    )
    from montagewright.schema import EDL, Clip, MusicSync

    grid = BeatGrid(bpm=120.0, meter=4, duration_seconds=40.0, cues=tuple(
        [Cue(cue_id="section_001", time_seconds=0.0, kind="section_boundary"),
         Cue(cue_id="section_002", time_seconds=12.0, kind="section_boundary")]
        + [Cue(cue_id=f"b{i:03d}", time_seconds=i * 0.5,
               kind="downbeat" if i % 4 == 0 else "beat") for i in range(80)]
    ))

    def cut(sync_to=None, hold=4.0):
        return EDL(project_id="p", clips=[Clip(
            clip_id="k00", source_id="C1",
            approx_in_seconds=0.0, approx_out_seconds=hold,
            music_sync=MusicSync(cut_on_beat=True, sync_to=sync_to),
        )])

    # Asked for four seconds and for the section at twelve: the section wins.
    landed = ground_timeline(cut(sync_to="section_002"), grid).clips[0]
    assert abs(landed.duration_seconds - 12.0) < 1e-6
    assert landed.landed_on == "section_002"

    # Without it, four seconds is what it gets, on the grid.
    plain = ground_timeline(cut(), grid).clips[0]
    assert abs(plain.duration_seconds - 4.0) < 0.5

    # A name this track does not have is a note, not a crash -- the planner
    # described intent the material cannot serve.
    missing = ground_timeline(cut(sync_to="chorus_1_start"), grid).clips[0]
    assert abs(missing.duration_seconds - 4.0) < 0.5
    assert "chorus_1_start" in (missing.note or "")

    # And a point already behind this shot cannot be reached backwards.
    late = EDL(project_id="p", clips=[
        Clip(clip_id="k00", source_id="C1", approx_in_seconds=0.0,
             approx_out_seconds=20.0, music_sync=MusicSync(cut_on_beat=True)),
        Clip(clip_id="k01", source_id="C1", approx_in_seconds=0.0,
             approx_out_seconds=4.0,
             music_sync=MusicSync(cut_on_beat=True, sync_to="section_002")),
    ])
    second = ground_timeline(late, grid).clips[1]
    assert "before this shot begins" in (second.note or "")


def test_cuts_are_placed_against_the_music_that_is_playing():
    """The grid measures the file; grounding counts from the first frame.

    Those are the same clock only when the bed starts at the track's
    beginning, and the rhythm pass is told outright that a thirty-second cut
    rarely wants the first thirty seconds of a two-minute piece. Pick 0:30 and
    the film's fourth second was matched against the track's fourth second
    while the thirty-fourth was sounding -- every cut placed against music
    nobody could hear. It never showed because that field has so far always
    been left at zero, and it exists to be moved.
    """

    from montagewright.grounding import (
        BeatGrid, Cue, apply_to_edl, ground_timeline,
    )
    from montagewright.schema import EDL, Clip, MusicSync

    # A downbeat every two seconds, for a minute.
    grid = BeatGrid(bpm=120.0, meter=4, duration_seconds=60.0, cues=tuple(
        Cue(cue_id=f"d{i:02d}", time_seconds=i * 2.0, kind="downbeat")
        for i in range(30)
    ))

    def film(music_from=0.0, spans=None):
        edl = EDL(project_id="p", clips=[Clip(
            clip_id="k00", source_id="C1", approx_in_seconds=0.0,
            approx_out_seconds=4.2, music_sync=MusicSync(cut_on_beat=True),
        )])
        if music_from:
            edl = edl.model_copy(update={"music_from_seconds": music_from})
        if spans:
            edl = edl.model_copy(update={"music_spans": spans})
        return ground_timeline(edl, grid).clips[0]

    # From the top, 4.2s lands on the downbeat at 4.0.
    assert abs(film().duration_seconds - 4.0) < 1e-6

    # Starting the bed at 30.5s, the film's downbeats fall on the half
    # second: what is heard at film time 3.5 is the track's 34.0.
    shifted = film(music_from=30.5)
    assert abs(shifted.duration_seconds - 3.5) < 1e-6

    # A cue outside what gets played is not a place to cut, because it is
    # not audible -- so it is dropped rather than moved to where it would
    # have been.
    heard = grid.as_heard(30.5)
    assert min(c.time_seconds for c in heard.cues) >= 0.0
    assert heard.duration_seconds > 0

    # Spliced music maps through the joins in order.
    joined = grid.as_heard(0.0, [(10.0, 14.0), (40.0, 44.0)])
    assert joined.duration_seconds == 8.0
    # The track's 40.0 is the fourth second of the film.
    assert any(abs(c.time_seconds - 4.0) < 1e-6 for c in joined.cues)
    # And nothing from the discarded middle survives.
    assert not any(14.0 < c.time_seconds < 40.0 for c in joined.cues)


def test_the_camera_motion_the_model_cannot_see_is_measured():
    """A frame a second cannot show what happens between frames.

    Asked whether a take is stable, Gemini answered "固定鏡頭…畫面穩定清晰"
    about one whose first three seconds are the operator still finding the
    frame. Not carelessly -- the question cannot be answered from a series of
    individually sharp stills.
    """

    from montagewright.motion import MotionInterval, describe

    found = [
        MotionInterval("m00", 0.0, 2.5, "moving", 0.31, 0.42, settles=True),
        MotionInterval("m01", 2.5, 9.5, "still", 0.001, 0.002, settles=False),
    ]
    said = describe(found)
    # Ids, so an answer can point at one rather than invent a second.
    assert "m00" in said and "m01" in said
    # Clock readings, like every other time a model reads here.
    assert "0:00.0" in said and "0:02.5" in said
    # And it says the movement happened, never whether it was any good.
    assert "位移" in said
    for judgement in ("不能用", "reject", "壞", "失敗"):
        assert judgement not in said


def test_a_setup_reframe_is_not_offered_as_a_span():
    """The camera getting ready is the gap between takes, not a take.

    C8340's card described one of its stretches as 攝影機構圖調整 -- the
    operator moving from one framing to another -- marked it eligible, and a
    replan chose it to escape an overexposed opening. Nothing gains a viewer
    anything there.
    """

    from montagewright.spans import spans_of

    def card(role):
        return {"usable": True, "segments": [
            {"from": "0:00", "to": "0:05", "status": "eligible",
             "why": "x", "motion_role": role},
        ]}

    for allowed in ("locked", "authored", "subject_follow", "handheld_texture"):
        assert spans_of(card(allowed), "C1", 5.0), allowed
    # Getting ready, recovering from a knock, and "I cannot tell" are all
    # refused -- the last because choosing anyway turns a no into a yes.
    for refused in ("setup_reframe", "disturbance", "unknown"):
        assert spans_of(card(refused), "C1", 5.0) == [], refused

    # The role travels with the span, because the executor needs it.
    assert spans_of(card("authored"), "C1", 5.0)[0].motion_role == "authored"


def test_a_digital_move_on_a_take_that_already_moves_is_reported():
    """The prompt has warned about this since it was written.

    Two movements stacked fight each other and the real one wins. The warning
    was advice: `camera_motion` reached the selection prompt and no field
    downstream held it, so there were never two halves to compare.
    """

    import inspect

    from montagewright import pipeline
    from montagewright.schema import reframe_of

    # It reaches the reframe from the span the planner named.
    shot = {
        "looks": [{"at": "a"}, {"at": "b"}],
        "frame": "travels", "source_motion_role": "authored",
    }
    built = reframe_of(shot)
    assert built.planned_to_move and built.source_motion_role == "authored"

    checking = inspect.getsource(pipeline.follow_subjects)
    assert "digital_move_on_a_moving_take" in checking
    # A locked take is not flagged for having a move added to it.
    assert 'in {"authored", "subject_follow"}' in checking


def test_each_span_tells_selection_its_own_source_motion_role():
    from montagewright.planner import MaterialItem, _describe_material
    from montagewright.spans import Span

    item = MaterialItem(
        source_id="C1", duration_seconds=8.0, summary="phone",
        spans=(
            Span("C1:s00", "C1", 0.0, 3.0, "", "authored"),
            Span("C1:s01", "C1", 3.0, 8.0, "", "locked"),
        ),
    )
    described = _describe_material([item])
    assert "C1:s00" in described and "原素材運動=authored" in described
    assert "C1:s01" in described and "原素材運動=locked" in described


def test_direction_owns_shot_density_and_selection_enforces_a_range():
    from montagewright.planner import (
        _direction_schema, _selection_schema, _shot_count_bounds,
    )

    direction = _direction_schema()
    for field in (
        "target_shot_count", "typical_shot_seconds", "max_static_seconds",
        "pacing_reason",
    ):
        assert field in direction["required"]
    lower, upper = _shot_count_bounds({"target_shot_count": 30}, 74)
    shots = _selection_schema(
        [f"C{i}:s00" for i in range(74)],
        min_shots=lower, max_shots=upper,
    )["properties"]["shots"]
    assert "26 to 34" in shots["description"]
    assert "minItems" not in shots and "maxItems" not in shots


def test_overlapping_adjacent_windows_of_one_span_are_reported():
    from montagewright.planner import sequence_disagreements

    shots = [
        {"source_id": "C1", "span_id": "C1:s00", "start_seconds": 0,
         "seconds_needed": 7},
        {"source_id": "C1", "span_id": "C1:s00", "start_seconds": 1,
         "seconds_needed": 5},
    ]
    assert "overlapping windows" in sequence_disagreements(shots)[0]


def test_an_explicit_follow_does_not_reuse_one_card_box_as_a_trajectory():
    import inspect

    from montagewright import pipeline
    from montagewright.schema import reframe_of

    built = reframe_of({
        "camera_intent": "follow_subject", "looks": [{"at": "phone"}]
    })
    assert built.camera_move == "follow_subject" and built.planned_to_move
    source = inspect.getsource(pipeline.follow_subjects)
    assert 'known is not None and move != "follow_subject"' in source


def test_unreadable_footage_says_so_rather_than_saying_still():
    """"The camera did not move" and "this cannot be read" are not the same.

    The module claimed to report the second and only ever produced the first,
    because nothing measured how well the estimate fitted. A docstring
    describing behaviour the code does not have is the same fault this
    project keeps finding elsewhere.
    """

    import subprocess
    import tempfile
    from pathlib import Path

    from montagewright.motion import NOT_A_SHIFT, measure

    with tempfile.TemporaryDirectory() as work:
        noise = Path(work) / "noise.mp4"
        made = subprocess.run(
            ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
             "-f", "lavfi",
             "-i", "nullsrc=s=320x180:d=3:r=25,geq=random(1)*255:128:128",
             "-c:v", "libx264", "-crf", "18", str(noise)],
            capture_output=True,
        )
        if made.returncode != 0 or not noise.exists():
            import pytest as _pytest

            _pytest.skip("ffmpeg cannot synthesise noise here")
        found = measure(noise, 3.0)

    # By construction no shift explains anything, so no shift is claimed.
    assert [one.state for one in found] == ["not_a_shift"]

    # And the threshold sits between what real footage leaves behind and
    # what noise does -- the first number chosen was above noise, so the
    # state it was added for could never have occurred.
    assert 12.2 < NOT_A_SHIFT < 22.7


def _pan_across(work, across_px_s, down_px_s=0, seconds=3.0):
    """A still plate with a 320x180 window travelling across it.

    Still is the point: an earlier version of this drew `random(1)` per
    frame, so every frame was a fresh field of noise and there was nothing
    for any offset to match -- the measurement correctly reported that
    nothing explained the change, and the test read it as a bug in the
    measurement. A repeating pattern is no better: `mod(X*3+Y*5,256)` has a
    period of 85 pixels, so a shift of 85 scores perfectly and the reading
    is whatever aliasing decides.

    The window is 320 wide, so a pan of N pixels a second is exactly
    N/320 frame widths a second, whatever the analysis is scaled to.
    """

    import subprocess
    from pathlib import Path

    plate = Path(work) / "plate.png"
    subprocess.run(
        ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-f", "lavfi",
         "-i", "nullsrc=s=160x45,geq=lum='random(1)*255':cb=128:cr=128",
         "-frames:v", "1",
         "-vf", "scale=2560:1440:flags=bicubic,gblur=sigma=3", str(plate)],
        capture_output=True,
    )
    made = Path(work) / "pan.mp4"
    built = subprocess.run(
        ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
         "-loop", "1", "-i", str(plate), "-t", str(seconds), "-r", "25",
         "-vf", f"crop=320:180:'min(t*{across_px_s}\\,2200)'"
                f":'min(t*{down_px_s}\\,1200)'",
         "-c:v", "libx264", "-crf", "14", "-pix_fmt", "yuv420p", str(made)],
        capture_output=True,
    )
    if built.returncode != 0 or not made.exists() or not plate.exists():
        import pytest as _pytest

        _pytest.skip("ffmpeg cannot synthesise a pan here")
    return made


def test_a_camera_that_only_tilts_is_still_a_camera_that_moved():
    """The search ran sideways only, so tilting was inexplicable.

    Every offset tried was horizontal, which means a frame that slid up or
    down matched nothing at any of them, left a residual no shift accounted
    for, and was reported as not-a-shift -- the state reserved for changes
    that are not the camera moving. Two clips of a twelve-clip sample were
    reported as locked-off throughout while the operator was tilting.

    The user found this by watching a file the measurement had called
    unreadable and seeing an ordinary shot.
    """

    import tempfile

    from montagewright.motion import measure

    with tempfile.TemporaryDirectory() as work:
        found = measure(_pan_across(work, 0, 90), 3.0)

    assert found[0].state == "moving"
    assert not [one for one in found if one.state == "not_a_shift"]


def test_the_speed_a_pan_is_measured_at_is_the_speed_it_was_shot_at():
    """A search radius is a ceiling, and a ceiling reads as a wall.

    The reach was eight pixels at the analysed width, which caps any reading
    at 8/192*4 frame widths per second -- and three clips came back at
    exactly that figure, pinned against the edge of the search rather than
    measured. Raising it to twenty-four moved the wall and left it standing:
    a faster pan would have found it. What removes it is looking on a
    smaller copy first, where a movement too big to find is small, so the
    only limit left is how much picture two frames still share.

    Checked against pans of known speed rather than against another
    estimate agreeing with it, because two approximations can be wrong
    together. The 320-wide window makes the answer arithmetic: N pixels a
    second is N/320 frame widths a second.

    Readings run about a tenth high. Sampling 25fps material at 4 leaves
    consecutive samples six or seven source frames apart, and a peak picks
    the seven -- a bias the thresholds share, since they were calibrated
    through the same sampler.
    """

    import tempfile

    from montagewright.motion import COARSE_FPS, SHARED, measure

    # Well past what the old fixed reach could see: 0.5 widths a second was
    # its ceiling and the fastest of these is three times that.
    for speed in (60, 240, 480):
        with tempfile.TemporaryDirectory() as work:
            found = measure(_pan_across(work, speed), 3.0)
        wanted = speed / 320.0
        fastest = max(one.peak_vw_s for one in found)
        assert found[0].state == "moving", speed
        assert abs(fastest - wanted) / wanted < 0.2, (speed, fastest, wanted)

    # And the ceiling that remains is the one physics imposes: past half a
    # frame there is not enough shared picture for any offset to be evidence.
    assert (1.0 - SHARED) * COARSE_FPS == 2.0


def test_a_movement_too_brief_for_the_model_to_see_is_not_split_out():
    """Asking about something invisible is how footage gets deleted.

    The measurement resolves a three-quarter-second movement; the model reads
    video at a frame a second, so both of its sampled frames can fall either
    side of one. Split out anyway, it becomes an interval with no picture
    behind it, the honest answer is `unknown`, and `unknown` earns no span --
    two reasonable rules combining to throw the take away.

    That the model cannot be given a finer rate is measured rather than
    assumed: the same clip sent with an fps hint and without came back at an
    identical token count, so the Interactions API ignores it.
    """

    from montagewright.motion import SEEN_SECONDS, _into_intervals

    # Still, a blink of movement, still. At four samples a second.
    shifts = (
        [(i / 4, 0.001, 0.2) for i in range(1, 21)]      # 5s still
        + [(5 + i / 4, 0.30, 0.2) for i in range(1, 3)]  # 0.5s moving
        + [(5.5 + i / 4, 0.001, 0.2) for i in range(1, 21)]
    )
    found = _into_intervals(shifts, 10.5)
    assert [one.state for one in found] == ["still"], [
        (one.event_id, one.state, one.seconds) for one in found
    ]

    # A movement long enough to have been seen is kept.
    longer = (
        [(i / 4, 0.001, 0.2) for i in range(1, 21)]
        + [(5 + i / 4, 0.30, 0.2) for i in range(1, 13)]   # 3s moving
        + [(8 + i / 4, 0.001, 0.2) for i in range(1, 21)]
    )
    assert "moving" in [one.state for one in _into_intervals(longer, 13.0)]

    # And a clip that opens with a brief one has nothing behind it to be
    # absorbed into, which is the commonest place for one.
    opens = (
        [(i / 4, 0.30, 0.2) for i in range(1, 3)]
        + [(0.5 + i / 4, 0.001, 0.2) for i in range(1, 21)]
    )
    assert [one.state for one in _into_intervals(opens, 5.5)] == ["still"]

    assert SEEN_SECONDS == 1.0


def test_a_moment_inside_the_shot_can_be_put_on_the_beat():
    """Only the cut was ever aligned, so only the cut could be asked for.

    "The phone snaps shut on the downbeat and the shot runs on" is an
    ordinary thing to want and could not be said. Worse, a model choosing a
    length so an action completed at the cut had that alignment taken away
    again by the snap, which moves the out-point by up to a beat: the two
    goals were fighting through the same one control.
    """

    from montagewright.grounding import (
        BeatGrid, Cue, apply_to_edl, ground_timeline,
    )
    from montagewright.schema import EDL, Clip, MusicSync

    # A downbeat every two seconds.
    grid = BeatGrid(bpm=120.0, meter=4, duration_seconds=60.0, cues=tuple(
        [Cue(cue_id=f"d{i:02d}", time_seconds=i * 2.0, kind="downbeat")
         for i in range(30)]
        + [Cue(cue_id=f"b{i:02d}", time_seconds=i * 0.5, kind="beat")
           for i in range(120)]
    ))

    def film(anchor=None, relation="on", window=(0.0, 20.0), start=3.0):
        return ground_timeline(EDL(project_id="p", clips=[Clip(
            clip_id="k00", source_id="C1",
            approx_in_seconds=start, approx_out_seconds=start + 4.0,
            moments={"a01": 5.3},
            usable_from_seconds=window[0], usable_to_seconds=window[1],
            music_sync=MusicSync(
                cut_on_beat=False, anchor=anchor, anchor_relation=relation,
            ),
        )]), grid).clips[0]

    # Unanchored, the shot starts where it was told and the moment lands
    # wherever it happens to -- 2.3 seconds in, which is nothing in
    # particular.
    assert film().clip.approx_in_seconds == 3.0

    # Anchored, the in-point moves so the moment falls on a downbeat. The
    # length is untouched -- the rhythm pass decided that.
    anchored = film(anchor="a01")
    assert anchored.clip.approx_in_seconds != 3.0, "nothing moved"
    landing = 0.0 + (5.3 - anchored.clip.approx_in_seconds)
    assert abs(landing % 2.0) < 1e-6, landing
    assert abs(anchored.duration_seconds - 4.0) < 1e-6

    original = EDL(project_id="p", clips=[Clip(
        clip_id="k00", source_id="C1",
        approx_in_seconds=3.0, approx_out_seconds=7.0,
        moments={"a01": 5.3}, usable_from_seconds=0.0,
        usable_to_seconds=20.0,
        music_sync=MusicSync(cut_on_beat=False, anchor="a01"),
    )])
    grounded = ground_timeline(original, grid)
    applied = apply_to_edl(original, grounded)
    assert applied.clips[0].approx_in_seconds == grounded.clips[0].clip.approx_in_seconds

    short = EDL(project_id="p", clips=[Clip(
        clip_id="k00", source_id="C1",
        approx_in_seconds=0.0, approx_out_seconds=1.8,
        usable_from_seconds=0.0, usable_to_seconds=1.2,
        music_sync=MusicSync(cut_on_beat=True),
    )])
    feasible = ground_timeline(short, grid).clips[0]
    assert feasible.duration_seconds <= 1.2 + 1e-6
    assert feasible.landed_on is None
    assert "usable source ends" in (feasible.note or "")

    # Deliberately loose is a decision, and it is a beat either side.
    late = film(anchor="a01", relation="after")
    assert late.clip.approx_in_seconds < anchored.clip.approx_in_seconds

    # A moment this take does not have is said rather than ignored.
    missing = film(anchor="a09")
    assert "a09" in (missing.note or "")

    # And an anchor that would need the shot to start outside the stretch it
    # may be cut from is refused, not approximated -- an anchor half
    # honoured is a cut placed where nothing asked for it.
    boxed = film(anchor="a01", window=(5.2, 9.0), start=5.2)
    assert "outside" in (boxed.note or ""), boxed.note
    assert boxed.clip.approx_in_seconds == 5.2


def test_a_sample_is_the_same_clips_every_time_and_spread_across_the_shoot():
    """The point is not to pay for seventy-four cards to try one change.

    Random would defeat it: a fresh set every run is a fresh set of cards to
    buy. Taking the first N would defeat it differently -- rushes arrive in
    shooting order, so the front of the folder is all one setup.
    """

    import inspect

    from montagewright import cli

    source = inspect.getsource(cli.command_render)
    sampling = source[source.index("if args.sample and"):]
    sampling = sampling[: sampling.index("flush=True,") + 12]

    # Evenly spaced, so twelve clips are twelve different setups.
    assert "len(sources_paths) / args.sample" in sampling
    # Deterministic: no randomness anywhere near it.
    assert "random" not in sampling.lower()

    # The arithmetic it describes, on a stand-in folder.
    def pick(total, n):
        paths = list(range(total))
        step = total / n
        return [paths[min(total - 1, int(i * step))] for i in range(n)]

    assert pick(74, 12) == [0, 6, 12, 18, 24, 30, 37, 43, 49, 55, 61, 67]
    assert pick(74, 12) == pick(74, 12)
    # Never off the end, and never the same clip twice.
    for total, n in ((74, 12), (5, 5), (7, 3), (3, 10)):
        got = pick(total, min(n, total))
        assert max(got) < total and len(set(got)) == len(got), (total, n)


def test_timeline_coverage_rejects_speaker_tails_used_as_duration_filler():
    """Nine plausible pauses are still nine pauses, not ten seconds of film."""

    from montagewright.coverage import selection_coverage_audit
    from montagewright.planner import MaterialItem

    material = []
    shots = []
    audio = []
    for index in range(9):
        source_id = f"S{index:02d}"
        span_id = f"t{index:02d}"
        material.append(MaterialItem(
            source_id=source_id,
            duration_seconds=10.0,
            summary="answer",
            speech=(f"`{span_id}` 0.0-5.6s speaker: answer",),
        ))
        shots.append({
            "source_id": source_id,
            "seconds_needed": 60.0 / 9.0,
            "picture_role": "speaker",
            "audio_role": "discard",
        })
        audio.append({
            "audio_span_id": span_id,
            "starts_at_shot_index": index,
            "offset_seconds": 0.0,
        })

    audit = selection_coverage_audit(
        {"shots": shots, "audio_assignments": audio}, material, 60.0
    )
    assert audit.duration_seconds == pytest.approx(60.0)
    assert audit.supported_seconds == pytest.approx(53.1)
    assert any("not longer holds" in fault for fault in audit.faults)
    assert sum("natural lead/tail" in fault for fault in audit.faults) == 9


def test_timeline_coverage_is_generic_to_visual_and_audio_evidence():
    """Products, reactions and end cards use the same proof as interviews."""

    from montagewright.coverage import selection_coverage_audit
    from montagewright.planner import MaterialItem

    material = [MaterialItem(
        source_id="PRODUCT", duration_seconds=20.0, summary="folding action",
        action=("phone unfolds 0.0-8.0s",),
    )]
    action = selection_coverage_audit({
        "shots": [{
            "source_id": "PRODUCT", "seconds_needed": 8.0,
            "picture_role": "primary_action", "audio_role": "discard",
        }],
        "audio_assignments": [],
    }, material, 8.0)
    assert not action.faults
    assert action.supported_seconds == pytest.approx(8.0)

    padded_broll = selection_coverage_audit({
        "shots": [{
            "source_id": "PRODUCT", "seconds_needed": 8.0,
            "picture_role": "illustrative_broll", "audio_role": "discard",
        }],
        "audio_assignments": [],
    }, material, 8.0)
    assert any("B-roll" in fault for fault in padded_broll.faults)

    detail = selection_coverage_audit({
        "shots": [{
            "source_id": "PRODUCT", "seconds_needed": 3.0,
            "picture_role": "illustrative_broll", "audio_role": "discard",
        }],
        "audio_assignments": [],
    }, material, 3.0)
    assert not detail.faults

    intentional = selection_coverage_audit({
        "shots": [
            {
                "source_id": "PRODUCT", "seconds_needed": 1.2,
                "picture_role": "reaction", "audio_role": "discard",
            },
            {
                "source_id": "PRODUCT", "seconds_needed": 1.4,
                "picture_role": "end_hold", "audio_role": "discard",
            },
        ],
        "audio_assignments": [],
    }, material, 2.6)
    assert not intentional.faults


def test_bounded_visual_hold_repair_removes_only_unsupported_tail():
    """A model's harmless overshoot is local repair, not another paid call."""

    from montagewright.coverage import repair_bounded_visual_holds

    chosen = {"shots": [
        {
            "source_id": "ACTION", "seconds_needed": 8.0,
            "picture_role": "primary_action",
            "coverage_claim_seconds": 8.0,
        },
        {
            "source_id": "END", "seconds_needed": 2.0,
            "picture_role": "end_hold",
            "coverage_claim_seconds": 2.0,
        },
    ]}
    repairs = repair_bounded_visual_holds(chosen)

    assert chosen["shots"][0]["seconds_needed"] == 8.0
    assert chosen["shots"][0]["coverage_claim_seconds"] == 8.0
    assert chosen["shots"][1]["seconds_needed"] == 1.5
    assert "coverage_claim_seconds" not in chosen["shots"][1]
    assert repairs == (
        "k01: shortened end_hold from 2.00s to 1.50s; the removed tail "
        "had no additional content evidence",
    )


@pytest.mark.parametrize("protected", [
    {"audio_role": "sync_action", "audio_completion": "complete_action_sound"},
    {"audio_role": "ambient_texture"},
])
def test_bounded_visual_hold_repair_never_cuts_retained_audio(protected):
    from montagewright.coverage import repair_bounded_visual_holds

    shot = {
        "source_id": "SOUND", "seconds_needed": 2.0,
        "picture_role": "reaction", **protected,
    }
    assert repair_bounded_visual_holds({"shots": [shot]}) == ()
    assert shot["seconds_needed"] == 2.0


def test_bounded_visual_hold_repair_never_cuts_independent_narrative():
    from montagewright.coverage import repair_bounded_visual_holds

    shot = {
        "source_id": "PICTURE", "seconds_needed": 2.0,
        "picture_role": "end_hold", "audio_role": "discard",
    }
    chosen = {
        "shots": [shot],
        "audio_assignments": [{
            "audio_span_id": "t00", "starts_at_shot_index": 0,
            "offset_seconds": 0.0,
        }],
    }
    assert repair_bounded_visual_holds(chosen) == ()
    assert shot["seconds_needed"] == 2.0


def test_repaired_soft_boundary_is_accepted_without_padding_elsewhere():
    from montagewright.coverage import (
        repair_bounded_visual_holds,
        selection_coverage_audit,
    )
    from montagewright.planner import MaterialItem

    shots = [
        {
            "source_id": f"S{index:02d}", "seconds_needed": 3.0,
            "picture_role": "illustrative_broll", "audio_role": "discard",
        }
        for index in range(8)
    ]
    shots.append({
        "source_id": "MOVE", "seconds_needed": 4.0,
        "picture_role": "primary_action", "audio_role": "discard",
    })
    shots.append({
        "source_id": "END", "seconds_needed": 3.0,
        "picture_role": "end_hold", "audio_role": "discard",
    })
    chosen = {"shots": shots, "audio_assignments": []}
    repair_bounded_visual_holds(chosen)
    audit = selection_coverage_audit(
        chosen,
        [MaterialItem(
            source_id=shot["source_id"], duration_seconds=4.0, summary="",
            camera_moves=shot["source_id"] == "MOVE",
        ) for shot in shots],
        30.0,
    )
    # 29.5 is within the 30s delivery tolerance and needs no filler.
    assert audit.duration_seconds == pytest.approx(29.5)
    assert not audit.faults

    # A genuinely short structure stays visible in the accounting, but a
    # normal duration request is a soft creative target and does not force
    # unsupported filler.
    shots[-2]["seconds_needed"] = 3.0
    shots[-1]["seconds_needed"] = 3.0
    repair_bounded_visual_holds(chosen)
    audit = selection_coverage_audit(chosen, [MaterialItem(
        source_id=shot["source_id"], duration_seconds=4.0, summary="",
        camera_moves=shot["source_id"] == "MOVE",
    ) for shot in shots], 30.0)
    assert audit.duration_seconds == pytest.approx(28.5)
    assert not audit.faults

    hard = selection_coverage_audit(chosen, [MaterialItem(
        source_id=shot["source_id"], duration_seconds=4.0, summary="",
        camera_moves=shot["source_id"] == "MOVE",
    ) for shot in shots], 30.0, hard_target=True)
    assert any("short by 1.50s" in fault for fault in hard.faults)


def test_rhythm_cannot_stretch_a_proven_visual_window_to_fill_target():
    from montagewright.planner import _apply
    from montagewright.schema import Clip, EDL

    edl = EDL(project_id="visual-cap", clips=[Clip(
        clip_id="k00", source_id="BROLL",
        approx_in_seconds=1.0, approx_out_seconds=4.0,
        picture_role="illustrative_broll", audio_role="discard",
        coverage_claim_seconds=3.0,
    )])
    made = _apply(edl, {"k00": {
        "hold_seconds": 4.0, "cut_on_beat": False,
        "rhythm_reason": "pad to target",
    }})
    assert made.clips[0].approx_out_seconds == pytest.approx(4.0)


def test_resolved_edl_coverage_uses_the_same_contract_as_selection():
    from montagewright.coverage import edl_coverage_audit
    from montagewright.schema import AudioClip, Clip, EDL

    edl = EDL(
        project_id="coverage",
        clips=[Clip(
            clip_id="k00", source_id="VOICE",
            approx_in_seconds=0.0, approx_out_seconds=5.0,
            picture_role="speaker", audio_role="discard",
        )],
        audio_clips=[AudioClip(
            audio_id="a00", source_id="VOICE",
            in_seconds=0.0, out_seconds=3.0,
            starts_at_clip_id="k00", offset_seconds=0.0,
            role="narrative", completion="complete_thought",
        )],
    )
    audit = edl_coverage_audit(edl, 5.0)
    assert audit.supported_seconds == pytest.approx(3.3)
    assert any("natural lead/tail" in fault for fault in audit.faults)


def test_only_the_last_shot_may_claim_an_end_hold():
    from montagewright.coverage import selection_coverage_audit
    from montagewright.planner import MaterialItem

    material = [MaterialItem(source_id="A", duration_seconds=5, summary="")]
    audit = selection_coverage_audit({
        "shots": [
            {"source_id": "A", "seconds_needed": 1.0,
             "picture_role": "end_hold", "audio_role": "discard"},
            {"source_id": "A", "seconds_needed": 1.0,
             "picture_role": "primary_action", "audio_role": "discard"},
        ],
        "audio_assignments": [],
    }, material, 2.0)
    assert any("only valid on the final shot" in fault for fault in audit.faults)


def test_independent_sync_audio_is_timeline_coverage_too():
    """The generic audit is not synonymous with interview narration."""

    from montagewright.coverage import edl_coverage_audit
    from montagewright.schema import AudioClip, Clip, EDL

    edl = EDL(
        project_id="sync-action",
        clips=[Clip(
            clip_id="k00", source_id="PICTURE",
            approx_in_seconds=0.0, approx_out_seconds=2.0,
            picture_role="illustrative_broll", audio_role="discard",
        )],
        audio_clips=[AudioClip(
            audio_id="a00", source_id="SOUND",
            in_seconds=4.0, out_seconds=6.0,
            starts_at_clip_id="k00", offset_seconds=0.0,
            role="sync_action", completion="complete_action_sound",
        )],
    )
    audit = edl_coverage_audit(edl, 2.0)
    assert not audit.faults
    assert audit.supported_seconds == pytest.approx(2.0)


@pytest.mark.parametrize("role", ["primary_action", "music_montage"])
def test_a_role_name_cannot_launder_a_static_minute(role):
    from montagewright.coverage import selection_coverage_audit
    from montagewright.planner import MaterialItem

    audit = selection_coverage_audit({
        "shots": [{
            "source_id": "STATIC", "seconds_needed": 60.0,
            "picture_role": role, "audio_role": "discard",
        }],
        "audio_assignments": [],
    }, [MaterialItem(
        source_id="STATIC", duration_seconds=60.0, summary="still product"
    )], 60.0)
    assert audit.supported_seconds <= 4.0
    assert audit.faults


def test_an_audio_role_name_cannot_launder_a_static_minute():
    from montagewright.coverage import selection_coverage_audit
    from montagewright.planner import MaterialItem

    audit = selection_coverage_audit({
        "shots": [{
            "source_id": "STATIC", "seconds_needed": 60.0,
            "picture_role": "illustrative_broll",
            "audio_role": "ambient_texture",
        }],
        "audio_assignments": [],
    }, [MaterialItem(
        source_id="STATIC", duration_seconds=60.0, summary="still product"
    )], 60.0)
    assert audit.supported_seconds <= 6.0
    assert audit.faults


def test_coverage_uses_exact_audio_spans_not_rounded_prompt_copy():
    from montagewright.coverage import selection_coverage_audit
    from montagewright.planner import MaterialItem

    material = [MaterialItem(
        source_id="VOICE", duration_seconds=2.0, summary="answer",
        speech=("`t00` 0.0-1.0s speaker: rounded",),
        audio_spans=(("t00", 0.0, 1.049),),
    )]
    audit = selection_coverage_audit({
        "shots": [{
            "source_id": "VOICE", "seconds_needed": 1.349,
            "picture_role": "speaker", "audio_role": "discard",
        }],
        "audio_assignments": [{
            "audio_span_id": "t00", "starts_at_shot_index": 0,
            "offset_seconds": 0.0,
        }],
    }, material, 1.349)
    assert not audit.faults
    assert audit.supported_seconds == pytest.approx(1.349)


def test_coverage_rejects_large_over_delivery_too():
    from montagewright.coverage import selection_coverage_audit
    from montagewright.planner import MaterialItem

    audit = selection_coverage_audit({
        "shots": [{
            "source_id": "MOVE", "seconds_needed": 100.0,
            "picture_role": "primary_action", "audio_role": "discard",
        }],
        "audio_assignments": [],
    }, [MaterialItem(
        source_id="MOVE", duration_seconds=100.0, summary="continuous move",
        camera_moves=True,
    )], 60.0)
    assert any("over by 40.00s" in fault for fault in audit.faults)


def test_missing_picture_role_is_bounded_not_assumed_to_be_action():
    from montagewright.coverage import selection_coverage_audit
    from montagewright.planner import MaterialItem

    audit = selection_coverage_audit({
        "shots": [{
            "source_id": "LEGACY", "seconds_needed": 10.0,
            "audio_role": "discard",
        }],
        "audio_assignments": [],
    }, [MaterialItem(
        source_id="LEGACY", duration_seconds=10.0, summary="old cache"
    )], 10.0)
    assert audit.supported_seconds == pytest.approx(3.0)
    assert audit.faults


def test_reference_identity_track_requires_two_agreeing_semantic_anchors():
    from types import SimpleNamespace

    from montagewright.reframe import observations_from_sam

    samples = [
        SimpleNamespace(
            analysis_sample_time_ms=at,
            tracking_state="tracked",
            semantic_identity_status=(
                "seed_grounded" if index == 0 else "not_revalidated"
            ),
            derived_tracking_box=[100 + index * 10, 200, 400 + index * 10, 700],
        )
        for index, at in enumerate((0, 500, 1000))
    ]
    track = SimpleNamespace(samples=samples, analysis_fps=2.0)

    observations, states = observations_from_sam(
        track,
        clip_start_seconds=0.0,
        semantic_anchors=(
            (0.0, (0.1, 0.2, 0.4, 0.7)),
            (1.0, (0.12, 0.2, 0.42, 0.7)),
        ),
        require_identity_validation=True,
    )
    assert len(observations) == 3
    assert states["tracked"] == 3


def test_reference_identity_track_fails_closed_with_one_anchor():
    from types import SimpleNamespace

    from montagewright.reframe import observations_from_sam

    track = SimpleNamespace(
        analysis_fps=2.0,
        samples=[SimpleNamespace(
            analysis_sample_time_ms=0,
            tracking_state="tracked",
            semantic_identity_status="seed_grounded",
            derived_tracking_box=[100, 200, 400, 700],
        )],
    )
    observations, states = observations_from_sam(
        track,
        clip_start_seconds=0.0,
        semantic_anchors=((0.0, (0.1, 0.2, 0.4, 0.7)),),
        require_identity_validation=True,
    )
    assert observations == []
    assert states["identity_unverified"] == 1


def test_reference_identity_quorum_counts_each_physical_sample_once():
    from types import SimpleNamespace

    from montagewright.reframe import observations_from_sam

    track = SimpleNamespace(
        analysis_fps=2.0,
        samples=[SimpleNamespace(
            analysis_sample_time_ms=index * 500,
            tracking_state="tracked",
            semantic_identity_status="not_revalidated",
            derived_tracking_box=[100, 100, 400, 400],
        ) for index in range(10)],
    )
    observations, states = observations_from_sam(
        track,
        clip_start_seconds=0.0,
        semantic_anchors=(
            (0.0, (0.1, 0.1, 0.4, 0.4)),
            (0.5, (0.1, 0.1, 0.4, 0.4)),
        ),
        require_identity_validation=True,
    )

    assert len(observations) == 2
    assert states == {"tracked": 2, "identity_unbracketed": 8}
    assert states["tracked"] / sum(states.values()) == pytest.approx(0.2)


def test_selection_requires_every_lock_mandated_grounding_target():
    from montagewright.planner import grounding_target_disagreements

    shots = [{
        "looks": [
            {"entity_id": "target.primary"},
            {"entity_id": "none"},
        ],
    }]

    assert grounding_target_disagreements(
        shots, ["target.primary", "target.required_detail"]
    ) == [
        "selection omits required grounding entity_id "
        "'target.required_detail'"
    ]
    assert grounding_target_disagreements(
        shots, ["target.primary"]
    ) == []


def test_pipeline_reference_grounding_hands_two_exact_pts_to_geometry(monkeypatch, tmp_path):
    from types import SimpleNamespace

    from montagewright.executor import Source
    from montagewright.pipeline import Report, _reference_subject_samples
    from montagewright.planner import Usage
    from montagewright.reframe import Observation

    # The cut's own window is the candidate now: no second discovery is
    # bought on the master to re-answer what the material screen answered on
    # the proxy, and the interval is built locally from the in and out
    # points. So the lineage has to be real enough to build one.
    digest = "c" * 64
    lineage = SimpleNamespace(
        asset_id=f"sha256:{digest}",
        source_start_pts=0,
        source_time_base=SimpleNamespace(numerator=1, denominator=1000),
        content_sha256=digest,
        duration_ms=8_000,
    )

    monkeypatch.setattr(
        "montagewright.reference_grounding.inspect_video_lineage",
        lambda path: lineage,
    )

    def materialize(path, requested_time_ms, destination, max_width=None):
        return SimpleNamespace(lineage=SimpleNamespace(
            frame_pts=requested_time_ms,
            frame_time_ms=requested_time_ms,
            video_asset_id=f"sha256:{digest}",
            frame_sha256=(f"{requested_time_ms:064x}"[-64:]),
            width=1440,
            height=810,
        ))

    monkeypatch.setattr(
        "montagewright.reference_grounding.materialize_frame_at_time", materialize
    )

    evaluations = tuple(
        SimpleNamespace(
            lineage=SimpleNamespace(
                frame_pts=at,
                frame_time_ms=at,
                video_asset_id=f"sha256:{digest}",
                frame_sha256=f"{at:064x}"[-64:],
                width=1440,
                height=810,
            ),
            decision=SimpleNamespace(
                excluded_instances=(),
                tracking_box_xyxy_1000=(100, 200, 500, 800),
                identity_evidence=("same hinge",),
                confidence=(0.99 if at == 2_000 else 0.9),
            ),
        )
        for at in (1_600, 2_000, 3_400)
    )
    batch = SimpleNamespace(
        query_lock_sha256="lock",
        grounding_spec_sha256="spec",
        matched_anchor_count=3,
        sam_seed_evaluations=lambda: evaluations,
        model_dump=lambda mode: {},
    )
    monkeypatch.setattr(
        "montagewright.reference_grounding.decide_exact_frame_bboxes",
        lambda *args, **kwargs: (batch, Usage(12, 3, 1)),
    )

    spec = SimpleNamespace(
        definition_sha256=lambda: "5" * 64,
        identity_lock=SimpleNamespace(
            query_id="grounding:target.fold",
            definition_sha256=lambda: "1" * 64,
        ),
    )
    handed_to_sam = {}

    def tracked(*args, **kwargs):
        handed_to_sam.update(kwargs)
        return [
            Observation(0.6, 0.2, 0.3, 0.2, 0.4),
            Observation(1.0, 0.4, 0.5, 0.2, 0.4),
            Observation(2.4, 0.8, 0.7, 0.2, 0.4),
        ], {"tracked": 3}

    monkeypatch.setattr("montagewright.pipeline._track_subject", tracked)
    report = Report()
    boxes, times, anchors = _reference_subject_samples(
        Source("A", tmp_path / "A.mp4", 5.0, 1920, 1080),
        SimpleNamespace(
            clip_id="k00", approx_in_seconds=1.0, approx_out_seconds=4.0
        ),
        "target.fold",
        spec=spec,
        client=object(),
        upload_cache=None,
        report=report,
        work=tmp_path,
        output=None,
        discoveries={},
        checkpoint=tmp_path / "sam.pt",
    )

    assert len(boxes) == len(times) == len(anchors) == 3
    assert boxes[0]["centre_x"] == pytest.approx(0.2)
    assert boxes[0]["geometry_source"] == "sam2.1"
    assert anchors[0][1] == pytest.approx((0.1, 0.2, 0.5, 0.8))
    exact = handed_to_sam["seed_lineage"]
    assert exact.frame_pts == 2_000
    assert exact.frame_sha256 == f"{2_000:064x}"[-64:]
    assert (exact.width, exact.height) == (1440, 810)
    assert handed_to_sam["require_identity_validation"] is True
    assert report.reference_grounding["k00"]["status"] == "sam_geometry_validated"
    assert [usage.input_tokens for usage in report.usages] == [12], (
        "one paid call, the exact frames -- the second discovery this used to "
        "buy on the master asked what the material screen had already answered"
    )


def test_reference_critical_grounding_without_sam_fails_before_gemini(
    monkeypatch, tmp_path
):
    from types import SimpleNamespace

    from montagewright.executor import Source
    from montagewright.pipeline import Report, _reference_subject_samples

    monkeypatch.setattr(
        "montagewright.reference_grounding.discover_reference_candidates",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("missing local geometry must fail before Gemini")
        ),
    )
    report = Report()
    with pytest.raises(RuntimeError, match="requires SAM/local geometry"):
        _reference_subject_samples(
            Source("A", tmp_path / "A.mp4", 5.0, 1920, 1080),
            SimpleNamespace(
                clip_id="k00", approx_in_seconds=1.0, approx_out_seconds=4.0
            ),
            "target.fold",
            spec=SimpleNamespace(),
            client=object(),
            upload_cache=None,
            report=report,
            work=tmp_path,
            output=None,
            discoveries={},
            checkpoint=None,
        )
    assert report.reference_grounding["k00"]["status"] == (
        "local_geometry_unavailable"
    )


def test_look_presentation_intent_allows_partial_without_weakening_complete():
    from pydantic import ValidationError

    from montagewright.schema import Look

    partial = Look(
        at="the device entering from frame right",
        presentation_intent="partial_reveal",
        must_be_whole=False,
    )
    assert partial.presentation_intent == "partial_reveal"

    with pytest.raises(ValidationError, match="complete_hold"):
        Look(
            at="the entire readable display",
            presentation_intent="complete_hold",
            must_be_whole=False,
        )
    with pytest.raises(ValidationError, match="partial_reveal"):
        Look(
            at="the entire readable display",
            presentation_intent="partial_reveal",
            must_be_whole=True,
        )


def test_exact_frame_lineage_reaches_single_target_sam_seed_api(
    monkeypatch, tmp_path
):
    from types import SimpleNamespace

    from montagewright import pipeline
    from montagewright.executor import Source

    captured = {}

    def fake_track_bbox_sam21(**kwargs):
        captured.update(kwargs)
        return SimpleNamespace(samples=[], analysis_fps=4.0)

    monkeypatch.setattr(
        "montagewright.measure.sam_tracking.track_bbox_sam21",
        fake_track_bbox_sam21,
    )
    monkeypatch.setattr(
        pipeline,
        "observations_from_sam",
        lambda *args, **kwargs: ([], {"tracked": 0}),
    )
    lineage = SimpleNamespace(
        video_asset_id="sha256:" + "a" * 64,
        frame_pts=2_100,
        frame_time_ms=2_000,
        frame_sha256="b" * 64,
        width=1440,
        height=810,
    )

    pipeline._track_subject(
        Source("A", tmp_path / "A.mp4", 5.0, 1920, 1080),
        SimpleNamespace(
            clip_id="k00", approx_in_seconds=1.0, approx_out_seconds=4.0
        ),
        "the locked foldable instance",
        [100, 200, 500, 800],
        tmp_path / "sam.pt",
        tmp_path,
        seed_time_seconds=2.0,
        require_identity_validation=True,
        seed_lineage=lineage,
    )

    assert captured["asset_id"] == lineage.video_asset_id
    assert captured["seed_time_ms"] == lineage.frame_time_ms
    assert captured["seed_frame_pts"] == lineage.frame_pts
    assert captured["seed_frame_sha256"] == lineage.frame_sha256
    assert captured["seed_source_width"] == lineage.width
    assert captured["seed_source_height"] == lineage.height
    assert captured["seed_source"] == "reference_exact_frame_grounding"


def test_the_beat_may_not_hold_a_shot_past_what_it_can_show():
    """Two frames of unsupported picture threw away a whole film.

    Selection asked for 3.00s and 5.00s -- exactly the visual-only ceilings
    those roles prove -- the rhythm pass honoured them, and then grounding
    rounded both up to the next cue: 3.34s and 5.07s. The release gate
    refused the timeline, correctly, and eight good shots were never
    rendered. With no speech every shot's whole length is visual-only, so
    every shot sits at its ceiling and the next cue is always past it: the
    ordinary case for a music-only cut, not an edge.

    What an editor does is take the beat before. Short reads as intent;
    held-too-long reads as a mistake. The beat before, though -- reaching
    back for a distant cue would halve the shot, which is a different edit
    and one the rhythm pass never saw.
    """

    from montagewright.grounding import BeatGrid, Cue, ground_timeline
    from montagewright.schema import EDL, Clip, MusicSync

    def film(cues, claim, wanted, audio_role="discard"):
        grid = BeatGrid(
            bpm=120.0, meter=4, duration_seconds=60.0,
            cues=tuple(
                Cue(cue_id=name, time_seconds=at, kind="beat")
                for name, at in cues
            ),
        )
        return ground_timeline(EDL(project_id="p", clips=[Clip(
            clip_id="k00", source_id="C1",
            approx_in_seconds=0.0, approx_out_seconds=wanted,
            audio_role=audio_role,
            coverage_claim_seconds=claim,
            music_sync=MusicSync(cut_on_beat=True),
        )]), grid).clips[0]

    # The nearest cue to the 3.00s this shot proves sits at 3.34 -- the exact
    # shape of the failure. The beat at 2.60 is inside the evidence and
    # within a beat of it, so that is where the cut goes.
    over = [("b0", 0.0), ("b1", 2.60), ("b2", 3.34)]
    held = film(over, claim=3.0, wanted=3.0)
    assert held.duration_seconds <= 3.0 + 1e-6, "never past the evidence"
    assert abs(held.duration_seconds - 2.60) < 1e-6, "the beat before"
    assert held.landed_on == "b1", "still on the grid, and says which cue"
    assert "past what this shot can show" in (held.note or "")

    # The only earlier cue is a second and a half back. Taking it would be a
    # different edit, so the supported length is kept and this one cut is
    # simply not on the music.
    distant = [("b0", 0.0), ("b1", 1.50), ("b2", 3.34)]
    off = film(distant, claim=3.0, wanted=3.0)
    assert abs(off.duration_seconds - 3.0) < 1e-6
    assert off.landed_on is None
    assert "off the grid" in (off.note or "")

    # A shot carrying its own audio is not bounded by a visual-only ceiling:
    # how long it runs is a question about the sound, answered elsewhere.
    speaking = film(over, claim=3.0, wanted=3.0, audio_role="narrative")
    assert abs(speaking.duration_seconds - 3.34) < 1e-6, "unchanged behaviour"


def test_a_segment_offers_only_the_subjects_it_can_show():
    """A plan named a subject sighted two seconds after its window ended.

    The listing described subjects as properties of the clip, so everything
    the take contained anywhere in its length looked available from any
    segment of it. The local check could only refuse the plan afterwards --
    twice, and then the run ended. A subject the chosen seconds cannot show
    should never have been on the menu.
    """

    from montagewright.planner import MaterialItem, _describe_material
    from montagewright.spans import Span

    described = _describe_material([MaterialItem(
        source_id="C1",
        duration_seconds=10.0,
        summary="a table of devices",
        proxy=None,
        spans=(
            Span("C1:s00", "C1", 0.0, 4.0, "locked", "still on the right"),
            Span("C1:s01", "C1", 4.0, 10.0, "authored", "pans left"),
        ),
        sightings=(("左側的白色折疊手機", 6.0), ("右側的紫色折疊手機", 1.0)),
        subjects=("左側的白色折疊手機", "右側的紫色折疊手機"),
    )])

    first, second = described.split("；C1:s01")
    assert "右側的紫色折疊手機（1.0s）" in first, "sighted inside the window"
    assert "左側的白色折疊手機" not in first, (
        "a subject seen at 6.0s is not offered from a window ending at 4.0s"
    )
    assert "左側的白色折疊手機（6.0s）" in second
    assert "此段沒有測到可命名的主體" not in described


def test_a_panning_take_offers_a_subject_only_while_it_is_on_screen():
    """Inside the seconds is not the same as inside the picture.

    On a take whose own camera travels, a subject measured at 0:06 has left
    the frame by 0:08 -- and the listing that offered it said only that the
    segment contained it. Selection named it from the far end twice, the
    local check refused both times, and the run ended. The window where the
    coordinate still holds is computed with the same measurement the check
    refuses by.
    """

    from montagewright.motion import MotionInterval
    from montagewright.planner import MaterialItem, _describe_material
    from montagewright.spans import Span

    def listing(motion):
        return _describe_material([MaterialItem(
            source_id="C1",
            duration_seconds=10.0,
            summary="a row of devices",
            proxy=None,
            # A 16:9 source cropped to 9:16 keeps about a third of the width,
            # so a coordinate survives about a sixth of a width of travel.
            crop_width=0.32,
            spans=(Span("C1:s00", "C1", 0.0, 10.0, "authored", "pans left"),),
            sightings=(("左側的白色折疊手機", 6.0),),
            subjects=("左側的白色折疊手機",),
            motion=motion,
        )])

    still = listing((MotionInterval(
        event_id="m0", starts_seconds=0.0, ends_seconds=10.0, state="still",
        peak_vw_s=0.0, travel_vw=0.0, settles=False,
    ),))
    assert "左側的白色折疊手機（6.0s）" in still, (
        "a locked-off take keeps a coordinate for the whole span"
    )

    panning = listing((MotionInterval(
        event_id="m0", starts_seconds=0.0, ends_seconds=10.0, state="moving",
        peak_vw_s=0.2, travel_vw=1.0, settles=True,
    ),))
    assert "取用時窗要落在" in panning, "say when the coordinate still holds"
    window = panning.split("取用時窗要落在")[1].split("s")[0]
    opens, closes = (float(one) for one in window.split("–"))
    assert opens < 6.0 < closes
    assert closes - opens < 10.0, "a travelling frame does not hold it all"


def test_the_beat_never_cuts_an_authored_move_short():
    """A shot that lets the source's own move play is not a budget.

    Its length is a statement about that move. Pulling it back to the
    earlier beat -- the right answer when a shot is simply held too long --
    left an authored three-second pan 2.564s to finish in, and the release
    check refused the timeline, after this same function had been told the
    shot could show its full three seconds.
    """

    from montagewright.grounding import BeatGrid, Cue, ground_timeline
    from montagewright.schema import EDL, Clip, MusicSync, Reframe

    grid = BeatGrid(bpm=120.0, meter=4, duration_seconds=60.0, cues=(
        Cue(cue_id="b0", time_seconds=0.0, kind="beat"),
        Cue(cue_id="b1", time_seconds=2.56, kind="beat"),
        Cue(cue_id="b2", time_seconds=3.34, kind="beat"),
    ))

    def one(intent):
        return ground_timeline(EDL(project_id="p", clips=[Clip(
            clip_id="k00", source_id="C1",
            approx_in_seconds=0.0, approx_out_seconds=3.0,
            audio_role="discard", coverage_claim_seconds=3.0,
            reframe=Reframe(editorial_intent=intent, intent="let it play"),
            music_sync=MusicSync(cut_on_beat=True),
        )]), grid).clips[0]

    held = one("hold")
    assert abs(held.duration_seconds - 2.56) < 1e-6, (
        "an ordinary held shot takes the beat before"
    )

    authored = one("use_source_motion")
    assert authored.duration_seconds >= 3.0 - 1e-6, (
        "the source move runs to the end it was measured to need"
    )
    assert authored.landed_on is None, "and this cut is simply not on a beat"


def test_a_beat_may_lengthen_an_authored_move_but_never_shorten_it():
    """`nearest_cue` takes the closest event in either direction.

    So a cue thirteen milliseconds early won, an authored three-second pan
    was delivered in 2.987s, and the release check refused the timeline --
    over a hundredth of a second, twice in one evening on two shots. The
    move completing is the point of the shot; landing on the music is not.
    """

    from montagewright.grounding import BeatGrid, Cue, ground_timeline
    from montagewright.schema import EDL, Clip, MusicSync, Reframe

    def film(cues, intent):
        grid = BeatGrid(bpm=120.0, meter=4, duration_seconds=60.0, cues=tuple(
            Cue(cue_id=name, time_seconds=at, kind="beat") for name, at in cues
        ))
        return ground_timeline(EDL(project_id="p", clips=[Clip(
            clip_id="k00", source_id="C1",
            approx_in_seconds=0.0, approx_out_seconds=3.0,
            audio_role="discard",
            reframe=Reframe(editorial_intent=intent, intent="let it play"),
            music_sync=MusicSync(cut_on_beat=True),
        )]), grid).clips[0]

    # The nearest cue sits 13ms before the move finishes; the next one is a
    # third of a second after it.
    cues = [("b0", 0.0), ("b1", 2.987), ("b2", 3.33)]

    held = film(cues, "hold")
    assert abs(held.duration_seconds - 2.987) < 1e-6, (
        "an ordinary shot still takes the nearest cue"
    )

    authored = film(cues, "use_source_motion")
    assert authored.duration_seconds >= 3.0 - 1e-6
    assert authored.landed_on == "b2", "the first cue that lets it finish"
    assert "source move needs" in (authored.note or "")

    # Nothing within a beat after it: keep the move whole, lose the grid.
    lonely = film([("b0", 0.0), ("b1", 2.987)], "use_source_motion")
    assert abs(lonely.duration_seconds - 3.0) < 1e-6
    assert lonely.landed_on is None
    assert "left the grid" in (lonely.note or "")


def test_a_tracker_that_holds_nothing_is_one_shot_not_the_run():
    """Right instance, no local geometry: still a recoverable shot.

    Refusing to crop a reference-critical target on a model's box from a
    sampled frame is the substitution the whole reference path exists to
    refuse, and it stays refused. But a tracker that passed 0 of 12 frames
    ends one shot, and the layer holding the selection has an alternate for
    that commitment -- so the failure has to arrive as something it can
    catch, not as a bare RuntimeError from four frames deep.
    """

    from montagewright.pipeline import (
        ReferenceGeometryUnavailable,
        ReferenceIdentityUnconfirmed,
        ReferenceShotUnusable,
    )

    for kind in (ReferenceGeometryUnavailable, ReferenceIdentityUnconfirmed):
        fault = kind("k03", "device.fold", "passed only 0/12 frames")
        assert isinstance(fault, ReferenceShotUnusable), (
            "one catch covers both ways a shot fails its identity"
        )
        assert fault.clip_id == "k03" and fault.entity_id == "device.fold"
        assert "0/12" in str(fault)


def test_an_unverified_track_says_so_instead_of_counting_frames():
    """"0/12 frames passed" reads as a tracker that lost its subject twelve
    times. What happened is one verdict over the whole track: fewer than two
    exact-frame anchors agreed with the mask, so every sample was discarded
    at once. The two have nothing in common to fix, so they cannot share a
    sentence -- and the numbers that decided it were never written down.
    """

    from types import SimpleNamespace

    from montagewright.reframe import observations_from_sam

    def sample(at_ms, box):
        return SimpleNamespace(
            analysis_sample_time_ms=at_ms,
            derived_tracking_box=box,
            tracking_state="tracked",
            semantic_identity_status="",
        )

    track = SimpleNamespace(
        analysis_fps=4.0,
        samples=[
            sample(0, (100, 100, 200, 200)),
            sample(250, (100, 100, 200, 200)),
        ],
    )
    # Anchors sit where the tracker is, but on a different extent: a mask
    # tight on the screen against a box drawn around the whole handset.
    observations, states = observations_from_sam(
        track,
        clip_start_seconds=0.0,
        semantic_anchors=((0.0, (0.10, 0.10, 0.40, 0.40)),
                          (0.25, (0.10, 0.10, 0.40, 0.40))),
        require_identity_validation=True,
    )

    assert observations == []
    assert states["identity_unverified"] == 2
    assert states["_anchors_offered"] == 2
    assert states["_anchors_agreed"] == 0
    assert 0 < states["_best_agreement_pct"] < 35, (
        "record how close it came, or nobody can tell 34% from 3%"
    )


def test_every_undeliverable_shot_is_found_in_one_pass():
    """One at a time meant one repair per attempt and a re-render between.

    Three bounded attempts therefore covered three shots, and material shot
    at an event with four similar handsets on the tables has more than
    three. The work spent on the other shots is discarded either way, so
    finding all of them costs nothing extra and lets the selection repair
    them together.
    """

    import inspect

    from montagewright import pipeline

    source = inspect.getsource(pipeline.follow_subjects)
    assert "except ReferenceShotUnusable" in source, (
        "a shot that cannot be delivered must not end the pass"
    )
    assert "unusable_shots.append" in source and "continue" in source
    assert "raise ReferenceShotsUnusable(unusable_shots)" in source, (
        "and the whole list has to reach the layer that owns the selection"
    )

    faults = [
        pipeline.ReferenceIdentityUnconfirmed("k03", "device.fold", "k03: no"),
        pipeline.ReferenceGeometryUnavailable("k05", "device.fold", "k05: no"),
    ]
    batched = pipeline.ReferenceShotsUnusable(faults)
    assert [one.clip_id for one in batched.faults] == ["k03", "k05"]
    assert "k03: no" in str(batched) and "k05: no" in str(batched)


def test_an_exact_frame_verdict_is_remembered_by_the_frames_it_judged():
    """It was written down and never read back.

    Every repair attempt therefore re-paid for every shot it had already
    judged, including the shots it was not repairing -- three attempts over
    eight shots at three cents a call. What the verdict answers is a fact
    about specific decoded frames under one identity lock, and those frames
    are content-hashed on the way in, so the question has a name.
    """

    import inspect

    from montagewright import pipeline

    source = inspect.getsource(pipeline._reference_subject_samples)
    key = source.split("hashlib.sha256")[1].split(")).hexdigest()")[0]
    assert "spec.definition_sha256()" in key, "a different lock is a different question"
    assert "target_id" in key
    assert "frame_pts" in key and "frame_sha256" in key, (
        "name the frames that were actually judged, not the shot they came from"
    )
    assert "if batch is None:" in source, "a remembered verdict skips the call"
    # The call, not the import of its name, and not the earlier `_afford`
    # that belongs to discovery.
    assert source.index("remembered.exists()") < source.index(
        "decide_exact_frame_bboxes(\n"
    ), "look before paying"

    # And the path reaches it from the run, not from the output directory.
    assert "grounding_memory" in inspect.getsource(pipeline.follow_subjects)


def test_no_silent_exit_from_the_reference_stage():
    """Every way out of this stage ends as the same sentence to the caller.

    "could not be confirmed on two exact source frames" describes a
    judgement -- and three of the ways to reach it never made one: the cut
    fell outside the take, no client was there to ask, or fewer than two
    distinct frames could be decoded. An evening was spent believing a model
    was refusing shots it had never been shown.
    """

    import inspect

    from montagewright import pipeline

    source = inspect.getsource(pipeline._reference_subject_samples)
    exits = source.count("return [], [], ()")
    explained = source.count("report.subject_notes[clip.clip_id]")
    assert exits >= 4, "this test is about the early exits; find them"
    assert explained >= exits - 1, (
        f"{exits} ways out, {explained} of them say why -- a silent one "
        "becomes a sentence about a judgement nobody made"
    )


def test_a_replacement_that_cannot_be_built_is_asked_again_not_raised():
    """The film was already on disk; everything after it was not.

    A replan promised `complete_hold` without asking for the whole subject
    -- a contradiction the Look model refuses, but only when one is built,
    which happens long after the replan call returns. So it came back
    valid, was accepted, and raised a bare ValidationError out of the
    rebuild. Building them inside the retry loop turns that into the one
    thing this pass is already good at: asking again with the reason.
    """

    import inspect

    import pytest as _pytest

    from montagewright.planner import replan_shots
    from montagewright.schema import Look, reframe_of

    with _pytest.raises(Exception):
        Look(at="the wordmark", presentation_intent="complete_hold")

    # The rule is stated where the model reads it, not only where it is
    # enforced.
    fields = Look.model_fields
    assert "must_be_whole=true" in fields["presentation_intent"].description
    assert "complete_hold" in fields["must_be_whole"].description

    source = inspect.getsource(replan_shots)
    assert "reframe_of(shot)" in source, "build them while a retry is possible"
    assert source.index("reframe_of(shot)") < source.index(
        "replan violated candidate commitments twice"
    ), "and feed the failure into the same retry the commitments use"


def test_rhythm_is_given_a_whole_and_a_structure_not_an_average():
    """An average per shot is an anchor, and it was obeyed.

    Eight shots came back between 2.56 and 3.88 seconds with six of them
    inside 2.5-3.0 -- a metronome, not an edit. The prompt had handed the
    pass "29 seconds, 8 shots, 3.6 seconds each". An editor thinks in bars:
    three shots across these eight, then one long one. The grid could
    always answer how long a bar and a phrase run, and was never asked.
    """

    import inspect

    from montagewright.grounding import BeatGrid, Cue
    from montagewright.planner import _describe_music, decide_rhythm

    grid = BeatGrid(bpm=120.0, meter=4, duration_seconds=60.0, cues=(
        Cue(cue_id="d0", time_seconds=0.0, kind="downbeat"),
    ))
    said = _describe_music(grid)
    assert "One bar is 2.00s" in said
    assert "four-bar phrase is 8.00s" in said

    source = inspect.getsource(decide_rhythm)
    assert "平均每顆" not in source, "no per-shot average to anchor on"
    assert "這是總量，不是每顆的配額" in source
    # And no shape is prescribed here. The prompt file already asks for one
    # -- "每顆都差不多長，就是還沒有做這件事" -- in the right terms: a
    # result to reach, not a pattern to copy. Naming a pattern would make
    # every film the same pattern.
    for prescription in ("短切", "先短後長", "一顆長的"):
        assert prescription not in source


def test_the_floor_of_a_shot_has_one_name():
    """Three runs died over hundredths of a second, one path at a time.

    The beat snap, the content ceiling and the usable-window clamp each had
    to be taught separately that an authored move may not be cut short, and
    each lesson cost a whole film. A rule remembered in four places is a
    rule that will be forgotten in a fifth, so the shortest a shot may be
    is computed once and every path reads it.
    """

    import inspect

    from montagewright.grounding import ground_timeline

    source = inspect.getsource(ground_timeline)
    assert "floor_seconds = wanted if keeps_source_move else 0.0" in source
    # No path may re-derive it from the intent on its own.
    after = source.split(
        "floor_seconds = wanted if keeps_source_move else 0.0", 1
    )[1]
    assert "keeps_source_move else" not in after, (
        "every later path reads the floor rather than recomputing it"
    )
    assert after.count("floor_seconds") >= 4, (
        "the snap, the ceiling and the feasibility clamp all consult it"
    )


def test_a_span_is_offered_only_where_the_identity_was_seen():
    """Any overlap kept the whole span, and a cut may land anywhere in it.

    So a span that clipped a sighting by a fraction was offered entire,
    the cut landed in the part where the target is not, and the shot died
    four stages later with its frames judged and nothing in them. Keeping
    the sighted part is the same fix as everything else here: do not offer
    what cannot be delivered.
    """

    from dataclasses import dataclass

    from montagewright.cli import _screen_material_identity
    from montagewright.spans import Span

    @dataclass(frozen=True)
    class Item:
        source_id: str
        proxy: object
        spans: tuple

    class Seen:
        def __init__(self, spans):
            self.candidates = [
                type("C", (), {
                    "target_id": "device.fold",
                    "identity_status": "matched_target",
                    "start_ms": int(a * 1000), "end_ms": int(b * 1000),
                })()
                for a, b in spans
            ]
            self.target_summaries = []

    item = Item("C1", None, (
        Span("C1:s00", "C1", 0.0, 6.0),   # sighted 2.0-6.0 only
        Span("C1:s01", "C1", 6.0, 9.0),   # sighted 6.0-6.2: too little
        Span("C1:s02", "C1", 9.0, 12.0),  # never sighted
    ))
    # The screen only reads `proxy`; a source without one is passed through,
    # so drive the narrowing logic directly against a stub discovery.
    from montagewright import cli

    def remembered(*_args, **_kwargs):
        return Seen([(2.0, 6.2)]), None

    original = cli.__dict__.get("remembered_discovery")
    import montagewright.reference_grounding as grounding
    was = grounding.remembered_discovery
    grounding.remembered_discovery = remembered
    try:
        from pathlib import Path
        import tempfile

        with tempfile.TemporaryDirectory() as work:
            proxy = Path(work) / "C1.mp4"
            proxy.write_bytes(b"proxy")
            spec = type("S", (), {
                "identity_lock": type("L", (), {
                    "framing": type("F", (), {"required_target_ids": ["device.fold"]})(),
                    "identity": type("I", (), {"targets": []})(),
                })(),
            })()
            kept, aside = _screen_material_identity(
                [Item("C1", proxy, item.spans)], spec,
                client=object(), cache=None,
                ledger=type("L", (), {"check": lambda self: None,
                                      "spent_usd": 0.0})(),
                library=Path(work),
            )
    finally:
        grounding.remembered_discovery = was
        del original

    assert aside == {}
    spans = kept[0].spans
    assert [one.span_id for one in spans] == ["C1:s00"], (
        "the unsighted span and the sliver both go"
    )
    assert spans[0].starts_seconds == 2.0 and spans[0].ends_seconds == 6.0, (
        "and what is left is the part the identity was actually seen in"
    )
