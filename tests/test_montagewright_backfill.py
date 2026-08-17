"""The join between corrected words and measured time.

What is being protected here is one property: a caption's edges are moments
somebody actually spoke, because the recogniser measured them. Text may come
from anywhere -- a model, a person typing -- and the clock stays the
recogniser's. These check that it does.
"""

from montagewright.backfill import Timed, across_lines, align, drift, what_was_heard
from montagewright.transcript import Word


def said(*pairs: tuple[str, float, float]) -> list[Word]:
    return [Word(text=t, starts_seconds=a, ends_seconds=b) for t, a, b in pairs]


HEARD = said(
    ("在", 0.0, 0.2), ("夏", 0.2, 0.4), ("天", 0.4, 0.6),
    ("吹", 0.6, 0.8), ("頭", 0.8, 1.0), ("發", 1.0, 1.3),
)


def test_text_the_recogniser_got_right_keeps_the_time_it_was_measured_with():
    timed = align("在夏天吹頭發", HEARD)

    assert [one.starts_seconds for one in timed] == [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    assert all(one.measured for one in timed)


def test_a_corrected_character_takes_the_span_of_the_one_it_replaced():
    # 發 → 髮 is the correction the recogniser cannot make and the model can.
    timed = align("在夏天吹頭髮", HEARD)

    assert timed[-1].text == "髮"
    assert (timed[-1].starts_seconds, timed[-1].ends_seconds) == (1.0, 1.3)
    # Flagged as worked out rather than measured, even though the span is
    # exactly the replaced character's -- nobody was recorded saying 髮.
    assert not timed[-1].measured
    assert all(one.measured for one in timed[:-1])


def test_punctuation_the_correction_added_takes_no_time():
    timed = align("在夏天，吹頭髮。", HEARD)

    comma = timed[3]
    assert comma.text == "，"
    assert comma.starts_seconds == comma.ends_seconds
    # And it has not pushed the real characters off their measured times.
    assert timed[4].text == "吹" and timed[4].starts_seconds == 0.6


def test_a_line_never_drifts_from_what_was_measured():
    for text in ("在夏天吹頭發", "在夏天吹頭髮", "在夏天，吹頭髮。", "今夏天吹頭髮"):
        assert drift(align(text, HEARD), HEARD) == 0.0


def test_heard_is_read_from_the_recogniser_not_reported_by_the_model():
    # The model, asked to correct errors and quote them unchanged in the same
    # breath, corrects both. The recogniser's own output is on disk.
    assert what_was_heard(HEARD, 0.6, 1.3) == "吹頭發"


def test_lines_are_timed_without_reading_the_models_timestamps():
    starts_and_ends = across_lines(["在夏天", "吹頭髮"], HEARD)

    assert [(a, b) for a, b, _ in starts_and_ends] == [(0.0, 0.6), (0.6, 1.3)]


def test_editing_one_line_leaves_its_neighbours_where_they_were():
    lines = ["在夏天", "吹頭髮"]
    before = across_lines(lines, HEARD)

    after = across_lines(["在夏天", "吹頭髮啦"], HEARD)

    assert after[0][:2] == before[0][:2]


def test_splitting_a_line_puts_the_break_on_a_measured_moment():
    whole = across_lines(["在夏天吹頭髮"], HEARD)
    halves = across_lines(["在夏天", "吹頭髮"], HEARD)

    assert halves[0][0] == whole[0][0]
    assert halves[-1][1] == whole[0][1]
    # The new edge is a word boundary the recogniser measured, not a
    # proportional guess at where half the characters land.
    assert halves[0][1] == 0.6


def test_a_card_that_never_stored_words_gives_back_nothing_rather_than_a_guess():
    assert align("在夏天", []) == []
    assert across_lines(["在夏天", "吹頭髮"], []) == [(0.0, 0.0, []), (0.0, 0.0, [])]


def test_a_line_the_recogniser_missed_entirely_is_marked_not_measured():
    timed = align("完全沒說過的話", said(("嗯", 0.0, 0.1)))

    assert timed and not any(one.measured for one in timed)


def test_an_english_word_spreads_across_its_own_characters():
    timed = align("hello", said(("hello", 0.0, 0.5)))

    assert timed[0].starts_seconds == 0.0
    assert round(timed[-1].ends_seconds, 6) == 0.5
    assert timed[1].starts_seconds > timed[0].starts_seconds


def test_drift_of_nothing_is_nothing():
    assert drift([], HEARD) == 0.0
    assert drift([Timed("在", 0.0, 0.2)], []) == 0.0


def test_an_edit_saved_from_the_browser_comes_back_on_the_measured_clock(
    tmp_path,
) -> None:
    """The round trip, not the mechanism.

    The browser sends the times a line had before it was edited. If the
    server keeps them, a line whose length changed sits at a moment nobody
    said it. This checks the edited text is re-timed and that the new times
    are sent back, because the browser cannot draw what it is not told.
    """

    import json

    import montagewright.webapp as web
    from fastapi.testclient import TestClient

    was, held = web.RUNS_ROOT, web._transcript_map
    try:
        web.RUNS_ROOT = tmp_path / "runs"
        here = web.RUNS_ROOT / "r1"
        (here / "out" / "work").mkdir(parents=True)
        (here / "run.json").write_text(
            json.dumps({"state": "done", "started_at": 0.0}), encoding="utf-8"
        )

        # A run whose words were measured, and whose shot is the whole of it.
        (here / "out" / "report.json").write_text(json.dumps({
            "selection": {"shots": [{"source_id": "a", "start_seconds": 0.0}]},
            "rhythm": {"k00": {"seconds": 1.3}},
            "direction": {"aspect": "9:16"},
        }), encoding="utf-8")

        def cards(run):
            return {"a": {
                "lines": [{
                    "text": "在夏天吹頭發",
                    "starts_seconds": 0.0, "ends_seconds": 1.3,
                }],
                "words": [
                    {
                        "text": one.text,
                        "starts_seconds": one.starts_seconds,
                        "ends_seconds": one.ends_seconds,
                    }
                    for one in HEARD
                ],
                "silences": [],
            }}

        web._transcript_map = cards
        client = TestClient(web.create_app())

        stale = [
            here / "out" / "deliverable-subtitled.mp4",
            here / "out" / "deliverable-graphics.mp4",
            here / "out" / "deliverable-graphics-subtitled.mp4",
            here / "out" / "work" / "graphics-render" / "layout.json",
        ]
        for path in stale:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"old subtitle authority")

        # Somebody splits the line in two, keeping the old times on both.
        saved = client.put("/api/runs/r1/subtitle-track", json={"lines": [
            {"at": 0.0, "until": 1.3, "text": "在夏天"},
            {"at": 0.0, "until": 1.3, "text": "吹頭髮"},
        ]})

        assert saved.status_code == 200
        back = saved.json()["timed"]
        assert [(one["at"], one["until"]) for one in back] == [
            (0.0, 0.6), (0.6, 1.3)
        ]
        assert [one["timing_source"] for one in back] == [
            "apple_audio_time_range", "apple_audio_time_range",
        ]
        assert all(one["timing_confidence"] == "unverified" for one in back)
        assert all(not one["timing_locked"] for one in back)
        assert all(not path.exists() for path in stale)

        # A timing edit is a different authority from a text correction.
        # Once a person locks the clock, a later save must not silently move
        # it back to Apple's measured word ranges.
        locked = client.put("/api/runs/r1/subtitle-track", json={"lines": [{
            "at": 0.125,
            "until": 0.875,
            "text": "在夏天吹頭髮",
            "timing_source": "manual",
            "timing_confidence": "human_locked",
            "timing_locked": True,
        }]})
        assert locked.status_code == 200
        assert locked.json()["timed"] == [{
            "at": 0.125,
            "until": 0.875,
            "text": "在夏天吹頭髮",
            "timing_source": "manual",
            "timing_confidence": "human_locked",
            "timing_locked": True,
        }]
        loaded = client.get("/api/runs/r1/subtitle-track")
        assert loaded.status_code == 200
        assert loaded.json()["lines"][0]["timing_locked"] is True
        assert loaded.json()["lines"][0]["timing_source"] == "manual"
    finally:
        web.RUNS_ROOT = was
        web._transcript_map = held
        web.RUNS.pop("r1", None)


# --- Gemini reads video in MM:SS; the card asks for seconds ---------------

def _card(**over):
    base = {
        "segments": [{"from": "0:00", "to": "0:10", "status": "eligible",
                      "why": ""}],
        "action": [], "subjects": [],
    }
    base.update(over)
    return base


def test_a_clock_reading_has_only_one_meaning():
    """The collision this repair existed for is now unwritable.

    A take lasting 113.4s came back saying it was usable to `1.53`, which is
    1:53 with the colon turned into a decimal point -- the dangerous shape,
    because 1.53 is inside the clip and so passes every range check while
    claiming two minutes of take is good for a second and a half. Another,
    71.1s long, said `110.0`, which is 1:10 with the colon dropped. Both
    clips over a minute in the library, both wrong, in opposite directions.

    Segments are asked for in the notation the model already reads video in,
    so neither is a thing that can be written. There is nothing to guess
    between.
    """

    from montagewright.spans import seconds_of

    assert seconds_of("1:53") == 113.0
    assert seconds_of("1:10") == 70.0
    assert seconds_of("0:00") == 0.0
    assert seconds_of("12:04") == 724.0

    # Padded or not is the same reading, and neither is worth refusing an
    # answer over.
    assert seconds_of("01:07") == 67.0
    assert seconds_of("0:3") == 3.0

    # Past ten minutes, and past an hour, which an interview shot on a locked
    # off camera reaches without ever being scene-split into pieces. Minutes
    # that keep counting and an hours field are the same moment and both get
    # written, so both are read.
    assert seconds_of("12:04") == 724.0
    assert seconds_of("72:15") == seconds_of("1:12:15") == 4335.0
    assert seconds_of("2:00:00") == 7200.0

    # Sixty in a place that only goes to fifty-nine means the colon is not
    # where it looks like it is.
    assert seconds_of("1:60") is None
    assert seconds_of("1:60:00") is None
    assert seconds_of("1:2:3:4") is None

    # A bare number below a minute is still read, because a model told six
    # times to write a clock will occasionally write one anyway, and there is
    # no colon that could have been lost from it.
    assert seconds_of("3") == 3.0
    assert seconds_of("67") is None   # two digits past 59 is not a clock
    assert seconds_of("110") is None  # 1:10 with the colon dropped

    # And the other half of that collision, which the first pass at this let
    # through: a decimal point with no colon beside it is either a second and
    # a half or 1:53 with the colon turned into a point. Both readings are
    # plausible, which is the ambiguity the notation exists to remove, so it
    # is refused rather than guessed at.
    assert seconds_of("1.53") is None
    assert seconds_of("3.5") is None

    # A real number is a resolved one -- this reads its own output back, and
    # by then there is nothing left to settle.
    assert seconds_of(1.53) == 1.53
    assert seconds_of(113.0) == 113.0

    assert seconds_of("") is None
    assert seconds_of("about a minute") is None


def test_both_ends_of_an_action_are_read_the_same_way():
    """1.1 and 1.13 used to be a thirty-millisecond action or 1:10 to 1:13.

    Which of those it was had to be guessed, and the guess had to be made for
    both ends together or a span could come back with its start read one way
    and its end the other. Asked as clock readings there is nothing to guess:
    the pair means what it says.
    """

    from montagewright.clipcard import times_on_receipt

    got = times_on_receipt(
        _card(action=[{"what": "x", "from": "1:10", "to": "1:13"}]), 113.4
    )

    assert [(a["from"], a["to"]) for a in got["action"]] == [(70.0, 73.0)]


def test_a_clip_that_never_had_the_problem_is_left_alone():
    from montagewright.clipcard import times_on_receipt

    was = _card(
        segments=[{"from": "0:00", "to": "0:27", "status": "eligible", "why": ""}],
        action=[{"what": "x", "from": "0:02", "to": "0:04"}],
        subjects=[{"label": "x", "seen_at": "0:02"}],
    )

    got = times_on_receipt(dict(was), 27.2)

    assert (got["segments"][0]["from"], got["segments"][0]["to"]) == (0.0, 27.0)
    assert [(a["from"], a["to"]) for a in got["action"]] == [(2.0, 4.0)]
    assert got["subjects"][0]["seen_at"] == 2.0


def test_a_genuinely_short_window_is_the_models_to_report():
    # Five usable seconds out of a hundred is a strong claim, and it is the
    # card's to make. Nothing here second-guesses a reading any more, which
    # is the point of asking in a notation with one meaning.
    from montagewright.clipcard import times_on_receipt

    got = times_on_receipt(
        _card(segments=[{"from": "0:00", "to": "0:05", "status": "eligible",
                         "why": "只有開頭這幾秒對到焦"}]),
        100.0,
    )

    assert got["segments"][0]["to"] == 5.0


def test_an_action_that_cannot_be_read_into_the_clip_is_dropped():
    # Not clamped. A missing action is a static shot, which is a fine thing
    # to be; an action at a wrong second puts a cut in the wrong place.
    from montagewright.clipcard import times_on_receipt

    got = times_on_receipt(
        _card(action=[{"what": "x", "from": "6:40", "to": "8:00"}]), 30.0
    )

    assert got["action"] == []


def test_a_clip_whose_length_is_unknown_is_not_second_guessed():
    from montagewright.clipcard import times_on_receipt

    was = _card(action=[{"what": "x", "from": "0:02", "to": "0:04"}])

    assert times_on_receipt(dict(was), 0.0) == was


def test_a_timestamp_rounded_up_past_the_end_is_kept_not_deleted():
    """Gemini samples at one frame a second, so it answers in whole seconds.

    On a clip lasting 12.012s the last frame it holds is at 12, and "ends at
    13" is that rounding rather than a mistake. Half a second of tolerance,
    which is what this had first, deleted the action instead.
    """

    from montagewright.clipcard import times_on_receipt

    got = times_on_receipt(
        _card(action=[{"what": "x", "from": "0:10", "to": "0:13"}]), 12.012
    )

    assert [(a["from"], a["to"]) for a in got["action"]] == [(10.0, 12.012)]


def test_the_slop_only_absorbs_the_rounding_it_was_sized_for():
    """It was sized against a notation error that can no longer be made.

    The smallest possible MM:SS collision was 1:01 read as 101 on a clip just
    past a minute -- overshooting by forty seconds -- while a frame-a-second
    clock rounds by one. Now that every time on the card is a clock reading,
    the only thing left for it to absorb is that rounding.
    """

    from montagewright.clipcard import SLOP, times_on_receipt

    assert 1.0 <= SLOP < 40

    # A second past the end is the sampler, and is kept, trimmed to the file.
    inside = times_on_receipt(
        _card(action=[{"what": "x", "from": "0:10", "to": "0:13"}]), 12.4
    )
    assert [a["to"] for a in inside["action"]] == [12.4]

    # Well past it is not, and is dropped rather than clamped.
    outside = times_on_receipt(
        _card(action=[{"what": "x", "from": "0:50", "to": "0:55"}]), 12.4
    )
    assert outside["action"] == []


# --- the proxy is a smaller copy, never a larger one ----------------------

def _clip(tmp_path, width, height, name="in.mp4"):
    import subprocess

    made = tmp_path / name
    subprocess.run(
        ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-f", "lavfi",
         "-i", f"testsrc2=size={width}x{height}:rate=30:duration=1",
         "-c:v", "libx264", "-crf", "28", "-pix_fmt", "yuv420p", str(made)],
        check=True,
    )
    return made


def _size(path):
    import subprocess

    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream=width,height", "-of", "csv=p=0", str(path)],
        capture_output=True, text=True,
    ).stdout.strip()
    return tuple(int(one) for one in out.split(","))


def test_a_clip_narrower_than_the_proxy_width_is_left_alone(tmp_path):
    """`scale=640` is a demand, not a limit.

    Handed a 320x240 clip it produced a 640x480 one -- bigger than the file
    it came from, blurrier than the picture it describes, and no more use to
    a model that caps the frame at 70 tokens regardless.
    """

    from montagewright.cli import _encode_proxy

    small = _clip(tmp_path, 320, 240)
    made = tmp_path / "out.mp4"
    _encode_proxy(small, made)

    assert _size(made) == (320, 240)


def test_a_clip_wider_than_the_proxy_width_is_shrunk_to_it(tmp_path):
    from montagewright.cli import _encode_proxy

    big = _clip(tmp_path, 1920, 1080)
    made = tmp_path / "out.mp4"
    _encode_proxy(big, made)

    assert _size(made) == (640, 360)


def test_the_proxy_keeps_the_length_it_was_made_from(tmp_path):
    """Forcing a frame rate made the duration land on a multiple of it.

    Cards describe the proxy and the edit cuts the original, so the two ran
    a tenth of a second apart until the rate was left alone.
    """

    from montagewright.cli import _encode_proxy
    from montagewright.clipcard import clip_seconds

    source = _clip(tmp_path, 1920, 1080)
    made = tmp_path / "out.mp4"
    _encode_proxy(source, made)

    assert abs(clip_seconds(made) - clip_seconds(source)) < 0.005


def test_the_resolution_key_is_the_one_the_api_reads():
    """The knob has to be connected to something.

    Every video part carried `media_resolution`, which the Interactions API
    does not define -- its field is `resolution`. The SDK dropped it on the
    floor and forwarded the unknown key, so five call sites looked like they
    were controlling frame detail and were not. It went unnoticed because
    `low` and the default are both 70 tokens a frame for video, so the
    setting that never applied would not have changed anything if it had.

    Checked against the SDK rather than a string, so the day the field is
    renamed this fails here instead of silently in a paid call.
    """

    import re
    from pathlib import Path

    from google.genai._gaos.types.interactions.videocontent import VideoContent

    root = Path(__file__).resolve().parents[1] / "src" / "montagewright"
    sending = [
        path for path in root.rglob("*.py")
        if re.search(r'"type":\s*"video"', path.read_text(encoding="utf-8"))
    ]
    assert sending, "no video parts found; this test has lost its subject"

    for path in sending:
        text = path.read_text(encoding="utf-8")
        assert '"media_resolution"' not in text, (
            f"{path.name} sets media_resolution, which the Interactions API "
            f"ignores; the field is `resolution`"
        )

    # And the key that is used actually lands on the model's own field.
    part = VideoContent(
        type="video", mime_type="video/mp4", uri="files/x", resolution="low"
    )
    assert part.resolution == "low"


# --- thinking is spent from the answer's budget --------------------------

def test_a_pass_that_spent_its_budget_thinking_says_so():
    """An exhausted budget produces no text at all.

    Not truncated JSON -- nothing. The old message was "returned no text",
    which reads as the model declining to answer rather than as a ceiling
    to raise. The API marks the run `incomplete`; that is worth reading
    instead of guessing from the shape of the output.
    """

    import pytest

    from montagewright.planner import PlannerError, _parse

    class Ran:
        status = "incomplete"
        output_text = ""
        usage = {"total_thought_tokens": 45, "total_output_tokens": 0}

    with pytest.raises(PlannerError) as raised:
        _parse(Ran(), what="selection")

    assert "output budget" in str(raised.value)
    assert "45" in str(raised.value)


def test_a_finished_pass_is_parsed_normally():
    from montagewright.planner import _parse

    class Ran:
        status = "completed"
        output_text = '{"shots": []}'
        usage = {}

    assert _parse(Ran(), what="selection") == {"shots": []}


def test_the_ceiling_is_the_models_own():
    # 65536 for gemini-3.7-flash. Half of it was still a ration, and the
    # billing is on what is produced rather than on what is allowed.
    from montagewright.planner import MAX_OUTPUT_TOKENS

    assert MAX_OUTPUT_TOKENS == 65536


# --- a description belongs beside its own footage ------------------------

def _material(tmp_path, ids, missing=()):
    from montagewright.planner import MaterialItem

    out = []
    for one in ids:
        proxy = tmp_path / f"{one}.mp4"
        if one not in missing:
            proxy.write_bytes(b"not really a video")
        out.append(
            MaterialItem(
                source_id=one, duration_seconds=10.0, summary=f"{one} 的內容",
                proxy=proxy, composition="horizontal",
            )
        )
    return out


class _Cache:
    def uri_for(self, path, client, *, mime_type):
        return f"files/{path.stem}", None


def test_each_clip_is_described_next_to_its_own_video(tmp_path):
    from montagewright.planner import _attach_material

    parts = _attach_material(_material(tmp_path, ["a", "b", "c"]), _Cache(), None)

    # text, video, text, video, text, video -- and each text names the clip
    # whose uri follows it.
    assert [one["type"] for one in parts] == ["text", "video"] * 3
    for said, shown in zip(parts[::2], parts[1::2]):
        assert shown["uri"].split("/")[-1] in said["text"]


def test_a_missing_proxy_takes_its_description_with_it(tmp_path):
    """The failure this shape exists to prevent.

    With the listing in the prompt and the videos after it, a clip that
    failed to encode was skipped among the videos while its line stayed in
    the listing -- so every clip after it was described against the wrong
    picture, and nothing raised.
    """

    from montagewright.planner import _attach_material

    parts = _attach_material(
        _material(tmp_path, ["a", "b", "c"], missing={"b"}), _Cache(), None
    )

    assert [one["type"] for one in parts] == ["text", "video"] * 2
    assert "b" not in "".join(
        one["text"] for one in parts if one["type"] == "text"
    ).replace("的內容", "")
    for said, shown in zip(parts[::2], parts[1::2]):
        assert shown["uri"].split("/")[-1] in said["text"]


def test_the_prompt_no_longer_carries_a_second_copy_of_the_listing():
    # Described twice is worse than described once in the wrong place: the
    # two copies can disagree, and only one of them sits by the footage.
    from pathlib import Path

    text = (
        Path(__file__).resolve().parents[1]
        / "src" / "montagewright" / "planner.py"
    ).read_text(encoding="utf-8")

    assert "依序附上影片" not in text


# --- a move needs somewhere to happen ------------------------------------

def test_travel_room_comes_from_resolution_not_only_from_shape():
    """The first version of this asked only about aspect and got it wrong.

    It assumed the crop is always the largest that fits, so a 4K take
    delivering 9:16 was reported as having no vertical room at all. The crop
    only has to be as tall as the delivery -- 1920 of the 2160 it has -- and
    the 240 left over are room to tilt.
    """

    from montagewright.reframe import travel_room

    def room(w, h, ow, oh):
        return travel_room(
            source_width=w, source_height=h, target_aspect=ow / oh,
            output_width=ow, output_height=oh,
        )

    across, up = room(3840, 2160, 1080, 1920)
    assert round(across, 3) == 0.719
    assert round(up, 3) == 0.111

    # A source with nothing spare really does have nowhere to go vertically.
    across, up = room(1920, 1080, 1080, 1920)
    assert round(across, 3) == 0.684
    assert up == 0.0

    # And delivering the aspect the source already is only has no room when
    # the resolution matches too -- 4K to FHD can put the frame anywhere.
    assert room(3840, 2160, 1920, 1080) == (0.5, 0.5)
    assert room(1920, 1080, 1920, 1080) == (0.0, 0.0)


def test_the_planner_is_told_which_moves_have_nowhere_to_go():
    from montagewright.planner import MaterialItem, _describe_material

    said = _describe_material([
        MaterialItem(
            source_id="a", duration_seconds=10.0, summary="x",
            push_room=1.5, pan_room=0.684, tilt_room=0.0,
        )
    ])

    assert "橫向可移 68%" in said
    assert "縱向沒有空間" in said
    # And what the room costs in seconds, which is the half that was missing:
    # a distance nobody can price is a move nobody asks for.
    assert "走完全程" in said and "medium 1.0s" in said


def test_a_source_already_at_the_delivery_aspect_says_neither_move_works():
    from montagewright.planner import MaterialItem, _describe_material

    said = _describe_material([
        MaterialItem(
            source_id="a", duration_seconds=10.0, summary="x",
            push_room=1.0, pan_room=0.0, tilt_room=0.0,
        )
    ])

    # Named by what the frame cannot do, not by a move name: the planner no
    # longer chooses from a menu, so `pan` and `tilt` are words it can neither
    # write nor be told.
    assert "橫向縱向都沒有空間" in said


# --- a push is the one move whose frame shrinks --------------------------

def _zoom(**over):
    from montagewright.reframe import build_zoom_path, zoom_budget

    budget = zoom_budget(
        source_width=3840, source_height=2160, source_aspect=16 / 9,
        target_aspect=1080 / 1920, output_width=1080, output_height=1920,
    )
    args = dict(
        source_aspect=16 / 9, target_aspect=1080 / 1920, duration_seconds=3.0,
        direction="push_in", centre_x=0.5, centre_y=0.5, energy="active",
        framing="fill", budget=budget, subject_height=0.35, clip_id="k00",
    )
    args.update(over)
    return build_zoom_path(**args)


def _where(crop, subject_x):
    """Where the subject sits inside the crop: 0 is the left edge, 1 the right."""

    return (subject_x - crop.x) / crop.width


def test_a_push_follows_a_subject_that_moves_while_the_frame_closes():
    """Aiming at the mean of five samples loses a walking subject.

    Measured before this: a subject crossing from 0.35 to 0.65 starts hard
    against the left edge and finishes at 1.22 -- outside the frame -- and
    nothing recorded that it had happened.
    """

    walk = [(3.0 * i / 4, 0.35 + 0.30 * i / 4, 0.5) for i in range(5)]
    degradations = []

    path = _zoom(track=walk, degradations=degradations)

    assert round(_where(path.keyframes[0].crop, 0.35), 2) == 0.5
    assert round(_where(path.keyframes[-1].crop, 0.65), 2) == 0.5
    assert any(
        one.ladder_other == "zoom_followed_subject" for one in degradations
    )


def test_a_push_given_no_track_aims_where_it_was_told_to():
    """Passing no track has to change nothing about a static push.

    This asserted "two keyframes" when it was written, which was an
    implementation detail rather than the property -- a later change gave
    every designed move a rest at each end, and the test failed for a
    reason that had nothing to do with what it was guarding.
    """

    without = _zoom()
    explicit_none = _zoom(track=None)

    assert [
        (one.seconds, one.crop.x, one.crop.width) for one in without.keyframes
    ] == [
        (one.seconds, one.crop.x, one.crop.width)
        for one in explicit_none.keyframes
    ]
    # Aimed at the centre it was given, not at the middle of the frame.
    assert without.keyframes[0].crop.x == _zoom(centre_x=0.5).keyframes[0].crop.x


def test_a_subject_that_only_jitters_is_not_chased():
    # A track that wanders by less than the deadband would make the push
    # wobble, which reads worse than aiming at one point.
    still = [(3.0 * i / 4, 0.5 + 0.001 * i, 0.5) for i in range(5)]

    path = _zoom(track=still)
    centres = [one.crop.x + one.crop.width / 2.0 for one in path.keyframes]
    assert max(centres) - min(centres) < 1e-6
    assert path.keyframes[0].crop == path.keyframes[1].crop
    assert path.keyframes[-2].crop == path.keyframes[-1].crop


# --- what a crop cannot measure ------------------------------------------

def test_the_card_asks_for_shot_size_and_facing():
    """Neither is derivable from geometry, and both decide what can follow
    what: two neighbouring shots at the same size read as a jump, and two
    facing the same way read as both people addressing the same side."""

    from montagewright.clipcard import card_schema

    schema = card_schema()

    assert "shot_size" in schema["required"]
    assert "facing" in schema["required"]
    assert schema["properties"]["shot_size"]["enum"] == [
        "wide", "medium", "close", "extreme_close"
    ]
    assert schema["properties"]["facing"]["enum"] == [
        "left", "right", "toward", "away", "flat"
    ]


def test_the_planner_is_shown_size_and_facing():
    from montagewright.planner import MaterialItem, _describe_material

    said = _describe_material([
        MaterialItem(
            source_id="a", duration_seconds=10.0, summary="x",
            shot_size="close", facing="right",
        )
    ])

    assert "景別close" in said and "朝向right" in said


def test_a_shot_with_no_direction_says_nothing_about_direction():
    # `flat` is "no direction to preserve", which is not a fact worth a line
    # in a listing the planner has to read seventy-four of.
    from montagewright.planner import MaterialItem, _describe_material

    said = _describe_material([
        MaterialItem(
            source_id="a", duration_seconds=10.0, summary="x",
            shot_size="wide", facing="flat",
        )
    ])

    assert "朝向" not in said


def test_selection_is_told_what_to_do_with_them():
    # A field nobody is told to use is a field nobody fills honestly.
    from pathlib import Path

    prompt = (
        Path(__file__).resolve().parents[1] / "src" / "montagewright"
        / "prompts" / "selection_zh-TW.txt"
    ).read_text(encoding="utf-8")

    assert "景別太接近會跳" in prompt
    assert "銀幕方向" in prompt or "朝向決定" in prompt


def test_the_overlay_reports_the_move_that_happened():
    """The crop overlay exists to check whether the move happened.

    It printed `camera_move`, which is what was asked for -- so a pan that
    degraded to a hold drew a motionless box captioned 橫搖, in the one view
    whose whole job is catching exactly that.
    """

    from pathlib import Path

    page = (
        Path(__file__).resolve().parents[1] / "src" / "montagewright"
        / "web" / "index.html"
    ).read_text(encoding="utf-8")

    assert "function cropDid(" in page
    # The label is built from what the keyframes did, and only mentions the
    # plan when the two disagree.
    assert "const did = cropDid(keys);" in page
    assert "if (planned !== did)" in page


def test_selection_is_told_pan_cannot_make_a_wide_subject_whole():
    """A pan may scan a row, but cannot contain the whole row in one frame.

    The prompt advised `pan` for a subject too wide to frame without saying
    it needs both ends named, so a row of watches came back as one subject,
    could not be followed, and rendered as a hold.
    """

    from pathlib import Path

    prompt = (
        Path(__file__).resolve().parents[1] / "src" / "montagewright"
        / "prompts" / "selection_zh-TW.txt"
    ).read_text(encoding="utf-8")

    # Preserve the distinction between sequentially reading a wide subject
    # and fulfilling simultaneous whole-frame containment.
    assert "只能露出" in prompt
    assert "依序展示局部" in prompt
    assert "同一畫面完整入鏡" in prompt


# --- a move has to arrive somewhere and stay there -----------------------

def _handoff(seconds, energy="calm", degradations=None):
    from montagewright.reframe import build_handoff_path

    return build_handoff_path(
        source_aspect=16 / 9, target_aspect=1080 / 1920,
        duration_seconds=seconds, from_centre=0.15, to_centre=0.85,
        from_width=0.10, to_width=0.10, energy=energy,
        clip_id="k00", degradations=degradations,
    )


def _centre(keyframe):
    return keyframe.crop.x + keyframe.crop.width / 2


def test_a_designed_move_rests_at_both_ends():
    """Two keyframes means moving in every frame of the shot.

    That is a pan with its first and last seconds cut off, and it reads as
    one: the eye never gets a still frame to recognise where it started or
    where it ended up.
    """

    path = _handoff(5.0, energy="active")

    assert len(path.keyframes) == 4
    # Still at the start, still at the end, travelling in between.
    assert _centre(path.keyframes[0]) == _centre(path.keyframes[1])
    assert _centre(path.keyframes[2]) == _centre(path.keyframes[3])
    assert path.keyframes[1].seconds > 0.0
    assert path.keyframes[2].seconds < 5.0


def test_a_move_that_cannot_cross_in_the_time_says_so():
    """It used to stop partway and report nothing.

    A 1.2s calm pan across 0.700 of frame arrived 43% of the way, and the
    destination -- usually the point of the shot -- never appeared.
    """

    degradations = []
    _handoff(1.2, energy="calm", degradations=degradations)

    assert [one.ladder_other for one in degradations] == [
        "move_does_not_fit_the_time",
        "camera_endpoint_not_reached_before_cut",
    ]
    assert degradations[-1].adjudication == "replan"
    measured = degradations[0].measured
    assert measured["needed_speed_vw_s"] > measured["max_speed_vw_s"]


def test_a_move_with_room_to_spare_reports_nothing():
    degradations = []
    _handoff(5.0, energy="active", degradations=degradations)

    assert degradations == []


def test_a_shot_too_short_to_rest_in_still_moves():
    # Settling is capped at a share of the shot, so a brief take is not all
    # settling and no move.
    path = _handoff(0.6, energy="active")

    assert _centre(path.keyframes[-1]) > _centre(path.keyframes[0])


def test_a_push_rests_too():
    # Same argument: a push that starts on the first frame and ends on the
    # last reads as cut out of a longer one.
    from montagewright.reframe import build_zoom_path, zoom_budget

    budget = zoom_budget(
        source_width=3840, source_height=2160, source_aspect=16 / 9,
        target_aspect=1080 / 1920, output_width=1080, output_height=1920,
    )
    path = build_zoom_path(
        source_aspect=16 / 9, target_aspect=1080 / 1920, duration_seconds=4.0,
        direction="push_in", centre_x=0.5, centre_y=0.5, energy="active",
        framing="fill", budget=budget, subject_height=0.35,
    )

    assert len(path.keyframes) == 4
    assert path.keyframes[0].crop.width == path.keyframes[1].crop.width
    assert path.keyframes[2].crop.width == path.keyframes[3].crop.width


# --- one builder for every shape a list of looks can be -----------------

WIDE, TIGHT = 0.3164, 0.20


def _looks(stops, seconds=4.0, degradations=None, energy="active", native_speed=0.0, native_settles_at=None):
    from montagewright.reframe import build_look_path

    return build_look_path(
        stops, source_aspect=16 / 9, target_aspect=1080 / 1920,
        duration_seconds=seconds, energy=energy, clip_id="k00",
        degradations=degradations, native_speed=native_speed,
        native_settles_at=native_settles_at,
    )


def _mids(path):
    return [round(k.crop.x + k.crop.width / 2, 2) for k in path.keyframes]


def test_one_look_is_a_hold():
    path = _looks([(0.0, 0.5, 0.5, WIDE)])

    assert path.is_static


def test_two_looks_are_a_move_that_rests_at_both_ends():
    path = _looks([(0.0, 0.2, 0.5, WIDE), (0.0, 0.8, 0.5, WIDE)])

    assert _mids(path) == [0.2, 0.2, 0.8, 0.8]


def test_two_looks_at_one_thing_are_a_push():
    path = _looks([(0.0, 0.5, 0.5, WIDE), (0.0, 0.5, 0.5, TIGHT)])

    widths = [round(k.crop.width, 2) for k in path.keyframes]
    assert widths == [0.32, 0.32, 0.20, 0.20]
    assert _mids(path) == [0.5, 0.5, 0.5, 0.5]


def test_three_looks_stop_on_the_way():
    """The shape the old menu could not express at all.

    `then_subject` named exactly two endpoints, so a row of three watches
    had no way to be introduced one at a time.
    """

    path = _looks(
        [(0.6, 0.15, 0.5, WIDE), (0.6, 0.5, 0.5, WIDE), (0.8, 0.85, 0.5, WIDE)],
        seconds=5.0,
    )

    assert len(path.keyframes) == 6
    assert _mids(path) == [0.16, 0.16, 0.5, 0.5, 0.84, 0.84]


def test_a_move_and_a_push_at_once():
    # Needed a builder of its own before; now it is just two looks that
    # disagree about both position and size.
    path = _looks([(0.0, 0.2, 0.5, WIDE), (0.0, 0.8, 0.5, TIGHT)])

    assert _mids(path)[0] != _mids(path)[-1]
    assert path.keyframes[0].crop.width > path.keyframes[-1].crop.width


def test_too_many_looks_for_the_time_is_reported():
    degradations = []
    _looks(
        [(0.6, 0.15, 0.5, WIDE), (0.6, 0.5, 0.5, WIDE), (0.8, 0.85, 0.5, WIDE)],
        seconds=2.0, degradations=degradations,
    )

    ladders = [one.ladder_other for one in degradations]
    assert "looks_do_not_fit_the_time" in ladders
    # Asking to rest 2.0s in a 2.0s shot and also cross the frame changes the
    # plan's own dwell, which is said separately: one note is the executor
    # reporting a speed it cannot reach, the other is the planner's number
    # being altered.
    assert "declared_dwell_shortened_to_fit" in ladders
    measured = next(
        one.measured for one in degradations
        if one.ladder_other == "looks_do_not_fit_the_time"
    )
    assert measured["needed_speed_vw_s"] > measured["max_speed_vw_s"]


def test_resting_never_eats_the_whole_shot():
    # Three looks asking for a second each, in a shot lasting two.
    path = _looks(
        [(1.0, 0.15, 0.5, WIDE), (1.0, 0.5, 0.5, WIDE), (1.0, 0.85, 0.5, WIDE)],
        seconds=2.0,
    )

    assert _mids(path)[0] < _mids(path)[-1]
    assert path.keyframes[-1].seconds <= 2.0


def test_the_move_is_read_off_the_looks_not_taken_from_the_plan():
    """Removing camera_move from the schema made every shot a hold.

    reframe_of still read the field, the new schema no longer sends it, and
    the default was "hold" -- so the whole cut would have rendered
    motionless with nothing raising. The name is now observed rather than
    chosen, which is also why it cannot disagree with the looks.
    """

    from montagewright.schema import reframe_of

    def move(looks):
        return reframe_of({"looks": looks, "why": "x"}).camera_move

    assert move([{"at": "the coin"}]) == "hold"
    assert move([{"at": "the left one"}, {"at": "the right one"}]) == "pan"
    assert move(
        [{"at": "the coin", "framing": "thirds"},
         {"at": "the coin", "framing": "fill"}]
    ) == "push_in"
    assert move(
        [{"at": "the table", "framing": "fill"},
         {"at": "the table", "framing": "thirds"}]
    ) == "pull_out"
    assert move([{"at": "a"}, {"at": "b"}, {"at": "c"}]) == "pan"


def test_a_plan_written_before_looks_still_says_what_it_meant():
    from montagewright.schema import reframe_of

    was = reframe_of({
        "subject": "the left handset", "camera_move": "pan",
        "then_subject": "the right handset", "why": "x",
    })

    assert was.camera_move == "pan"
    assert [one.at for one in was.looks] == [
        "the left handset", "the right handset"
    ]


def _measured(monkeypatch, looks, places, reference_samples=None):
    """Run _measure_looks against fake grounding, counting the calls."""

    from montagewright import pipeline
    from montagewright.executor import Source

    asked = []

    def located(frames, description, *, client):
        asked.append(description)

        class Used:
            input_tokens = output_tokens = thought_tokens = 0

        return [{
            "present": True, "centre_x": places[description], "centre_y": 0.5,
            "width": 0.1, "height": 0.3, "frame_index": 0,
        }], Used()

    monkeypatch.setattr(pipeline, "locate_subject", located)
    monkeypatch.setattr(pipeline, "_sample_frames", lambda *a, **k: ([], []))
    monkeypatch.setattr(pipeline, "_may_ask", lambda client: True)

    class Clip:
        clip_id = "k00"
        approx_in_seconds, approx_out_seconds = 0.0, 6.0

    stops, missing, tracks = pipeline._measure_looks(
        looks,
        Source(source_id="s", path=None, duration_seconds=6.0,
               width=3840, height=2160),
        Clip(), None, pipeline.Report(), object(), 1080 / 1920,
        reference_samples=reference_samples,
    )
    return stops, missing, asked


def _identity_reference(centre_x: float):
    """One confirmed carrier box, the shape reference grounding hands over."""

    return (
        [{
            "present": True, "centre_x": centre_x, "centre_y": 0.5,
            "width": 0.4, "height": 0.6, "frame_index": 0,
        }],
        [0.0],
        (),
    )


def test_three_looks_reach_the_renderer_as_three_stops(monkeypatch):
    """They used to be truncated to two with nothing recorded.

    `pan` read `subject` and `then_subject`; a third look had nowhere to go,
    so a row of three watches lost its middle stop silently.
    """

    from montagewright.schema import Look

    stops, missing, _ = _measured(
        monkeypatch,
        [Look(at="A", seconds=0.6), Look(at="B", seconds=0.6),
         Look(at="C", seconds=0.8)],
        {"A": 0.15, "B": 0.5, "C": 0.85},
    )

    assert missing == ""
    assert [round(one[1], 2) for one in stops] == [0.15, 0.5, 0.85]
    assert [one[0] for one in stops] == [0.6, 0.6, 0.8]


def test_one_subject_looked_at_twice_is_measured_once(monkeypatch):
    # Two looks at one thing is how a push is written, and grounding the
    # same description twice would buy one answer twice.
    from montagewright.schema import Look

    stops, _, asked = _measured(
        monkeypatch,
        [Look(at="A", framing="thirds"), Look(at="A", framing="fill")],
        {"A": 0.4},
    )

    assert asked == ["A"]
    assert len(stops) == 2
    # Same place, tighter crop -- which is what a push in is.
    assert stops[0][1] == stops[1][1]
    assert stops[1][3] < stops[0][3]


def test_a_push_on_a_locked_target_still_costs_no_grounding(monkeypatch):
    """The economy that makes a grounded push cheap must survive.

    Both looks name the same thing and carry the same locked identity, so the
    confirmed reference already answers where it is. Nothing here may buy a
    description grounding, and both stops must land on the carrier.
    """

    from montagewright.schema import Look

    stops, missing, asked = _measured(
        monkeypatch,
        [
            Look(at="the phone", entity_id="device.primary", framing="thirds"),
            Look(at="the phone", entity_id="device.primary", framing="fill"),
        ],
        {},
        reference_samples={"device.primary": _identity_reference(0.42)},
    )

    assert missing == ""
    assert asked == []
    assert [round(one[1], 3) for one in stops] == [0.42, 0.42]
    assert stops[1][3] < stops[0][3]


def test_two_instances_of_one_locked_target_are_measured_apart(monkeypatch):
    """A compare between siblings is not a push, and must not collapse.

    Selection may ask to compare two phones that are both the locked model.
    Keying the measurement on the target alone answered the first look and
    reused it for the second, so the two stops landed on one box, the frame
    travelled nowhere, and a comparison the material fully supported was
    reported as `compare compiled to a static crop`. The identity lock says
    which model; the look's own words say which one of them.
    """

    from montagewright.schema import Look

    stops, missing, asked = _measured(
        monkeypatch,
        [
            Look(at="the left phone", entity_id="device.primary"),
            Look(at="the middle phone", entity_id="device.primary"),
        ],
        {"the left phone": 0.2, "the middle phone": 0.75},
        reference_samples={"device.primary": _identity_reference(0.2)},
    )

    assert missing == ""
    assert asked == ["the left phone", "the middle phone"]
    assert [round(one[1], 2) for one in stops] == [0.2, 0.75]


def test_multi_look_uses_sam_track_at_the_gemini_seed_time(monkeypatch, tmp_path):
    """A push used to bypass SAM, and its seed time was hard-coded."""

    from montagewright import pipeline
    from montagewright.executor import Source
    from montagewright.reframe import Observation
    from montagewright.schema import Look

    frames = [tmp_path / f"{index}.jpg" for index in range(5)]
    times = [10.5, 11.5, 12.5, 13.5, 14.5]
    monkeypatch.setattr(
        pipeline, "_sample_frames", lambda *a, **k: (frames, times)
    )
    monkeypatch.setattr(pipeline, "_may_ask", lambda client: True)

    class Used:
        input_tokens = output_tokens = thought_tokens = 0

    def located(frames, description, *, client):
        return [
            {"present": False, "frame_index": 0},
            {
                "present": True, "frame_index": 1,
                "centre_x": 0.4, "centre_y": 0.5,
                "width": 0.1, "height": 0.3,
            },
        ], Used()

    monkeypatch.setattr(pipeline, "locate_subject", located)
    asked = {}

    def tracked(*args, **kwargs):
        asked.update(kwargs)
        return [
            Observation(0.0, 0.2, 0.5, 0.1, 0.3),
            Observation(5.0, 0.8, 0.5, 0.1, 0.3),
        ], {"tracked": 2}

    monkeypatch.setattr(pipeline, "_track_subject", tracked)

    class Clip:
        clip_id = "k11"
        approx_in_seconds, approx_out_seconds = 10.0, 15.0

    _, _, tracks = pipeline._measure_looks(
        [Look(at="coin", framing="thirds"),
         Look(at="coin", framing="fill")],
        Source(source_id="s", path=tmp_path / "source.mp4",
               duration_seconds=20.0, width=3840, height=2160),
        Clip(), tmp_path, pipeline.Report(), object(), 1080 / 1920,
        tmp_path / "sam.pt",
    )

    assert asked["seed_time_seconds"] == 11.5
    assert tracks[0][0][:2] == (0.0, 0.2)
    assert tracks[0][-1][:2] == (5.0, 0.8)
    assert [(at, x) for at, x, _ in tracks[1]] == [
        (at, x) for at, x, _ in tracks[0]
    ]


def test_sam_checkpoint_is_discovered_by_default(tmp_path):
    from montagewright.cli import SAM_CHECKPOINT_NAME, _default_sam_checkpoint

    checkpoint = tmp_path / "artifacts" / "models" / SAM_CHECKPOINT_NAME
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"checkpoint")

    assert _default_sam_checkpoint((tmp_path,)) == checkpoint.resolve()


def test_sam_can_be_explicitly_disabled():
    from types import SimpleNamespace
    from montagewright.cli import _sam_checkpoint_for

    assert _sam_checkpoint_for(SimpleNamespace(
        sam_checkpoint=None, no_sam_tracking=True,
    )) is None


def test_a_subject_nobody_can_find_is_named_rather_than_guessed(monkeypatch):
    from montagewright import pipeline
    from montagewright.executor import Source
    from montagewright.schema import Look

    def nothing(frames, description, *, client):
        class Used:
            input_tokens = output_tokens = thought_tokens = 0

        return [{"present": False}], Used()

    monkeypatch.setattr(pipeline, "locate_subject", nothing)
    monkeypatch.setattr(pipeline, "_sample_frames", lambda *a, **k: ([], []))
    monkeypatch.setattr(pipeline, "_may_ask", lambda client: True)

    class Clip:
        clip_id = "k00"
        approx_in_seconds, approx_out_seconds = 0.0, 6.0

    stops, missing, tracks = pipeline._measure_looks(
        [Look(at="the ghost")],
        Source(source_id="s", path=None, duration_seconds=6.0,
               width=3840, height=2160),
        Clip(), None, pipeline.Report(), object(), 1080 / 1920,
    )

    assert stops == [] and missing == "the ghost"


def test_only_one_place_knows_what_shape_a_shot_is_written_in():
    """Six readers survived the field being removed from the schema.

    Two raised -- one of them after direction and selection had been paid
    for -- and four returned an empty string or "hold" and carried on. The
    lesson had been written an hour earlier and not applied: when a field
    goes, search for who still reads it, not who still writes it.

    So there is one reader, and this is what keeps it that way.
    """

    import re
    from pathlib import Path

    root = Path(__file__).resolve().parents[1] / "src" / "montagewright"
    reading = re.compile(
        r'\[["\'](?:subject|camera_move|then_subject|must_be_whole)["\']\]'
        r'|\.get\(\s*["\'](?:subject|camera_move|then_subject|must_be_whole)["\']'
    )
    offenders = []
    for path in sorted(root.rglob("*.py")):
        if path.name == "schema.py":
            continue  # the one place allowed to know both shapes
        for number, line in enumerate(
            path.read_text(encoding="utf-8").splitlines(), start=1
        ):
            if reading.search(line):
                offenders.append(f"{path.name}:{number}  {line.strip()}")

    assert not offenders, (
        "these read a selection shot's old fields directly instead of going "
        "through schema.looks_of / subject_of / move_of_shot:\n  "
        + "\n  ".join(offenders)
    )


def test_a_three_look_plan_actually_reaches_the_look_builder(monkeypatch):
    """The wiring, not the parts.

    build_look_path had seven tests and _measure_looks three, and every one
    passed while the branch that calls them was unreachable: the
    two-subject pan sat above it, and `then_subject` is set exactly when
    there is a second look, so it caught every pan first. Three watches went
    on losing their middle stop.

    Testing the pieces is not testing that anything calls them.
    """

    from montagewright import pipeline
    from montagewright.executor import Source
    from montagewright.schema import Clip, EDL, reframe_of

    called = {}

    def look_path(stops, **kw):
        from montagewright.reframe import CropBox, CropPath, Keyframe

        called["stops"] = stops
        return CropPath([Keyframe(0.0, CropBox(0.0, 0.0, 0.3, 1.0))])

    def located(frames, description, *, client):
        class Used:
            input_tokens = output_tokens = thought_tokens = 0

        return [{
            "present": True, "centre_x": {"A": 0.2, "B": 0.5, "C": 0.8}[description],
            "centre_y": 0.5, "width": 0.1, "height": 0.3, "frame_index": 0,
        }], Used()

    monkeypatch.setattr(pipeline, "build_declared_look_path", look_path)
    monkeypatch.setattr(pipeline, "locate_subject", located)
    monkeypatch.setattr(pipeline, "_sample_frames", lambda *a, **k: ([], []))
    monkeypatch.setattr(pipeline, "_may_ask", lambda client: True)

    # Built the way production builds it. Constructing a Reframe by hand
    # gave one with no `then_subject`, and `then_subject` is precisely what
    # the shadowing branch tested for -- so the test passed with the bug
    # reintroduced. A fake that does not have the shape of the real thing
    # cannot catch a bug about that shape.
    clip = Clip(
        clip_id="k00", source_id="s", approx_in_seconds=0.0,
        approx_out_seconds=5.0,
        reframe=reframe_of({
            "looks": [{"at": "A"}, {"at": "B"}, {"at": "C"}], "why": "x",
        }),
    )
    pipeline.follow_subjects(
        EDL(project_id="t", clips=[clip]),
        {"s": Source(source_id="s", path=None, duration_seconds=5.0,
                     width=3840, height=2160)},
        target_aspect=1080 / 1920,
        report=pipeline.Report(),
        client=object(),
    )

    assert "stops" in called, (
        "a three-look plan never reached build_look_path -- something above "
        "the looks branch is catching it first"
    )
    assert len(called["stops"]) == 3


def test_a_command_line_run_that_crashed_is_not_listed_as_done(tmp_path):
    """The note beside the output is written before the work starts.

    So its presence says the run began. A run that died partway was picked
    up as "done" with no shots, no spend and no film -- indistinguishable
    from one that worked.
    """

    import json

    import montagewright.webapp as web

    was = web.RUNS_ROOT
    try:
        web.RUNS_ROOT = tmp_path / "runs"
        for name, finished in (("worked", True), ("crashed", False)):
            here = web.RUNS_ROOT / name / "out"
            here.mkdir(parents=True)
            (here / "command.json").write_text("{}", encoding="utf-8")
            if finished:
                (here / "report.json").write_text(
                    json.dumps({"shots": []}), encoding="utf-8"
                )
        web.RUNS.clear()
        web.recall()

        assert web.RUNS["worked"].state == "done"
        assert web.RUNS["crashed"].state == "interrupted"
    finally:
        web.RUNS_ROOT = was
        web.RUNS.clear()


def test_a_run_without_a_report_can_still_be_opened():
    """The state most worth inspecting was the one that would not open.

    showWorkspace(true) was called only from render(), and render() only
    runs when the API returns a report -- so a crashed run left the cold
    page covering the workspace, and clicking its card did nothing. The log,
    the fault, the resume button and whatever film it managed to render all
    live inside `.work`, hidden.
    """

    from pathlib import Path

    page = (
        Path(__file__).resolve().parents[1] / "src" / "montagewright"
        / "web" / "index.html"
    ).read_text(encoding="utf-8")

    opening = page[page.index("async function openRun("):]
    opening = opening[: opening.index("\n}\n")]
    shows = opening.index("showWorkspace(true)")
    guarded = opening.index("if (data.report)")

    assert shows < guarded, (
        "the workspace is only revealed once a report exists, so a run that "
        "crashed cannot be opened"
    )


def test_the_report_survives_a_crash_in_the_review_loop(tmp_path, monkeypatch):
    """It was written once, at the very end.

    So a crash anywhere after the render discarded everything already
    decided and paid for. One run left a finished film, a review round and
    three replans on disk with no report -- which the interface reads as a
    run that produced nothing but a video.
    """

    import json
    import re
    from pathlib import Path

    source = (
        Path(__file__).resolve().parents[1] / "src" / "montagewright" / "cli.py"
    ).read_text(encoding="utf-8")

    # The review loop is inside a try whose handlers set `stopped` and let
    # _write_report run; the crash is re-raised only after delivery.
    loop = source.index("if args.review:")
    guard = source.rindex("try:", 0, loop)
    writes = source.index("_write_report(", loop)
    handler = source.index("except Exception as error:", guard)
    reraise = source.index("raise crashed", writes)

    assert guard < loop < handler < writes < reraise, (
        "the report has to be written between catching the failure and "
        "giving up on the run"
    )
    # And the failure is named rather than swallowed.
    assert "traceback.print_exc()" in source[handler:writes]
    assert re.search(r"stopped = f\"\{type\(error\)\.__name__\}", source)


def test_the_providers_own_cap_is_this_project_s_budget(monkeypatch):
    """A 429 for money is not a 429 for pace.

    Hitting the provider's spending cap arrived as a raw traceback out of
    the SDK. BudgetSpent already means exactly this and already has a path:
    stop, keep what exists, do not degrade to continue. Whose cap it was
    does not change what to do about it.
    """

    import pytest

    from montagewright.cost import BudgetSpent
    from montagewright.planner import _is_spend_cap, _provider_budget_message, ask

    class Capped(Exception):
        code = 429

        def __str__(self):
            return (
                "429 RESOURCE_EXHAUSTED. Your project has exceeded its "
                "monthly spending cap."
            )

    class TooFast(Exception):
        code = 429

        def __str__(self):
            return "429 RESOURCE_EXHAUSTED. Quota exceeded for requests"

    assert _is_spend_cap(Capped())
    assert not _is_spend_cap(TooFast())
    assert "monthly spending cap" in (_provider_budget_message(Capped()) or "")
    assert _provider_budget_message(TooFast()) is None

    class PrepayEmpty(Exception):
        code = 429

        def __str__(self):
            return (
                "429 RESOURCE_EXHAUSTED. Your prepayment credits are depleted. "
                "Please manage your project and billing."
            )

    prepay = _provider_budget_message(PrepayEmpty()) or ""
    assert "Prepay credits are depleted" in prepay
    assert "Provider detail" in prepay

    class Client:
        def __init__(self, error):
            self.error = error

        @property
        def interactions(self):
            raise self.error

    with pytest.raises(BudgetSpent):
        ask(Client(Capped()))
    # Pace is the SDK's to retry, so it keeps its own type.
    with pytest.raises(TooFast):
        ask(Client(TooFast()))


def test_renderer_retries_a_busy_videotoolbox_encoder_in_software(monkeypatch):
    """Encoder presence is not proof that macOS can open a session now."""

    from types import SimpleNamespace

    from montagewright.renderer import _run

    calls = []

    def fake_run(command, **_):
        calls.append(command)
        if len(calls) == 1:
            return SimpleNamespace(
                returncode=187,
                stderr=(
                    "Cannot create compression session: -12903\n"
                    "The hardware encoder may be busy, or not supported."
                ),
                stdout="",
            )
        return SimpleNamespace(returncode=0, stderr="", stdout="")

    monkeypatch.setattr("montagewright.renderer.subprocess.run", fake_run)
    result = _run([
        "ffmpeg", "-i", "in.mp4", "-c:v", "h264_videotoolbox", "out.mp4"
    ])

    assert result.returncode == 0
    assert calls[0][4] == "h264_videotoolbox"
    assert calls[1][4] == "libx264"


# --- how long a shot needs is a fact about that shot --------------------

def test_the_floor_comes_from_the_shot_not_from_the_move_name():
    """`min_seconds` said every pan needs 2.5 seconds.

    One number for every pan on every clip ever, while the real floor for
    the same word runs from about a second to over five: it forbade a short
    pan across a narrow gap and permitted a long one that could not arrive.
    """

    from montagewright.reframe import seconds_needed_for

    W = 0.3164
    near = seconds_needed_for([(0, 0.4, 0.5, W), (0, 0.6, 0.5, W)], "active")
    far = seconds_needed_for([(0, 0.15, 0.5, W), (0, 0.85, 0.5, W)], "active")
    three = seconds_needed_for(
        [(0, 0.15, 0.5, W), (0, 0.5, 0.5, W), (0, 0.85, 0.5, W)], "active"
    )

    assert near < far < three
    # And the same shot needs longer when it was asked to move calmly.
    assert seconds_needed_for(
        [(0, 0.15, 0.5, W), (0, 0.85, 0.5, W)], "calm"
    ) > far


def test_time_the_planner_asked_for_is_counted_in():
    # A shot that stops to be read needs the reading time on top of the
    # travel; a two-second look is two seconds whatever the distance.
    from montagewright.reframe import seconds_needed_for

    W = 0.3164
    quick = seconds_needed_for([(0, 0.2, 0.5, W), (0, 0.8, 0.5, W)], "active")
    read = seconds_needed_for([(1, 0.2, 0.5, W), (1, 0.8, 0.5, W)], "active")

    assert read - quick > 1.0


def test_an_unknown_distance_is_not_a_zero_one(tmp_path):
    """All the looks or none.

    A half-known path gives a travel distance that is wrong in a way nobody
    can see, and a floor built on it would be confidently too small.
    """

    from montagewright.cli import _look_boxes
    from montagewright.schema import reframe_of

    card = {
        "subjects": [{
            "label": "the left watch", "centre_x": 0.2, "centre_y": 0.5,
            "width": 0.1, "height": 0.3, "moves": False, "at_seconds": 0.0,
        }],
    }
    both = reframe_of({"looks": [{"at": "the left watch"}, {"at": "a ghost"}]})
    one = reframe_of({"looks": [{"at": "the left watch"}]})

    assert _look_boxes(card, both) == []
    assert len(_look_boxes(card, one)) == 1


def test_the_rhythm_prompt_asks_for_a_shape_not_a_beat_count():
    """The old one opened on music and got beats back.

    Eight shots quantised to six, seven or eight, four of them the same
    length to the centisecond, with reasons reading "8 beats" -- which the
    prompt already forbade. Naming the failure was not enough; the question
    itself was about the track.
    """

    from pathlib import Path

    prompt = (
        Path(__file__).resolve().parents[1] / "src" / "montagewright"
        / "prompts" / "rhythm_zh-TW.txt"
    ).read_text(encoding="utf-8")

    # The sequence is the subject, and sameness is named as the failure.
    assert "一串鏡頭合起來的形狀" in prompt
    assert "每顆都差不多長，就是還沒有做這件事" in prompt
    # Music is an input rather than the ruler, and its absence is normal.
    assert "音樂是另一個輸入，不是尺" in prompt
    assert "沒有配樂的時候不必找替代的格線" in prompt
    # And the measured floor is described as measured.
    assert "運鏡本身至少要" in prompt


def test_the_planner_is_shown_each_shot_s_measured_floor():
    from montagewright.planner import _needs_at_least
    from montagewright.schema import Clip, reframe_of

    reframe = reframe_of({"looks": [{"at": "A"}, {"at": "B"}], "why": "x"})
    reframe = reframe.model_copy(update={
        "look_boxes": [(0.15, 0.5, 1.0), (0.85, 0.5, 1.0)],
    })
    clip = Clip(
        clip_id="k00", source_id="s", approx_in_seconds=0.0,
        approx_out_seconds=3.0, reframe=reframe,
    )

    assert _needs_at_least(clip) > 0.0
    # Unknown geometry cannot justify the removed flat MOVE_FLOORS fallback;
    # only measured travel or declared rests may create a camera floor.
    bare = Clip(
        clip_id="k01", source_id="s", approx_in_seconds=0.0,
        approx_out_seconds=3.0,
        reframe=reframe_of({"looks": [{"at": "A"}, {"at": "B"}], "why": "x"}),
    )
    assert _needs_at_least(bare) == 0.0


def test_no_prompt_teaches_a_field_the_schema_does_not_have():
    """The selection prompt spent an hour describing `then_subject`.

    It was added to explain how to read across a wide subject, the schema
    was replaced with `looks` shortly after, and the prompt kept teaching a
    field that no longer existed -- the same shape as the bug it had been
    added to fix, which was a prompt recommending an impossible combination.
    """

    import re
    from pathlib import Path

    from montagewright.planner import _selection_schema

    shot = _selection_schema(["C1"])["properties"]["shots"]["items"]
    real = set(shot["properties"]) | set(
        shot["properties"]["looks"]["items"]["properties"]
    )

    prompts = Path(__file__).resolve().parents[1] / "src" / "montagewright" / "prompts"
    retired = {"then_subject", "camera_move"}
    offenders = []
    for name in ("selection_zh-TW.txt", "replan_zh-TW.txt"):
        text = (prompts / name).read_text(encoding="utf-8")
        for field in sorted(retired | {"subject"}):
            if field in real:
                continue
            # Backticked, so `camera_moves` in prose does not count.
            if re.search(rf"`{field}`", text):
                offenders.append(f"{name} still teaches `{field}`")

    assert not offenders, "\n  ".join(offenders)


def test_the_music_lane_is_drawn_over_the_span_the_cut_uses(tmp_path):
    """It trimmed by report.json, which is written last.

    So a run that stopped partway drew the whole track: a 2m35s bed
    stretched across a 29s timeline, every position on the lane pointing at
    the wrong moment, in the one view whose job is showing where sound sits.
    The film is on disk and its length is the answer.
    """

    import inspect

    import montagewright.webapp as web

    source = inspect.getsource(web.create_app)
    drawing = source[source.index('def waveform('):]
    drawing = drawing[: drawing.index("@app.get")]

    measured = drawing.index("probe_duration(")
    fallback = drawing.index("run.report()")

    assert measured < fallback, (
        "the cut's own length has to be measured before the report is "
        "consulted, because a crashed run has a film and no report"
    )


# --- the music is edited too --------------------------------------------

def test_the_bed_starts_where_the_rhythm_pass_pointed(tmp_path):
    """A 30s cut of a 2m35s track always took the first 30 seconds.

    Which is the intro -- written to have no energy yet -- so the picture
    carried the whole film alone, and no amount of pacing helped because
    there was nothing underneath it.
    """

    import subprocess

    from montagewright.renderer import _mux_music

    music = tmp_path / "track.wav"
    # Silent for the first eight seconds, then a tone. Silence rather than a
    # quiet intro because the chain ends in loudnorm, which pulls any two
    # real signals to the same loudness -- the first version of this test
    # measured -13.1 dB either way and proved nothing.
    subprocess.run(
        ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
         "-f", "lavfi", "-i", "anullsrc=r=48000:cl=mono:d=8",
         "-f", "lavfi", "-i", "sine=f=220:d=8:r=48000",
         "-filter_complex", "[0:a][1:a]concat=n=2:v=0:a=1",
         str(music)],
        check=True,
    )
    picture = tmp_path / "pic.mp4"
    subprocess.run(
        ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
         "-f", "lavfi", "-i", "color=c=black:s=160x90:r=25:d=4",
         "-f", "lavfi", "-i", "anullsrc=r=48000:cl=stereo", "-t", "4",
         "-c:v", "libx264", "-c:a", "aac", "-shortest", str(picture)],
        check=True,
    )

    def loudness(path):
        out = subprocess.run(
            ["ffmpeg", "-nostdin", "-hide_banner", "-t", "2", "-i", str(path),
             "-af", "volumedetect", "-f", "null", "-"],
            capture_output=True, text=True,
        )
        line = next(l for l in out.stderr.splitlines() if "mean_volume" in l)
        return float(line.split("mean_volume:")[1].split("dB")[0])

    intro = tmp_path / "from-intro.mp4"
    chorus = tmp_path / "from-chorus.mp4"
    _mux_music(picture, music, intro, video_encoder="libx264",
               music_from_seconds=0.0, fade_out_seconds=0.0)
    _mux_music(picture, music, chorus, video_encoder="libx264",
               music_from_seconds=9.0, fade_out_seconds=0.0)

    # Starting at zero scores the film with the silence; starting at nine
    # scores it with the tone.
    assert loudness(intro) < -60.0
    assert loudness(chorus) > -30.0


def test_the_bed_ends_rather_than_stopping(tmp_path):
    # A track cut off mid-phrase is the most audible thing in an otherwise
    # finished cut.
    import inspect

    from montagewright.renderer import MUSIC_FADE_SECONDS, _mux_music

    assert MUSIC_FADE_SECONDS > 0
    said = inspect.getsource(_mux_music)
    assert "afade=t=out" in said
    # Every mix gets it -- with a voice under the bed and without -- since
    # the one that had neither is the b-roll case, which is most of them.
    assert said.count("{tail}{fade}") >= 2


def test_a_start_past_the_end_of_the_track_is_pulled_back():
    # Pointing at 2:00 of a 2:35 track for a 60s cut would run off the end.
    import inspect

    from montagewright.renderer import _mux_music

    assert "spare = max(0.0, (probe_duration(music) or 0.0) - duration)" in (
        inspect.getsource(_mux_music)
    )


def test_the_rhythm_pass_is_told_it_can_choose_the_section():
    # A field nobody is told about is a field nobody fills. The analysis has
    # measured section boundaries all along and the prompt already listed
    # them -- there was just nowhere to say "start at the third one".
    from pathlib import Path

    from montagewright.planner import _rhythm_schema

    assert "music_from_seconds" in _rhythm_schema(["k00"])["properties"]

    prompt = (
        Path(__file__).resolve().parents[1] / "src" / "montagewright"
        / "prompts" / "rhythm_zh-TW.txt"
    ).read_text(encoding="utf-8")
    assert "music_from_seconds" in prompt
    assert "intro 就是寫成還沒有能量的" in prompt


def test_a_join_lands_on_a_phrase_line(tmp_path):
    """A splice anywhere else in the bar is audible however clean it is."""

    from montagewright.grounding import BeatGrid, Cue

    grid = BeatGrid(
        bpm=117, meter=4, duration_seconds=155.0,
        cues=(
            Cue("section-001", 16.4, "section_boundary"),
            Cue("section-002", 57.9, "section_boundary"),
        ),
    )

    # A boundary the analyser actually found wins over the grid, because it
    # is where the music itself changes.
    assert grid.on_phrase(15.0) == 16.4
    assert grid.on_phrase(58.5) == 57.9
    # With nothing near, it lands on a four-bar line.
    span = grid.phrase_seconds()
    assert abs(grid.on_phrase(100.0) % span) < 0.01


def test_pieces_of_the_track_are_played_in_order_and_cut_to_the_picture(tmp_path):
    """Three spans of a real track render, joined, at the picture's length.

    A chain that parses is not a chain that produces audio -- the first
    version read `[1:a][1:a]atrim=`, because a spliced bed takes its input
    more than once and cannot be written as a suffix on one label.
    """

    import subprocess

    from montagewright.renderer import _mux_music, _spliced

    before, tail = _spliced([(1.0, 3.0), (6.0, 8.0)], 3.0)
    assert before.count("[1:a]") == 2
    assert "acrossfade" in before
    assert tail.startswith("[j1]")

    music = tmp_path / "track.wav"
    subprocess.run(
        ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-f", "lavfi",
         "-i", "sine=f=330:d=12:r=48000", str(music)], check=True,
    )
    picture = tmp_path / "pic.mp4"
    subprocess.run(
        ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
         "-f", "lavfi", "-i", "color=c=black:s=160x90:r=25:d=3",
         "-f", "lavfi", "-i", "anullsrc=r=48000:cl=stereo", "-t", "3",
         "-c:v", "libx264", "-c:a", "aac", "-shortest", str(picture)],
        check=True,
    )
    made = tmp_path / "out.mp4"
    _mux_music(picture, music, made, video_encoder="libx264",
               music_spans=[(1.0, 3.0), (6.0, 8.0)])

    got = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=nw=1:nk=1", str(made)],
        capture_output=True, text=True,
    ).stdout.strip()
    assert abs(float(got) - 3.0) < 0.2


def test_a_length_the_caller_fixed_is_not_a_length_to_decide():
    # Writing "make it 15 seconds" in the brief is a request the direction
    # pass weighs; --seconds is the slot the film has to fit.
    import inspect

    from montagewright.planner import decide_direction

    said = inspect.getsource(decide_direction)
    assert "精確規格" in said
    assert "偏好上限" in said
    assert "duration_mode" in said
    # Overwritten rather than trusted: a pass that occasionally does not
    # repeat the number would silently change the film's length.
    assert 'decided["target_seconds"] = seconds' in said


def test_the_reviewer_is_told_the_length_that_was_asked_for():
    """Nothing else was checking it.

    Three planning layers each made a defensible call and delivered 17.9
    seconds against 30, and the only trace was a number in report.json that
    no stage read. Now that a caller can fix the length with --seconds, an
    unchecked target is a promise nobody keeps.
    """

    import inspect

    from montagewright.review import review_cut

    said = inspect.getsource(review_cut)
    assert "wanted_seconds" in said and "delivered_seconds" in said
    # Named as something to fix rather than something to scale away.
    assert "有顆不該在裡面，或有顆給得不夠" in said


def test_a_length_close_enough_is_not_raised_as_a_fault():
    # 29.1 against 30 is a cut that landed. Calling that a problem trains
    # the reviewer to spend rounds on arithmetic.
    import inspect

    from montagewright.review import review_cut

    said = inspect.getsource(review_cut)
    assert "max(1.5, wanted_seconds * 0.12)" in said
    assert "在範圍內" in said


def test_nothing_truncates_the_cut_to_the_target():
    """A target is honoured by planning or missed, never enforced.

    Cutting the film to length in the executor would be the execution layer
    answering an editorial question -- which shot goes -- and the report
    would describe a plan that was not what ran.
    """

    from pathlib import Path

    root = Path(__file__).resolve().parents[1] / "src" / "montagewright"
    for name in ("grounding.py", "executor.py", "renderer.py"):
        assert "target_seconds" not in (root / name).read_text(encoding="utf-8")


def test_the_page_says_which_part_of_the_track_was_used():
    """The rhythm pass chooses a section and the choice is audible.

    Nothing recorded it, so "why does the music sound like that" had no
    answer on the page -- and it is the field to look at to know whether
    choosing a section works at all.
    """

    from pathlib import Path

    root = Path(__file__).resolve().parents[1] / "src" / "montagewright"
    served = (root / "webapp.py").read_text(encoding="utf-8")
    page = (root / "web" / "index.html").read_text(encoding="utf-8")
    written = (root / "cli.py").read_text(encoding="utf-8")

    assert 'current.get("music_from_seconds"' in served
    assert '"music_from_seconds": getattr(plan, "music_from_seconds"' in written
    assert "function musicName()" in page
    assert "音樂 · 從" in page


def test_a_fixed_length_reaches_the_run_from_the_page():
    from pathlib import Path

    root = Path(__file__).resolve().parents[1] / "src" / "montagewright"
    served = (root / "webapp.py").read_text(encoding="utf-8")
    page = (root / "web" / "index.html").read_text(encoding="utf-8")

    assert 'seconds: float = Form(0.0)' in served
    assert '"--seconds"' in served
    assert 'id="seconds"' in page
    assert "body.append('seconds'" in page


def test_the_brief_and_the_flag_cannot_quietly_disagree():
    # Both reach the model. Which wins has to be said rather than inferred:
    # a flag that silently contradicts the brief makes the reasoning wrong
    # even when the output length comes out right.
    import inspect

    from montagewright.planner import decide_direction

    assert "brief 裡如果提到別的長度，以這裡為準" in inspect.getsource(
        decide_direction
    )


def test_the_delivery_aspect_is_the_request_not_the_model_s_choice():
    """Two aspects that agreed only by luck.

    The direction pass chose an aspect from an enum, uninformed by --aspect,
    and that choice was what selection and the interface were told to plan
    and draw for -- while the executor cropped to --aspect. When they
    differed, the film was planned for one frame and rendered in another.

    The aspect is a delivery requirement, so it is no longer the model's to
    pick: the schema does not offer it, and whatever the pass returns, the
    requested aspect is stamped onto the result that everything downstream
    reads.
    """

    from montagewright import planner

    schema = planner._direction_schema()
    assert "aspect" not in schema["properties"]
    assert "aspect" not in schema["required"]

    # A fake client that answers with a different aspect than requested; the
    # requested one is what comes back regardless.
    import json as _json

    class Reply:
        status = "complete"
        usage = {}

        def __init__(self, payload):
            self.output_text = _json.dumps(payload)

    class Client:
        def __init__(self, payload):
            self._payload = payload

        @property
        def interactions(self):
            return self

        def create(self, **_):
            return Reply(self._payload)

    answer = {
        "reasoning": "r", "material_assessment": "m", "direction": "d",
        "target_seconds": "0:20", "music_under_speech": "duck",
        "unusable": [], "aspect": "16:9",   # the model tries to say 16:9
    }
    decided, _ = planner.decide_direction(
        [], brief="", aspect="9:16", client=Client(answer),
    )
    assert decided["aspect"] == "9:16"


def test_direction_receives_the_same_measured_music_map_as_local_rhythm():
    """Listening to a track and executing unrelated clocks is two truths."""

    import json as _json

    from montagewright import planner
    from montagewright.grounding import BeatGrid, Cue

    class Reply:
        status = "complete"
        usage = {}

        def __init__(self, payload):
            self.output_text = _json.dumps(payload)

    class Client:
        def __init__(self):
            self.request = None

        @property
        def interactions(self):
            return self

        def create(self, **request):
            self.request = request
            return Reply({
                "reasoning": "r", "material_assessment": "m",
                "direction": "d", "target_seconds": "0:20",
                "music_under_speech": "duck", "unusable": [],
            })

    grid = BeatGrid(
        bpm=120.0, meter=4, duration_seconds=20.0,
        cues=(Cue("section-opening", 0.0, "section_boundary", 1.0),),
    )
    client = Client()
    planner.decide_direction(
        [], brief="", aspect="9:16", music_grid=grid, client=client,
    )
    prompt = str(client.request["input"][0]["text"])
    assert "section-opening" in prompt
    assert "BPM 120" in prompt
    assert "不要自創時間點" in prompt


def test_the_card_version_moves_when_the_card_s_shape_does():
    """Two required fields were added and no card was rewritten.

    Cards are keyed by the bytes they describe, so a schema change does not
    invalidate them -- only the version does, and that was a string somebody
    had to remember to bump. Every cached card stayed, `shot_size` and
    `facing` were absent from every listing that had asked for them, and the
    selection guidance written around them had nothing to read.
    """

    from montagewright import clipcard

    was = clipcard.card_schema

    def with_one_more():
        schema = was()
        schema["required"] = schema["required"] + ["something_new"]
        return schema

    before = clipcard._card_version()
    clipcard.card_schema = with_one_more
    try:
        assert clipcard._card_version() != before
    finally:
        clipcard.card_schema = was

    # A description change is not a shape change, and rewriting the library
    # for one would cost a full pass for nothing.
    assert clipcard._card_version() == before


def test_a_card_from_an_older_shape_is_not_loaded(tmp_path):
    import json

    from montagewright.clipcard import CARD_VERSION, load_card

    stale = tmp_path / "old.json"
    stale.write_text(
        json.dumps({"summary": "x", "version": "montagewright-clip-card-v1"}),
        encoding="utf-8",
    )
    current = tmp_path / "new.json"
    current.write_text(
        json.dumps({"summary": "x", "version": CARD_VERSION}), encoding="utf-8"
    )

    assert load_card(stale) is None
    assert load_card(current) is not None


def test_a_new_round_starts_from_what_the_open_run_was_made_from():
    """Opening a cut and asking for another round is how you try a length.

    The form started empty, so the answer to "will it use the footage I
    already pointed at" was no -- it would refuse for having no material,
    after the material had been sitting in the run that was open.
    """

    from pathlib import Path

    root = Path(__file__).resolve().parents[1] / "src" / "montagewright"
    served = (root / "webapp.py").read_text(encoding="utf-8")
    page = (root / "web" / "index.html").read_text(encoding="utf-8")

    assert '"source_path": _ran_with(run, "render", after=False)' in served
    assert '"music_path": _ran_with(run, "--music")' in served
    assert "openedFrom = {" in page
    # Only where nothing has been typed: a half-filled form belongs to
    # whoever was filling it.
    assert "!$('path').value.trim()" in page


def test_the_source_is_read_off_the_command_that_ran():
    import montagewright.webapp as web

    class Ran:
        command = [
            "python", "-m", "montagewright.cli", "render", "/rushes/clip",
            "--aspect", "9:16", "--music", "/music/track.mp3",
        ]

    assert web._ran_with(Ran(), "render", after=False) == "/rushes/clip"
    assert web._ran_with(Ran(), "--music") == "/music/track.mp3"
    assert web._ran_with(Ran(), "--brief") == ""


def test_the_readings_the_recogniser_weighed_reach_the_correction():
    """It always had them and was never asked.

    The prompt points the correction at the low-confidence words, which is a
    good marker -- but a marker says the recogniser was unsure, not what it
    was unsure between. So the correction invented a replacement out of the
    picture and the sense while the candidates it should have been choosing
    from came out of the audio and sat unread in the same result.
    """

    from montagewright.transcript import hesitations

    payload = {
        "utterances": [
            {
                "text": "有場",
                "starts_seconds": 1.0,
                "ends_seconds": 1.6,
                "alternatives": ["有廠", "有唱"],
                "words": [],
            },
            {
                "text": "所以你是",
                "starts_seconds": 1.6,
                "ends_seconds": 2.4,
                "alternatives": [],
                "words": [],
            },
        ]
    }
    weighed = hesitations(payload)
    assert weighed == [(1.0, 1.6, "有場", ["有廠", "有唱"])]

    # A transcript from before this was recorded simply has none.
    assert hesitations({"utterances": [{"text": "x", "starts_seconds": 0.0,
                                        "ends_seconds": 1.0}]}) == []
    assert hesitations({}) == []


def test_the_recogniser_is_asked_for_its_other_readings():
    """The option exists, is off by default, and costs nothing to turn on."""

    from pathlib import Path

    swift = (
        Path(__file__).resolve().parents[1] / "tools" / "transcribe"
        / "Transcribe.swift"
    ).read_text(encoding="utf-8")
    assert "reportingOptions: [.alternativeTranscriptions]" in swift
    assert "result.alternatives" in swift
    # Not the first reading again under another name, and not repeated.
    assert "said != text" in swift and "!others.contains(said)" in swift

    # A stretch with alternatives reaches the prompt as its own block: they
    # belong to a span of speech, not to a word, because a second reading of
    # a phrase can split it differently from the first.
    import inspect

    from montagewright import transcript

    building = inspect.getsource(transcript.describe)
    assert "hesitations(heard)" in building
    assert "{considered}" in building

    prompt = (
        Path(transcript.__file__).resolve().parent
        / "prompts" / "transcript_zh-TW.txt"
    ).read_text(encoding="utf-8")
    assert "辨識器猶豫過的地方" in prompt
    # And that a candidate is evidence, not an answer.
    assert "候選是從聲音來的" in prompt
    assert "不是正確答案的保證" in prompt
def test_old_transcript_with_apple_word_clock_migrates_without_retranscribing(
    tmp_path,
) -> None:
    import json

    from montagewright.transcript import ALIGNMENT_VERSION, CARD_VERSION, load

    path = tmp_path / "old.json"
    path.write_text(json.dumps({
        "version": "montagewright-transcript-older",
        "lines": [{
            "text": "吹頭髮很熱，還會流汗。",
            "speaker": "受訪者",
            "starts_seconds": 99,
            "ends_seconds": 100,
        }],
        "words": [
            {"text": "吹頭髮很熱", "starts_seconds": 2.0, "ends_seconds": 3.0},
            {"text": "還會流汗", "starts_seconds": 3.1, "ends_seconds": 4.0},
        ],
    }, ensure_ascii=False), encoding="utf-8")

    migrated = load(path)

    assert migrated is not None
    assert migrated["version"] == CARD_VERSION
    assert migrated["timing"]["aligner"] == ALIGNMENT_VERSION
    assert migrated["lines"][0]["starts_seconds"] == 2.0
    assert migrated["lines"][0]["ends_seconds"] == 4.0
    assert migrated["lines"][0]["timed_text"]


def test_travel_between_landings_takes_only_the_time_it_needs():
    """The crossing is a connective; the landings are the content.

    A Before panel and an After panel are two things worth looking at, and
    the frame between them shows half of each and neither whole. Handing
    travel a fixed share of the shot whatever it was for turned a plan of
    1.5s on each panel into 0.75s and 0.75s with the frame straddling the
    divider for the middle half of a three-second shot -- the planner had
    answered correctly and a constant overruled it. Energy already says how
    fast the frame may move; dwell already says how long each end is worth.
    """

    degradations = []
    path = _looks(
        [(1.5, 0.395, 0.5, WIDE), (1.5, 0.715, 0.5, WIDE)],
        seconds=3.0, degradations=degradations,
    )
    rests = [
        round(later.seconds - earlier.seconds, 2)
        for earlier, later in zip(path.keyframes, path.keyframes[1:])
        if abs(earlier.crop.x - later.crop.x) < 1e-4
    ]

    assert rests == [1.26, 1.26]
    moving = 3.0 - sum(rests)
    assert moving < 0.6
    # The plan's number did change, so it is said rather than absorbed.
    assert [one.ladder_other for one in degradations] == [
        "declared_dwell_shortened_to_fit"
    ]


def test_dwell_that_already_fits_is_left_exactly_alone():
    degradations = []
    path = _looks(
        [(0.5, 0.30, 0.5, WIDE), (0.5, 0.70, 0.5, WIDE)],
        seconds=3.0, degradations=degradations,
    )
    rests = [
        round(later.seconds - earlier.seconds, 2)
        for earlier, later in zip(path.keyframes, path.keyframes[1:])
        if abs(earlier.crop.x - later.crop.x) < 1e-4
    ]

    assert rests == [0.5, 0.5]
    assert not degradations


def test_a_route_too_long_for_its_shot_still_stops_at_both_ends():
    """Moving in every frame is a pan with both ends cut off.

    When the crossing cannot fit at all, the landings keep a readable settle
    and the overrun is reported by the speed checks, rather than the shot
    never stopping.
    """

    degradations = []
    path = _looks(
        [(2.0, 0.05, 0.5, WIDE), (2.0, 0.95, 0.5, WIDE)],
        seconds=2.5, energy="calm", degradations=degradations,
    )
    rests = [
        later.seconds - earlier.seconds
        for earlier, later in zip(path.keyframes, path.keyframes[1:])
        if abs(earlier.crop.x - later.crop.x) < 1e-4
    ]

    assert rests and min(rests) >= 0.34
    assert "looks_do_not_fit_the_time" in [
        one.ladder_other for one in degradations
    ]


def test_a_digital_move_leaves_room_for_the_takes_own_motion():
    """What the viewer sees is the sum of the two moves.

    A take panning at 0.08 of frame a second, with the digital crop given the
    full 0.67 an active shot allows, moves at 0.75 on screen -- past the
    ceiling the energy names -- and then both stop at once, which reads as a
    rebound. The crop's budget is the ceiling less what the take already
    spends, so the composite stays within the limit.
    """

    native = 0.083
    fast = _looks(
        [(1.0, 0.16, 0.5, WIDE), (1.5, 0.69, 0.5, WIDE)],
        seconds=3.111, native_speed=native,
    )
    peak = max(
        abs((later.crop.x + later.crop.width / 2)
            - (earlier.crop.x + earlier.crop.width / 2))
        / max(later.seconds - earlier.seconds, 1e-9)
        for earlier, later in zip(fast.keyframes, fast.keyframes[1:])
    )

    assert peak + native <= 0.67 + 1e-3
    # A locked take spends nothing, so the same shape keeps the full budget
    # and travels faster.
    locked = _looks(
        [(1.0, 0.16, 0.5, WIDE), (1.5, 0.69, 0.5, WIDE)],
        seconds=3.111, native_speed=0.0,
    )
    locked_peak = max(
        abs((later.crop.x + later.crop.width / 2)
            - (earlier.crop.x + earlier.crop.width / 2))
        / max(later.seconds - earlier.seconds, 1e-9)
        for earlier, later in zip(locked.keyframes, locked.keyframes[1:])
    )
    assert locked_peak > peak


def test_a_move_that_drifts_past_the_takes_settle_is_reported():
    """A crop still travelling after the source locks off is a lone drift.

    It is not silently retimed: compressing the move into the pre-settle
    window would raise its speed past the budget on exactly the shots that
    settle earliest, so the disagreement is surfaced for review instead.
    """

    degradations = []
    _looks(
        [(0.4, 0.16, 0.5, WIDE), (0.4, 0.84, 0.5, WIDE)],
        seconds=3.0, degradations=degradations, native_settles_at=0.6,
    )
    drift = [
        one for one in degradations
        if one.ladder_other == "digital_moves_after_take_settles"
    ]
    assert len(drift) == 1
    m = drift[0].measured
    assert m["settles_at_seconds"] == 0.6
    assert m["digital_arrives_seconds"] > 0.3
    assert drift[0].severity == "advisory"


def test_a_move_that_lands_before_the_take_settles_is_not_flagged():
    degradations = []
    _looks(
        [(0.4, 0.16, 0.5, WIDE), (0.4, 0.84, 0.5, WIDE)],
        seconds=3.0, degradations=degradations, native_settles_at=2.9,
    )
    assert not [
        one for one in degradations
        if one.ladder_other == "digital_moves_after_take_settles"
    ]


def test_a_take_that_barely_moved_is_not_a_drift():
    """A source that locked inside the first readable settle barely moved,
    so a digital move over it is the shot's intended motion, not a drift."""

    degradations = []
    _looks(
        [(0.4, 0.16, 0.5, WIDE), (0.4, 0.84, 0.5, WIDE)],
        seconds=3.0, degradations=degradations, native_settles_at=0.12,
    )
    assert not [
        one for one in degradations
        if one.ladder_other == "digital_moves_after_take_settles"
    ]
