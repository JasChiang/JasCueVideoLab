"""Every field a prompt names is a field its schema has.

This bug has now landed three times: `camera_move` removed from the selection
schema while six readers still asked for it, `then_subject` taught in prose
after it stopped existing, and the direction prompt telling the model to fill
`material_notes` with a `supersedes` key -- against a schema whose only list
is `unusable`, keyed `superseded_by`, pointing the other way.

None of them raised. A model handed instructions for a field that is not in
its response schema writes something plausible into the fields it does have,
and the run completes, and the guidance silently did nothing.

The existing prompt tests assert that particular sentences are present, which
catches a deletion and nothing else -- a sentence can stay word for word
correct while the field it names is renamed underneath it. This asserts the
join instead: pull every `backticked` identifier out of each prompt and
require the paired schema to define it.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from montagewright import clipcard, planner, review, transcript

PROMPTS = Path(planner.__file__).resolve().parent / "prompts"

# `looks[].must_be_whole` is one name written as a path; `.` and `[]` are
# punctuation around the parts, not part of any one of them.
NAME = re.compile(r"`([A-Za-z_][A-Za-z0-9_]*(?:(?:\[\])?\.[A-Za-z_][A-Za-z0-9_]*)*)`")


def names_in(prompt: str) -> set[str]:
    found: set[str] = set()
    for whole in NAME.findall(prompt):
        found.update(part for part in whole.replace("[]", "").split(".") if part)
    return found


def names_of(schema: dict) -> set[str]:
    """Every property name and every enum value anywhere in the schema.

    Both, because a prompt legitimately names either: `unusable` is a field
    and `bed` is one of the values `music_under_speech` accepts, and the
    prompt has to be able to say both without the test caring which is which.
    """

    found: set[str] = set()
    if not isinstance(schema, dict):
        return found
    for value in schema.get("enum", []) or []:
        if isinstance(value, str):
            found.add(value)
    properties = schema.get("properties")
    if isinstance(properties, dict):
        for key, child in properties.items():
            found.add(key)
            found |= names_of(child)
    for key in ("items", "additionalProperties"):
        found |= names_of(schema.get(key))
    return found


# Words a prompt backticks that are not the model's to write: things it is
# told about rather than asked for. Each is here because it is read from
# somewhere else, not because the check was inconvenient.
SPOKEN_OF = {
    # The transcript's per-line times and `heard` are written by local code
    # after the answer comes back, and the shot review names the ladder rungs
    # the executor records rather than any field of its own.
    "clipcard_zh-TW.txt": set(),
    # `candidate_options` is a real direction field, but it is attached to the
    # schema only when grounding/commitments are enabled (planner ~1675), so
    # the bare _direction_schema() the lint builds does not list it. It is a
    # genuine field the prompt must name, so it is allowed here explicitly.
    "direction_zh-TW.txt": {"candidate_options"},
    # Selection reads the card's vocabulary off the material listing and
    # never writes it: shot size, which way a subject faces, and how the
    # content is laid out are all decided one step earlier. Taken from the
    # card schema rather than listed, so renaming one of those values fails
    # here too instead of quietly joining the allowlist.
    "selection_zh-TW.txt": "card",
    "rhythm_zh-TW.txt": {"beats", "chorus_1_start"},
    "replan_zh-TW.txt": set(),
    "review_zh-TW.txt": set(),
    "shotreview_zh-TW.txt": {"none_recorded"},
    "transcript_zh-TW.txt": set(),
}

PAIRED = [
    ("clipcard_zh-TW.txt", lambda: clipcard.card_schema()),
    ("direction_zh-TW.txt", lambda: planner._direction_schema()),
    ("selection_zh-TW.txt", lambda: planner._selection_schema(["A"])),
    ("rhythm_zh-TW.txt", lambda: planner._rhythm_schema(["k00"])),
    (
        "replan_zh-TW.txt",
        lambda: planner._selection_schema(["A"], replace_clip_ids=["k00"]),
    ),
    ("review_zh-TW.txt", lambda: review._verdict_schema()),
    ("shotreview_zh-TW.txt", lambda: review._shot_schema(["k00"])),
    ("transcript_zh-TW.txt", lambda: transcript._schema()),
]


def _allowed(filename: str) -> set[str]:
    spoken = SPOKEN_OF[filename]
    return names_of(clipcard.card_schema()) if spoken == "card" else spoken


@pytest.mark.parametrize("filename,build", PAIRED, ids=[name for name, _ in PAIRED])
def test_a_prompt_only_names_fields_its_schema_defines(filename, build):
    prompt = (PROMPTS / filename).read_text(encoding="utf-8")
    unknown = names_in(prompt) - names_of(build()) - _allowed(filename)
    assert not unknown, (
        f"{filename} instructs the model about "
        f"{sorted(unknown)}, which its response schema does not define"
    )


def test_the_check_would_notice_a_renamed_field():
    """Reintroducing the bug has to fail this, or it guards nothing."""

    schema = {"properties": {"unusable": {"type": "array"}}}
    assert names_in("填 `material_notes`，用 `supersedes` 指出") - names_of(schema) == {
        "material_notes",
        "supersedes",
    }


def test_the_microphone_is_not_offered_as_evidence_of_who_is_speaking():
    """It was, and in an interview it points the other way.

    The prompt listed three cues for identifying a speaker -- whose mouth is
    moving, who is holding the microphone, who the camera is on -- and two of
    them are unreliable in exactly the material this runs on. The microphone
    in a street interview is held by the person asking and extended toward
    the person answering, so "who holds it" identifies the one not talking;
    the camera is on the listener during every reaction shot.

    This matters beyond the label. Selection is told to frame whoever is
    speaking using these same words, so a wrong attribution becomes a talking
    shot pointed at the wrong person.
    """

    # The listening pass is the one that can see anybody, so that is where
    # the cues live now. The correction pass never gets the video.
    heard = (PROMPTS / "hearing_zh-TW.txt").read_text(encoding="utf-8")
    assert "拿著麥克風的人通常不是正在講的那個" in heard
    assert "鏡頭對著誰不代表誰在講" in heard
    # And the reliable cue is the one that does not need the picture at all.
    assert "判斷以話的內容為準" in heard

    prompt = (PROMPTS / "transcript_zh-TW.txt").read_text(encoding="utf-8")
    speaking = prompt[prompt.index("## 誰在講"):]
    speaking = speaking[: speaking.index("## 切成字幕")]

    # What is left for the correction is the case the blocks cannot answer:
    # the recogniser merges two voices with no pause between them into one
    # stretch, and only the words say where the turn changed.
    assert "以**話的內容**為準" in speaking
    assert "換人的地方一定要斷行" in speaking
    # Not knowing is an available answer, because guessing here is invisible.
    assert "uncertain" in speaking

    # The schema says the same thing: a model reads the field description
    # even when the prompt scrolls past.
    from montagewright.transcript import _schema

    said = _schema()["properties"]["lines"]["items"]["properties"]["speaker"]
    assert "麥克風都不是證據" in said["description"]


def test_a_talking_shot_is_checked_against_who_is_talking():
    """The prompt called it the most obvious error and nothing looked for it.

    Neither reviewer mentioned the speaker. The one that watches a single
    shot against its own plan is the only one that can see it -- in a
    finished cut the next shot covers it in a second and a half.
    """

    shots = (PROMPTS / "shotreview_zh-TW.txt").read_text(encoding="utf-8")
    assert "正在講話的那一個" in shots
    assert "delivered` 就是 false" in shots
    # Off-screen speech is not the same fault, or every voiceover fails.
    assert "不在畫面裡" in shots


def test_the_second_listener_never_sees_the_first_ones_answer():
    """Order, not count, is what the split is for.

    Shown a transcript and asked to fix it, a model agrees with any line that
    reads well -- and the errors hardest to catch are exactly the ones that
    read well: a plausible word that is not the word that was said. Only a
    listener that has not seen the answer can disagree with it.
    """

    import inspect

    from montagewright import transcript

    source = inspect.getsource(transcript.describe)
    listening = source[source.index("listening = ask("):source.index("listened = _parse")]
    # The video goes to the first call and the recogniser's words do not.
    assert '"type": "video"' in listening
    assert "hearing_zh-TW.txt" in listening
    assert "rough" not in listening

    # The second call carries no media at all: everything the picture had to
    # say was said by the first one.
    correcting = source[source.index("instruction = "):source.index("payload = _parse")]
    assert '"type": "video"' not in correcting
    assert "uri" not in correcting
    assert "rough" in correcting and "said_by_ear" in correcting

    # And the recogniser's clock reaches neither of them.
    assert "starts_seconds" not in source[source.index("rough = "):source.index("listening = ask(")]


def test_the_terms_the_video_pass_harvested_reach_the_correction():
    """The working version of an idea that failed one layer lower.

    Feeding the recogniser a vocabulary through `contextualStrings` was
    measured and did nothing -- byte-identical output with words that were in
    the audio and had been misheard. The same idea at the correction layer
    works, because that is a model that reads what it is given, and the terms
    come from the pass that just watched the clip including whatever was
    written on screen.
    """

    import inspect

    from montagewright import transcript

    source = inspect.getsource(transcript.describe)
    assert 'listened.get("terms")' in source
    assert "專有名詞" in source

    schema = transcript._hearing_schema()["properties"]
    assert "terms" in schema
    assert "辨識器很可能聽錯" in schema["terms"]["description"]

    prompt = (PROMPTS / "transcript_zh-TW.txt").read_text(encoding="utf-8")
    assert "專有名詞清單" in prompt


def test_the_correction_is_told_where_the_second_listener_is_wrong():
    """A second opinion that is trusted everywhere is a second set of errors.

    It is the better witness on soundalikes and the worse one on disfluency:
    it tidies stutters and false starts away, and a caption that has been
    tidied no longer matches the sound it sits on.
    """

    prompt = (PROMPTS / "transcript_zh-TW.txt").read_text(encoding="utf-8")
    second = prompt[prompt.index("## 另一個聽眾"):]
    second = second[: second.index("## 誰在講")]

    assert "盲聽" in second
    assert "永遠不要為了跟它一致而刪掉重複的字" in second
    assert "不要從它那裡引進辨識器完全沒有的整句話" in second

    # And the listening prompt fights for the disfluencies in the first place.
    heard = (PROMPTS / "hearing_zh-TW.txt").read_text(encoding="utf-8")
    assert "贅字、口頭禪、結巴、重複" in heard


# Every schema a model answers, and the fields in each that are a time.
def _time_fields(schema: dict, path: str = ""):
    if not isinstance(schema, dict):
        return
    for key, child in (schema.get("properties") or {}).items():
        here = f"{path}.{key}" if path else key
        if isinstance(child, dict) and TIMEY.search(key):
            yield here, child
        yield from _time_fields(child, here)
    yield from _time_fields(schema.get("items"), path + "[]")


TIMEY = re.compile(r"second|_at$|^at$|^from$|^to$|hold|dur", re.I)

# `at` on a look is what the frame settles on, described in words, and
# `sync_to` names a section rather than a moment. Both read as times and are
# not; nothing else in any schema gets to be an exception.
NOT_A_TIME = {"shots[].looks[].at", "decisions[].sync_to"}

ANSWERED = [
    ("clip card", lambda: clipcard.card_schema()),
    ("direction", lambda: planner._direction_schema()),
    ("selection", lambda: planner._selection_schema(["A:s00"])),
    ("rhythm", lambda: planner._rhythm_schema(["k00"])),
    ("subject", lambda: planner._subject_schema(3)),
    ("review", lambda: review._verdict_schema()),
    ("shot review", lambda: review._shot_schema(["k00"])),
    ("hearing", lambda: transcript._hearing_schema()),
    ("transcript", lambda: transcript._schema()),
]


def test_the_card_states_whether_a_screen_is_lit():
    """A dark screen was narrated as a screen being shown.

    A model unfolded a phone whose display was off; the card called it a
    "black big-screen phone" and its summary said the shot "reveals the big
    screen". Both from one pass, and contradictory: the colour word noticed
    the screen was dark while the summary asserted it was showing something.
    A brief that asked for lit screens caught the static displays whose cards
    said "unlit", and missed this one, because the only pass that watches the
    footage had described the dark screen as content.

    Whether a display is lit is an observable fact, not a product-specific
    judgement -- it holds for a phone, a watch, a monitor. The card is told
    to state it, so a downstream reader has something to act on.
    """

    prompt = (PROMPTS / "clipcard_zh-TW.txt").read_text(encoding="utf-8")
    assert "會亮的裝置面板" in prompt or "螢幕" in prompt
    # The specific trap: a dark screen is not the screen being shown.
    assert "暗掉的螢幕不是" in prompt


@pytest.mark.parametrize("name,build", ANSWERED, ids=[one for one, _ in ANSWERED])
def test_every_time_a_model_writes_is_a_clock_reading(name, build):
    """A number is two readings; `1:53` is one.

    Gemini reads video in MM:SS, so a field asking for bare seconds is asking
    it to convert -- and both ways it gets that wrong are silent. A 71.1s take
    said its usable range ended at `110.0`, which is 1:10 with the colon
    dropped. A 113.4s take said `1.53`, which is 1:53 with the colon turned
    into a point, and that one passes every range check on the way through
    while claiming two minutes of material is good for a second and a half.

    The rule is the whole rule: everything through a model is a clock, and
    anything else is converted here. Sub-second precision is not lost by
    this, because a model watching video at a frame a second never had any --
    what precision exists is measured locally and applied after.
    """

    numeric = [
        path for path, field in _time_fields(build())
        if path not in NOT_A_TIME and field.get("type") != "string"
    ]
    assert not numeric, (
        f"{name} asks the model for {numeric} as a number; times that go "
        f"through a model are M:SS, and local code converts"
    )


def test_one_reader_turns_every_one_of_them_back_into_seconds():
    """The conversion happens at the boundary, not at each reader.

    Downstream wants floats and should not learn a notation to get them, so
    each pass normalises its own answer as it parses it -- the same move
    `expand_spans` makes for the span itself.
    """

    import inspect

    from montagewright import planner, review, transcript

    for where in (
        planner.expand_spans,          # selection and replan
        planner.decide_direction,      # target length
        planner._apply,                # rhythm holds and music positions
        review.review_cut,             # where in the cut an issue sits
    ):
        assert "seconds_of" in inspect.getsource(where), where.__name__

    # And the card, whose times are read straight off the stored JSON.
    assert "seconds_of" in inspect.getsource(clipcard.times_on_receipt)
    assert "seconds_of" in inspect.getsource(clipcard.action_beats)
    # The transcript pass asks for no times at all, which is the same rule
    # taken to its end.
    assert "second" not in json.dumps(
        transcript._schema()["properties"]["lines"], ensure_ascii=False
    )


def test_the_card_version_moves_when_anything_about_the_card_moves():
    """It hashed the top-level required names and nothing else.

    So `segments` could change shape, an enum could gain a value, a unit
    could flip from seconds to a clock reading, and every cached card stayed
    valid while meaning something different. All three happened.
    """

    import hashlib
    import json as _json

    from montagewright import clipcard
    from montagewright.motion import READING

    was = clipcard.CARD_VERSION
    assert was.startswith("montagewright-clip-card-")

    # The digest covers the whole schema, the prompt beside it, and what the
    # measurement it is shown means -- the card is answering all three.
    shape = _json.dumps(clipcard.card_schema(), sort_keys=True, ensure_ascii=False)
    prompt = (PROMPTS / "clipcard_zh-TW.txt").read_text(encoding="utf-8")
    expected = hashlib.sha256((shape + prompt + READING).encode("utf-8")).hexdigest()[:8]
    assert was.endswith(expected)

    # A nested change moves it, which the old version could not see.
    deeper = _json.loads(shape)
    deeper["properties"]["segments"]["items"]["properties"]["status"]["enum"].append("maybe")
    moved = hashlib.sha256(
        (_json.dumps(deeper, sort_keys=True, ensure_ascii=False) + prompt + READING)
        .encode("utf-8")
    ).hexdigest()[:8]
    assert moved != expected


def test_whether_the_camera_moved_is_measured_rather_than_asked():
    """It was an optional boolean answered by a model reading 1 fps.

    Which is the rate at which a moving camera and a still one look the
    same -- and being optional, a missing answer became "still" on the field
    that decides whether a digital move gets stacked on a take that already
    has one. It is a fact about the pixels, so it is read off them.
    """

    import inspect

    from montagewright import cli

    assert "camera_moves" not in clipcard.card_schema()["properties"]
    filled = inspect.getsource(cli.command_render)
    filled = filled[filled.index("camera_moves="):]
    assert 'one.state == "moving"' in filled[:400]


def test_a_decision_is_keyed_on_the_material_it_was_made_from():
    """Source ids, brief, aspect and a music path -- and nothing else.

    So a card rewritten with better segments, or a span boundary moved, left
    the key identical and the next run reused a selection made against
    material that no longer had that shape.
    """

    import inspect

    from montagewright import cli

    keyed = inspect.getsource(cli.command_render)
    keyed = keyed[keyed.index("catalogue = _asked("):keyed.index("direction = _decided(")]
    assert "CARD_VERSION" in keyed
    assert "span_id" in keyed
    assert "starts_seconds" in keyed and "ends_seconds" in keyed
