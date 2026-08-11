"""Data contracts for the autonomous edit pipeline.

Two rules shape everything here, and both come from what the previous system
got wrong rather than from taste.

Coordinates are normalised floats in 0..1 everywhere inside the pipeline. The
old code carried pixel speeds and viewport fractions side by side, which made
a limit meaningful only next to the resolution it was written against. Pixels
appear once, at the ffmpeg boundary.

A constraint the local layer will reject has to be stated in the contract the
model reads. Five separate rules in the previous system lived only inside
validators -- a classified motion role needing a reason, an atomic region not
also declaring a visible fraction, and three more -- and each one cost a full
paid plan to discover. Every coupling below is written where the model sees
it, and every closed vocabulary carries an `other` escape so an unforeseen
answer degrades to a described value instead of a rejected response.
"""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

# 0..1 across the frame, origin top-left. Never pixels.
Normalised = Annotated[float, Field(ge=0.0, le=1.0)]

CameraEnergy = Literal["calm", "active", "dynamic"]
CoverageType = Literal["literal", "implied", "tonal"]
EnergyIntent = Literal["low", "medium", "high"]
AudioRole = Literal[
    "auto", "discard", "narrative", "sync_action", "ambient_texture"
]
PictureRole = Literal[
    "speaker", "primary_action", "illustrative_broll", "reaction",
    "establishing", "transition", "punchline_hold", "end_hold",
    "title_read", "music_montage",
]
AudioCompletion = Literal[
    "none", "complete_thought", "complete_action_sound", "intentional_cut"
]


class ModelFacing(BaseModel):
    """Base for anything the model fills in.

    Unknown keys are kept rather than rejected: an extra field is the model
    volunteering something, and throwing away a paid response over it buys
    nothing. Callers log what they ignored.
    """

    model_config = ConfigDict(extra="allow")


class Local(BaseModel):
    """Base for anything only local code writes."""

    model_config = ConfigDict(extra="forbid")


class NamedFact(ModelFacing):
    """One product name, model number, year, or on-screen string, as seen.

    Transcribed before planning and quoted by everything downstream. The point
    is that a 2026 handset is newer than the model's training data, so its own
    memory of what a model number "should" be is the least reliable source in
    the room.
    """

    text: str = Field(
        description=(
            "The string exactly as it appears, including spacing and any "
            "character that looks like a typo. Never corrected, normalised, "
            "or completed."
        )
    )
    source_id: str = Field(description="Asset the string was read from.")
    at_seconds: float = Field(
        ge=0.0, description="Roughly when it is legible in that asset."
    )
    legibility: Literal["clear", "partial", "uncertain"] = Field(
        description=(
            "How readable it actually was. 'uncertain' is the honest answer "
            "for a guess and costs nothing; a confident wrong string is what "
            "downstream copy cannot recover from."
        )
    )


class MaterialNote(ModelFacing):
    """A judgement about one asset that only watching it can produce."""

    source_id: str
    usable: bool = Field(
        description=(
            "False only for footage that cannot serve any purpose -- a failed "
            "take, a duplicate attempt superseded by a better one. Camera "
            "shake, soft focus, and unusual framing are style choices "
            "available to the edit, not defects; say so in `note` and leave "
            "this true."
        )
    )
    note: str = Field(
        description="What is here and what it is good for, in one or two lines."
    )
    supersedes: list[str] = Field(
        default_factory=list,
        description=(
            "Other source_ids this take beats, when several attempts at the "
            "same action exist."
        ),
    )


class StyleDecision(ModelFacing):
    """Stage-one creative direction, formed after seeing everything.

    This replaces the fixed profile the previous system carried. A profile
    tuned on one shoot silently mis-fits the next; a decision made against the
    actual material can be read, argued with, and checked against the cut.
    """

    reasoning: str = Field(
        description=(
            "Why this material wants this treatment. Written before the "
            "decisions below, because a conclusion reached first and "
            "explained afterwards tends to be the conventional one."
        )
    )
    material_assessment: str = Field(
        description="What this footage is: kind, light, mood, what it affords."
    )
    material_notes: list[MaterialNote] = Field(default_factory=list)
    direction: str = Field(
        description="Overall tone, cutting rhythm, and grade direction."
    )
    target_seconds: float = Field(gt=0.0)
    aspect: Literal["16:9", "9:16", "1:1", "other"]
    aspect_other: str | None = Field(
        default=None,
        description="Required when aspect is 'other'; e.g. '4:5'.",
    )
    music_suggestion: str | None = Field(
        default=None,
        description=(
            "When no track was supplied: what music would suit this cut. The "
            "rhythm strategy in `direction` still has to stand on its own."
        ),
    )

    @model_validator(mode="after")
    def describe_other_aspect(self) -> "StyleDecision":
        if self.aspect == "other" and not (self.aspect_other or "").strip():
            raise ValueError("aspect 'other' needs aspect_other spelled out")
        return self


class Subject(ModelFacing):
    """Who or what the frame is about, in words a detector can be pointed at."""

    description: str = Field(
        description=(
            "The subject as a viewer would name it, distinguishing it from "
            "anything similar in the same frame. Two handsets on a table need "
            "'the left, grey one', not 'the handset' -- an ambiguous subject "
            "is the single most common reason a shot cannot be reframed."
        )
    )
    min_visible: Normalised = Field(
        default=0.85,
        description=(
            "Smallest share of the subject that still carries the meaning. "
            "This is the normal way to say a subject matters, because a "
            "fraction leaves a crop planable. Reserve 1.0 for content partial "
            "clipping destroys outright -- rendered text, a UI state, a "
            "readout -- since a subject demanding all of itself can only be "
            "honoured or abandoned."
        ),
    )


class Look(ModelFacing):
    """One thing the frame settles on, and how long it stays there.

    The primitive the camera vocabulary is built from, rather than a menu of
    named moves. A shot is an ordered list of these: one look is a hold, two
    are a move from the first to the second, three are a move that stops on
    the way. What the moves used to be named is now what a list of looks
    happens to describe, and the executor labels it afterwards.

    That matters because the named menu could express nonsense. `pan` with
    one static subject was a legal combination and an impossible instruction
    -- a move and its targets were separate fields, so they could disagree.
    A list of looks cannot disagree with itself: a look is somewhere the
    frame goes.

    Nothing here is geometry. The model says what to look at and how long to
    look; local code measures where that is and how fast to travel. Handing
    it coordinates has been tried three times in this project -- nine-box
    positions, timestamps, 0..1000 boxes -- and it has invented plausible
    answers every time.
    """

    at: str = Field(
        description=(
            "What the frame settles on, named so a detector can be pointed "
            "at it and told apart from anything similar beside it. Two "
            "handsets on a table need 'the left, grey one', not 'the "
            "handset'. Use the same wording twice to look at one thing "
            "again at a different framing -- that is what a push in is."
        )
    )
    seconds: float = Field(
        default=0.0,
        description=(
            "How long the frame rests here before moving on. Zero lets local "
            "code choose a floor that reads as a stop at all. This is the "
            "judgement worth making: a two-word logo needs less than a row "
            "of three handsets, and only someone who has watched the shot "
            "knows which this is."
        ),
    )
    framing: str = Field(
        default="thirds",
        description=(
            "Where this subject sits, and how tightly. `thirds` and `centre` "
            "place it in the space available; `fill` closes in until it "
            "carries the frame. Two looks at the same subject, the first "
            "`thirds` and the second `fill`, is a push in."
        ),
    )
    must_be_whole: bool = Field(
        default=False,
        description=(
            "True only when partial clipping destroys the meaning -- rendered "
            "text, a UI state, a readout. A subject wider than any crop of "
            "its source cannot be whole and still, so this and a single look "
            "is a contradiction local code will report rather than resolve."
        ),
    )


class Reframe(ModelFacing):
    """What the camera should attend to, never where to put the crop."""

    # Where each look was measured to be, when the card knew. Centre across,
    # centre down, and the crop width its framing asks for -- enough to work
    # out how far the frame has to travel, and so how long this shot needs,
    # before anything is grounded. Empty when the card could not place them.
    look_boxes: list[tuple[float, float, float]] = Field(default_factory=list)
    looks: list[Look] = Field(
        default_factory=list,
        description=(
            "Where the frame goes, in order. One look holds on it; two move "
            "from the first to the second; more stop on the way. Empty means "
            "no subject at all -- a landscape or an empty frame -- and then "
            "`pan_hint` says which way to drift."
        ),
    )
    subject: Subject | None = Field(
        default=None,
        description=(
            "The first look, kept so anything reading a single subject still "
            "works. Derived from `looks`; not filled by the planner."
        ),
    )
    pan_hint: Literal["left", "right", "up", "down", "none"] = Field(
        default="none",
        description="Drift direction for an empty frame with no subject.",
    )
    intent: str = Field(
        default="",
        description="The move in words: hold, follow, ease in on the hands.",
    )
    then_subject: Subject | None = Field(
        default=None,
        description=(
            "A second subject this shot hands over to. The pipeline splits "
            "the shot in two rather than changing target mid-move, so both "
            "halves get a frame that commits to one thing."
        ),
    )
    framing: str = Field(
        default="thirds",
        description=(
            "Where the subject sits when it does not fill the output ratio. "
            "Negative space is a composition rather than a shortfall, so this "
            "asks where the subject goes in that space -- not how to remove "
            "it. Local code holds one sharpness ceiling regardless."
        ),
    )
    camera_move: str = Field(
        default="hold",
        description=(
            "The move the planner chose from the declared capability list. "
            "The executor dispatches on this, so a name outside the list is "
            "carried out as a hold and recorded as unsupported rather than "
            "silently ignored."
        ),
    )
    # What the plan said the frame would do, kept so the executor can tell a
    # substitution from an instruction carried out. One look means "stay on
    # this", which for something that walks is a follow and for something
    # standing still is a held frame -- both are the plan happening, and
    # without this every settled shot was recorded as a downgraded follow.
    planned_to_move: bool = False
    # What the source camera was already doing. `authored` and
    # `subject_follow` mean the take moves on its own, which is the case the
    # selection prompt calls out -- two moves stacked fight each other, and
    # the real one wins because it is real. Nothing downstream held this, so
    # the warning was advice with no way to check it.
    source_motion_role: str = "locked"
    source_motion_description: str = ""
    pacing_exception: bool = False
    pacing_exception_reason: str = ""
    camera_energy: CameraEnergy = Field(
        default="calm",
        description=(
            "How lively the move is. This picks a parameter set locally; it "
            "is not a speed. Say what the shot wants and let the executor "
            "size it against the measured motion."
        ),
    )

    @model_validator(mode="after")
    def empty_frames_need_a_direction(self) -> "Reframe":
        # `looks` counts as saying what to attend to. This predates them and
        # only knew about `subject`, so a reframe built from looks alone --
        # which is the shape everything is moving to -- failed validation for
        # having no subject, while carrying three.
        if (
            not self.looks
            and self.subject is None
            and self.pan_hint == "none"
            and not self.intent
        ):
            raise ValueError(
                "a subjectless reframe needs a pan_hint or a stated intent"
            )
        return self


class MusicSync(ModelFacing):
    """How this shot sits against the music.

    Timing resolves locally against a measured grid, so nothing here is a
    timestamp. What is here is the editorial part: whether this cut wants to
    be felt with the music or against it, and why.

    Cutting everything to a fixed beat count is the failure mode on the other
    side of ignoring the music entirely. A shot whose action needs another
    half-second should get it and land late; a hard accent should be hit even
    if the shot before it has to give way. Say which of those this is.
    """

    cut_on_beat: bool = Field(
        default=True,
        description=(
            "True to land the out-point on the nearest musical event. False "
            "when the content should govern -- an action mid-completion, a "
            "reaction still landing, a held beat of silence. Cutting through "
            "a finished gesture to hit an accent is worse than arriving a "
            "moment late."
        ),
    )
    # A moment inside the shot that should land on the music, rather than
    # the cut. Only the out-point was ever aligned, so "the phone snaps shut
    # on the downbeat and the shot runs on" could not be said -- and a model
    # choosing a length so an action completes at the cut had that alignment
    # taken away again by the snap, which moves the out-point by up to a beat.
    anchor: str | None = None
    anchor_lands_on: str = "downbeat"
    anchor_relation: str = "on"
    sync_to: str | None = Field(
        default=None,
        description=(
            "A named point this shot should coincide with, when it matters: "
            "'chorus_1_start', 'drop', 'outro_start'. Leave null when no "
            "particular moment in the track is being targeted."
        ),
    )
    beats: int | None = Field(
        default=None,
        ge=1,
        description=(
            "Length in beats, only when the shot genuinely wants a musical "
            "count -- a montage pulse, a repeated figure. Leave null and let "
            "the clip's own in and out times express the length whenever the "
            "content, not the metre, decides how long this is on screen."
        ),
    )
    rhythm_reason: str = Field(
        default="",
        description=(
            "Why this shot takes the length it does, in terms of what is "
            "happening on screen and in the track. One line. This is the "
            "record of an editorial decision, so 'eight beats' is not a "
            "reason -- 'the fold completes on the downbeat' is."
        ),
    )


class Clip(ModelFacing):
    """One shot. Times are approximate; grounding snaps them."""

    clip_id: str
    source_id: str
    approx_in_seconds: float = Field(ge=0.0)
    approx_out_seconds: float = Field(gt=0.0)
    in_looks_like: str = Field(
        default="",
        description=(
            "What is on screen at the in-point. Grounding uses this to snap "
            "to the right boundary when the timestamp lands near several."
        ),
    )
    out_looks_like: str = Field(default="")
    music_sync: MusicSync = Field(default_factory=MusicSync)
    energy_intent: EnergyIntent = Field(
        default="medium",
        description=(
            "How energetic this beat of the film is, so a high-energy stretch "
            "can be checked for actually holding motion."
        ),
    )
    audio_role: AudioRole = Field(
        default="auto",
        description=(
            "What this shot contributes to the soundtrack. 'discard' removes "
            "its source audio; 'narrative' carries spoken meaning; "
            "'sync_action' preserves a sound tied to a visible event; "
            "'ambient_texture' preserves deliberate atmosphere. 'auto' is "
            "legacy-only and follows the render's global voice setting."
        ),
    )
    audio_completion: AudioCompletion = Field(
        default="none",
        description=(
            "The audible obligation this edit must finish. It is independent "
            "of what picture is shown, which is what later permits B-roll "
            "over a complete answer."
        ),
    )
    picture_role: PictureRole = Field(
        default="primary_action",
        description="Why this source is visible during this editorial beat.",
    )
    coverage_claim_seconds: float | None = Field(
        default=None,
        ge=0.0,
        description=(
            "Locally measured visual evidence available to this shot. The "
            "planner may name a role but may not author this duration."
        ),
    )
    reframe: Reframe | None = Field(
        default=None,
        description="Required when the output aspect differs from the source.",
    )
    named_facts: list[str] = Field(
        default_factory=list,
        description=(
            "Strings quoted verbatim from the transcription list. Anything "
            "here that was never transcribed is flagged, not silently trusted."
        ),
    )
    sync_group: str | None = Field(
        default=None,
        description=(
            "Reserved. Groups simultaneous angles of one moment once audio "
            "fingerprint alignment exists; unused today."
        ),
    )
    # The stretch of this source that was judged worth cutting into, carried
    # with the clip rather than left on the card.
    #
    # The card has always said where a take becomes usable and where it stops
    # being usable, and until now that answer reached exactly two places: a
    # line of prompt text, and the dataclass that builds that line. Nothing
    # downstream could see it, so it was advice. The in-point could be
    # snapped onto an action outside it, the length could be stretched past
    # it to reach a beat, and the executor's only question was whether the
    # time existed in the file.
    #
    # Zero for either end means nobody said, and nothing is enforced.
    # Where each nameable moment in this take sits, by the id the card gave
    # it. Carried because grounding is what puts one on a beat, and the card
    # is read three layers earlier.
    moments: dict[str, float] = Field(default_factory=dict)
    usable_from_seconds: float = Field(default=0.0, ge=0.0)
    usable_to_seconds: float = Field(default=0.0, ge=0.0)

    @property
    def usable_window(self) -> "tuple[float, float] | None":
        """The bounds a cut of this clip may not leave, if they are known."""

        if self.usable_to_seconds <= self.usable_from_seconds:
            return None
        return (self.usable_from_seconds, self.usable_to_seconds)

    @model_validator(mode="after")
    def out_follows_in(self) -> "Clip":
        if self.approx_out_seconds <= self.approx_in_seconds:
            raise ValueError(
                f"clip {self.clip_id}: out must fall after in"
            )
        return self


class Goal(ModelFacing):
    """Something the audience should feel or understand.

    Satisfied by the film as a whole. The previous system read a brief as a
    shot list and went hunting for one picture per line, which is how a brief
    asking to convey craftsmanship turns into four redundant process shots
    instead of one good macro.
    """

    goal_id: str
    text: str
    covered_by: list[str] = Field(
        default_factory=list, description="clip_ids contributing. May overlap."
    )
    coverage_type: CoverageType = "implied"
    uncovered_reason: str | None = Field(
        default=None,
        description=(
            "Why the cut does not serve this goal. A stated, argued omission "
            "is a legitimate outcome, not a failure."
        ),
    )

    @model_validator(mode="after")
    def account_for_the_gap(self) -> "Goal":
        if not self.covered_by and not (self.uncovered_reason or "").strip():
            raise ValueError(
                f"goal {self.goal_id}: needs contributing clips or a reason"
            )
        return self


class HardConstraint(ModelFacing):
    """Something that must literally appear. Checked off one by one."""

    constraint_id: str
    text: str
    satisfied_by: list[str] = Field(default_factory=list)


class BriefCoverage(ModelFacing):
    hard_constraints: list[HardConstraint] = Field(default_factory=list)
    goals: list[Goal] = Field(default_factory=list)


class AudioClip(ModelFacing):
    """One continuous source-audio assignment on the edit timeline."""

    audio_id: str
    source_id: str
    in_seconds: float = Field(ge=0.0)
    out_seconds: float = Field(gt=0.0)
    starts_at_clip_id: str
    offset_seconds: float = 0.0
    role: Literal["narrative", "sync_action", "ambient_texture"]
    completion: AudioCompletion = "none"
    gain_db: float = 0.0
    why: str = ""

    @model_validator(mode="after")
    def source_window_is_forward(self) -> "AudioClip":
        if self.out_seconds <= self.in_seconds:
            raise ValueError("audio assignment out must follow in")
        if self.role == "narrative" and self.completion not in {
            "complete_thought", "intentional_cut"
        }:
            raise ValueError("narrative audio must complete a thought or cut intentionally")
        return self


class EDL(ModelFacing):
    """The edit decision list: the whole plan, from any source.

    The executor cannot tell whether this came from the planner or from a text
    editor, and that is the point -- a hand-written EDL is how an execution
    bug gets told apart from a planning one.
    """

    edl_version: Literal["montagewright-edl-v1"] = "montagewright-edl-v1"
    project_id: str
    # Where in the track this film sits. A thirty-second cut of a two-minute
    # piece used to always take the first thirty seconds, which is the intro
    # -- written to have no energy yet, so the picture carried the film on
    # its own. Decided by the rhythm pass, which can hear the track.
    music_from_seconds: float = 0.0
    # Pieces of the track played in order, when one stretch will not carry
    # the film. Every edge gets moved onto a phrase line before it is used.
    music_spans: list[tuple[float, float]] = Field(default_factory=list)
    style_decision: StyleDecision | None = Field(
        default=None,
        description=(
            "Carried so review can judge the cut against its own stated "
            "intent. Absent for a hand-written EDL."
        ),
    )
    clips: list[Clip] = Field(min_length=1)
    audio_clips: list[AudioClip] = Field(default_factory=list)
    brief_coverage: BriefCoverage = Field(default_factory=BriefCoverage)
    named_facts: list[NamedFact] = Field(default_factory=list)

    @model_validator(mode="after")
    def clip_ids_are_unique(self) -> "EDL":
        ids = [clip.clip_id for clip in self.clips]
        duplicates = {value for value in ids if ids.count(value) > 1}
        if duplicates:
            raise ValueError(f"duplicate clip_ids: {sorted(duplicates)}")
        known = set(ids)
        missing = {
            audio.starts_at_clip_id for audio in self.audio_clips
            if audio.starts_at_clip_id not in known
        }
        if missing:
            raise ValueError(f"audio assignments name unknown clip_ids: {sorted(missing)}")
        return self

    @model_validator(mode="after")
    def coverage_points_at_real_clips(self) -> "EDL":
        known = {clip.clip_id for clip in self.clips}
        for goal in self.brief_coverage.goals:
            unknown = set(goal.covered_by) - known
            if unknown:
                raise ValueError(
                    f"goal {goal.goal_id} cites unknown clips: {sorted(unknown)}"
                )
        for constraint in self.brief_coverage.hard_constraints:
            unknown = set(constraint.satisfied_by) - known
            if unknown:
                raise ValueError(
                    f"constraint {constraint.constraint_id} cites unknown "
                    f"clips: {sorted(unknown)}"
                )
        return self


class DegradationStep(Local):
    """One rung down from the plan, with the measurement that forced it.

    Degrading is not free. Each step needs its own evidence that the rung
    above was tried and failed, because a system allowed to skip straight to
    the safe option will take it every time -- which is the conservative bias
    this pipeline exists to remove.
    """

    clip_id: str
    ladder: Literal[
        "full_move", "reduced_zoom", "slower_follow", "static_on_subject",
        "center_crop", "other",
    ]
    ladder_other: str | None = None
    trigger: str = Field(description="What was attempted and how it failed.")
    measured: dict[str, float] = Field(
        default_factory=dict,
        description="The numbers behind the claim, so it can be argued with.",
    )
    adjudication: Literal["accept", "replan", "unadjudicated"] = "unadjudicated"
    adjudication_reason: str | None = None


class Issue(ModelFacing):
    """One actionable note from review."""

    issue_type: Literal[
        "pacing", "framing", "music_sync", "coverage", "named_fact",
        "continuity", "audio_content", "other",
    ]
    issue_type_other: str | None = None
    severity: Literal["minor", "major", "blocking"]
    clip_id: str | None = None
    at_seconds: float | None = Field(
        default=None,
        description="Roughly where, in the cut. Grounding snaps it; a rough "
        "position beats no position.",
    )
    description: str
    fix: str = Field(
        description=(
            "What to change. A note nobody can act on does not start another "
            "round, so 'this could feel better' belongs in `description`."
        )
    )


class ReviewVerdict(ModelFacing):
    """The reviewer's call. It may approve; that is how the loop converges."""

    verdict: Literal["approve", "revise"]
    overall: str
    issues: list[Issue] = Field(default_factory=list)

    @model_validator(mode="after")
    def revision_needs_something_to_do(self) -> "ReviewVerdict":
        if self.verdict == "revise" and not any(
            issue.severity in {"major", "blocking"} for issue in self.issues
        ):
            raise ValueError(
                "revise needs at least one major or blocking issue; minor "
                "notes alone do not justify another render"
            )
        return self


def looks_of(shot: dict) -> "list[Look]":
    """The looks in a selection shot, whichever shape it was written in.

    The one place that knows both. Taking `camera_move` out of the schema
    left six readers still asking for it or for `subject`: two raised, and
    four returned an empty string or "hold" and carried on. The run that
    found this had already paid for direction and selection.

    A lesson written an hour earlier and then not applied -- when a field
    goes, the thing to search for is not who still writes it but who still
    reads it. So there is one reader now, and everything else asks it.
    """

    looks = [
        Look(
            at=str(one.get("at", "")),
            seconds=float(one.get("seconds", 0.0) or 0.0),
            framing=str(one.get("framing", "thirds") or "thirds"),
            must_be_whole=bool(one.get("must_be_whole", False)),
        )
        for one in (shot.get("looks") or [])
        if str(one.get("at", "")).strip()
    ]
    if looks:
        return looks

    named = str(shot.get("subject", "") or "").strip()
    if not named:
        return []
    framing = str(shot.get("framing", "thirds") or "thirds")
    move = str(shot.get("camera_move", "hold") or "hold")
    whole = bool(shot.get("must_be_whole"))
    first = Look(
        at=named,
        framing="thirds" if move == "push_in" else framing,
        must_be_whole=whole,
    )
    if shot.get("then_subject"):
        return [first, Look(at=str(shot["then_subject"]), framing=framing)]
    if move in {"push_in", "pull_out"}:
        tight = Look(at=named, framing="fill", must_be_whole=whole)
        return [tight, first] if move == "pull_out" else [first, tight]
    return [first]


def subject_of(shot: dict) -> str:
    """What this shot is about, for anything that wants one name."""

    looks = looks_of(shot)
    return looks[0].at if looks else ""


def must_be_whole_of(shot: dict) -> bool:
    """Whether anything this shot looks at has to be entire."""

    return any(one.must_be_whole for one in looks_of(shot))


def move_of_shot(shot: dict) -> str:
    """The physical label shown to people, including explicit follows.

    One look cannot geometrically reveal a follow, so reading only `looks`
    made the planner, replan log and browser call an explicit follow a hold.
    The intent owns that distinction; multi-point paths are still measured
    from their looks.
    """

    intent = camera_intent_of(shot)
    explicit = {
        "hold": "hold",
        "use_source_motion": "source_motion",
        "follow_subject": "follow_subject",
        "push_in": "push_in",
        "pull_out": "pull_out",
    }.get(intent)
    if explicit is not None:
        return explicit
    looks = looks_of(shot)
    if looks:
        return move_of(looks)
    return str(shot.get("camera_move", "hold") or "hold")


def camera_intent_of(shot: dict) -> str:
    """The editorial motion choice, including legacy plans.

    Looks say where the crop goes; this says why.  Keeping the reader here
    lets old cached selections survive while new ones distinguish a fixed
    crop over authored source motion from a genuinely static shot.
    """

    named = str(shot.get("camera_intent", "") or "").strip()
    if named:
        return named
    legacy_move = str(shot.get("camera_move", "") or "").strip()
    if legacy_move in {"follow_subject", "push_in", "pull_out"}:
        return legacy_move
    if legacy_move and legacy_move != "hold":
        return "reveal"
    looks = looks_of(shot)
    legacy_frame = str(shot.get("frame", ""))
    if legacy_frame == "travels" and len(looks) == 1:
        return "follow_subject"
    if legacy_frame == "settles" or len(looks) < 2:
        return "hold"
    move = move_of(looks)
    return move if move in {"push_in", "pull_out"} else (
        "multi_stop" if len(looks) >= 3 else "reveal"
    )


def move_of(looks: "list[Look]") -> str:
    """What a list of looks turns out to be, in the old vocabulary.

    The names are no longer chosen; they are observed. That is the whole
    point of the change -- a move and its targets cannot disagree when one is
    read off the other -- but the names still have to exist, because the
    report, the interface and the reviewer all speak them.

    The axis is guessed rather than measured here, since nothing at this
    point knows where either subject is. The executor measures both centres
    and moves along whichever axis they actually differ on; this is only the
    label, and it is corrected once the boxes are known.
    """

    if len(looks) < 2:
        return "hold"
    tightness = {"fill": 2, "thirds": 1, "centre": 1}
    if all(one.at == looks[0].at for one in looks):
        first = tightness.get(looks[0].framing, 1)
        last = tightness.get(looks[-1].framing, 1)
        if last > first:
            return "push_in"
        if last < first:
            return "pull_out"
        return "hold"
    return "pan"


# Two vocabularies met here and neither knew about the other. A shot says its
# energy is low, medium or high; the executor's speed table is keyed calm,
# active and dynamic. Nothing joined them, so every move ran at one hard-coded
# speed and the label a shot chose for itself was decoration.
#
# Not a coincidence of names: how lively a passage is and how fast a frame
# may travel are the same judgement seen from two sides, which is why the
# planner is asked for one of them and not both.
LOOK_ENERGIES: dict[str, CameraEnergy] = {
    "low": "calm",
    "medium": "active",
    "high": "dynamic",
}

# What travel is priced at when nothing has said otherwise -- quoted to the
# planner in the material listing, so the number it budgets against and the
# number the executor runs at are read from one place.
LOOK_ENERGY: CameraEnergy = "active"


def look_energy(energy: object) -> CameraEnergy:
    """The camera speed a shot's energy asks for."""

    return LOOK_ENERGIES.get(str(energy or ""), LOOK_ENERGY)


def reframe_of(shot: dict) -> Reframe:
    """The reframe a shot asked for, built the one way.

    A pan between two subjects in one frame is the only move that needs the
    second one, and the rebuild left it out -- so the handoff branch was
    never reached, the move fell through to a follow with nothing to follow,
    and a pan measured at 0.278 -> 0.696 came back as a held centre crop.
    """

    looks = looks_of(shot)

    first = looks[0] if looks else None
    editorial_intent = camera_intent_of(shot)
    inferred_move = move_of(looks) if looks else str(
        shot.get("camera_move", "hold") or "hold"
    )
    camera_move = {
        "hold": "hold",
        "use_source_motion": "hold",
        "follow_subject": "follow_subject",
        "push_in": "push_in",
        "pull_out": "pull_out",
    }.get(editorial_intent, inferred_move)
    return Reframe(
        looks=looks,
        subject=(
            Subject(
                description=first.at,
                min_visible=1.0 if first.must_be_whole else 0.85,
            )
            if first is not None
            else None
        ),
        then_subject=(
            Subject(description=looks[1].at) if len(looks) > 1 else None
        ),
        intent=shot.get("why", "")[:120],
        # Read off the looks rather than taken from the plan. A plan written
        # before looks existed still carries the old field, and it is only
        # trusted when the looks cannot say -- otherwise the two could
        # disagree, which is the whole thing this replaced.
        camera_move=camera_move,
        framing=(looks[0].framing if looks else
                 str(shot.get("framing", "thirds") or "thirds")),
        camera_energy=look_energy(shot.get("energy")),
        planned_to_move=editorial_intent not in {"hold", "use_source_motion"},
        source_motion_role=str(shot.get("source_motion_role", "locked") or "locked"),
        source_motion_description=str(
            shot.get("source_motion_description", "") or ""
        )[:240],
        pacing_exception=bool(shot.get("pacing_exception", False)),
        pacing_exception_reason=str(
            shot.get("pacing_exception_reason", "") or ""
        )[:200],
    )
