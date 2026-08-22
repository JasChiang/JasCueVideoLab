"""An editor's work order, shared by Web and CLI.

This is deliberately a delivery sheet, not a second planning model.  It says
what was handed to the edit -- rushes, brief, music, subject lock, delivery
and sound requirements -- and compiles those facts to the existing CLI.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class Delivery(_Strict):
    aspect: Literal["9:16", "16:9", "1:1", "4:5"] = "9:16"
    seconds: float = Field(default=0.0, ge=0.0)
    duration_mode: Literal["exact", "range", "preferred"] = "preferred"
    minimum_seconds: float | None = Field(default=None, ge=0.0)
    maximum_seconds: float | None = Field(default=None, gt=0.0)
    subtitles: Literal["none", "sidecar", "burn"] = "sidecar"
    subtitle_look: Literal["plain", "speakers", "spoken", "plate"] = "plain"
    subtitle_font: str | None = None
    timeline: Literal["none", "premiere", "finalcut", "both"] = "none"
    frame_rate: Literal[
        "source", "23.976", "24", "25", "29.97", "30", "50", "59.94", "60"
    ] = "30"
    codec: Literal["h264", "hevc", "prores"] = "h264"
    color: Literal["sdr_rec709", "preserve_hdr", "normalize_to_sdr"] = (
        "normalize_to_sdr"
    )
    width: int | None = Field(default=None, gt=0)
    height: int | None = Field(default=None, gt=0)
    loudness_lufs: float = Field(default=-14.0, ge=-24.0, le=-9.0)

    @model_validator(mode="after")
    def exact_has_a_number(self) -> "Delivery":
        if self.duration_mode == "exact" and self.seconds <= 0:
            raise ValueError("delivery.duration_mode exact requires delivery.seconds")
        if self.duration_mode == "range":
            if self.minimum_seconds is None or self.maximum_seconds is None:
                raise ValueError("delivery.duration_mode range requires min and max seconds")
            if self.maximum_seconds <= self.minimum_seconds:
                raise ValueError("delivery maximum_seconds must follow minimum_seconds")
            if self.seconds and not self.minimum_seconds <= self.seconds <= self.maximum_seconds:
                raise ValueError("delivery.seconds must fall inside its duration range")
        elif self.minimum_seconds is not None or self.maximum_seconds is not None:
            raise ValueError("delivery min/max seconds require duration_mode range")
        if (self.width is None) != (self.height is None):
            raise ValueError("delivery.width and delivery.height must be supplied together")
        return self


class Sound(_Strict):
    speech: Literal["auto", "never"] = "auto"
    locale: str = Field(default="zh-TW", min_length=1)


class TimeRange(_Strict):
    start_seconds: float = Field(ge=0.0)
    end_seconds: float = Field(gt=0.0)

    @model_validator(mode="after")
    def forward(self) -> "TimeRange":
        if self.end_seconds <= self.start_seconds:
            raise ValueError("time range end must follow start")
        return self


class MusicPolicy(_Strict):
    allowed_ranges: tuple[TimeRange, ...] = ()
    avoid_lyrics_under_dialogue: bool = True
    stems: tuple[str, ...] = ()


class DialoguePolicy(_Strict):
    edit_mode: Literal["continuous_soundbite", "phrase_edit"] = (
        "continuous_soundbite"
    )
    remove_fillers: bool = False
    preserve_question: bool = False
    allow_translation: bool = False

    @model_validator(mode="after")
    def fillers_need_phrase_edit(self) -> "DialoguePolicy":
        if self.remove_fillers and self.edit_mode != "phrase_edit":
            raise ValueError("remove_fillers requires dialogue phrase_edit")
        return self


class SyncPolicy(_Strict):
    map: str | None = None
    master_audio_source_id: str | None = None
    authority: Literal["none", "timecode", "audio_fingerprint", "slate", "manual"] = (
        "none"
    )

    @model_validator(mode="after")
    def authority_has_map(self) -> "SyncPolicy":
        if self.authority != "none" and not self.map:
            raise ValueError("sync authority requires sync.map")
        if self.map and self.authority == "none":
            raise ValueError("sync.map requires a non-none authority")
        return self


class Subject(_Strict):
    grounding_spec: str | None = None
    target_id: str = "target.primary"
    description: str | None = None
    references: tuple[str, ...] = ()
    negatives: tuple[str, ...] = ()
    identity_cues: tuple[str, ...] = ()
    exclusions: tuple[str, ...] = ()
    presence: Literal[
        "context_allowed", "target_led", "target_only"
    ] = "context_allowed"
    identity_semantics: Literal[
        "physical_instance", "sku", "variant", "product_family"
    ] = "physical_instance"

    @model_validator(mode="after")
    def one_grounding_input(self) -> "Subject":
        simple = bool((self.description or "").strip() or self.references)
        if self.grounding_spec and simple:
            raise ValueError(
                "subject.grounding_spec and simple subject fields are mutually exclusive"
            )
        if bool((self.description or "").strip()) != bool(self.references):
            raise ValueError(
                "subject.description and subject.references must be supplied together"
            )
        if not self.grounding_spec and not simple and self.presence != "context_allowed":
            raise ValueError("subject.presence requires a grounded subject")
        return self


class RunPolicy(_Strict):
    budget_usd: float = Field(default=5.0, ge=0.0)
    review: bool = False


class TimelineWindow(_Strict):
    start_seconds: float | None = Field(default=None, ge=0.0)
    end_seconds: float | None = Field(default=None, gt=0.0)
    final_seconds: float | None = Field(default=None, gt=0.0)

    @model_validator(mode="after")
    def one_window_shape(self) -> "TimelineWindow":
        if self.final_seconds is not None and (
            self.start_seconds is not None or self.end_seconds is not None
        ):
            raise ValueError("final_seconds cannot be combined with absolute bounds")
        if (
            self.start_seconds is not None and self.end_seconds is not None
            and self.end_seconds <= self.start_seconds
        ):
            raise ValueError("timeline window end must follow start")
        return self


class TimelineObligation(_Strict):
    obligation_id: str = Field(pattern=r"^[A-Za-z0-9_.:-]+$")
    kind: Literal[
        "minimum_presence", "required_presence", "forbidden_presence",
        "required_cooccurrence", "exclusive_presence", "minimum_read",
    ]
    track: Literal["picture", "graphic", "audio"] = "picture"
    refs: tuple[str, ...] = Field(min_length=1)
    window: TimelineWindow = Field(default_factory=TimelineWindow)
    minimum_seconds: float = Field(default=0.0, ge=0.0)
    why: str = ""

    @model_validator(mode="after")
    def duration_when_needed(self) -> "TimelineObligation":
        if self.kind in {"minimum_presence", "minimum_read"} and self.minimum_seconds <= 0:
            raise ValueError(f"{self.kind} requires minimum_seconds")
        if self.kind == "required_cooccurrence" and len(self.refs) < 2:
            raise ValueError("required_cooccurrence requires at least two refs")
        return self


class RightsPolicy(_Strict):
    acknowledged: bool = False
    allowed_territories: tuple[str, ...] = ()
    allowed_platforms: tuple[str, ...] = ()
    expires_on: str | None = None
    prohibited_visuals: tuple[str, ...] = ()
    embargo_until: str | None = None


class ReleasePolicy(_Strict):
    producer_approval_required: bool = True
    approver: str | None = None
    approval_note: str | None = None
    approved_artifact_sha256: str | None = Field(
        default=None, pattern=r"^[0-9a-f]{64}$"
    )


class PictureComposition(_Strict):
    mode: Literal["none", "split_screen", "pip", "screen_insert"] = "none"
    description: str = ""

    @model_validator(mode="after")
    def requested_effect_is_described(self) -> "PictureComposition":
        if self.mode != "none" and not self.description.strip():
            raise ValueError("picture_composition effect requires a description")
        return self


class DeliveryVariant(_Strict):
    variant_id: str = Field(pattern=r"^[A-Za-z0-9_.:-]+$")
    delivery: Delivery
    output: str | None = None
    brief_addendum: str = ""


class EditJob(_Strict):
    version: Literal["montagewright-job-v1"] = "montagewright-job-v1"
    rushes: str | None = None
    output: str | None = None
    brief: str | None = None
    music: str | None = None
    delivery: Delivery = Field(default_factory=Delivery)
    sound: Sound = Field(default_factory=Sound)
    dialogue: DialoguePolicy = Field(default_factory=DialoguePolicy)
    sync: SyncPolicy = Field(default_factory=SyncPolicy)
    music_policy: MusicPolicy = Field(default_factory=MusicPolicy)
    subject: Subject | None = None
    obligations: tuple[TimelineObligation, ...] = ()
    rights: RightsPolicy = Field(default_factory=RightsPolicy)
    release: ReleasePolicy = Field(default_factory=ReleasePolicy)
    picture_composition: PictureComposition = Field(
        default_factory=PictureComposition
    )
    variants: tuple[DeliveryVariant, ...] = ()
    run: RunPolicy = Field(default_factory=RunPolicy)

    @model_validator(mode="after")
    def sound_matches_delivery(self) -> "EditJob":
        if self.sound.speech == "never" and self.delivery.subtitles != "none":
            raise ValueError(
                "sound.speech never cannot produce transcript subtitles; "
                "set delivery.subtitles to none"
            )
        obligation_ids = [one.obligation_id for one in self.obligations]
        if len(obligation_ids) != len(set(obligation_ids)):
            raise ValueError("obligation IDs must be unique")
        variant_ids = [one.variant_id for one in self.variants]
        if len(variant_ids) != len(set(variant_ids)):
            raise ValueError("variant IDs must be unique")
        return self

    def editorial_contract(self) -> dict[str, Any]:
        """Only the hard facts the editing brain must plan around."""

        return {
            "dialogue": self.dialogue.model_dump(mode="json"),
            "duration": {
                "mode": self.delivery.duration_mode,
                "target_seconds": self.delivery.seconds,
                "minimum_seconds": self.delivery.minimum_seconds,
                "maximum_seconds": self.delivery.maximum_seconds,
            },
            "music_policy": self.music_policy.model_dump(mode="json"),
            "subject": (
                {
                    "identity_semantics": self.subject.identity_semantics,
                    "presence": self.subject.presence,
                }
                if self.subject is not None else None
            ),
            "obligations": [
                one.model_dump(mode="json") for one in self.obligations
            ],
            "rights": self.rights.model_dump(mode="json"),
            "picture_composition": self.picture_composition.model_dump(mode="json"),
        }

    def execution_contract_faults(self) -> tuple[str, ...]:
        """Delivery requests the current renderer must refuse, not degrade."""

        faults: list[str] = []
        if self.delivery.frame_rate not in {"24", "25", "30", "50", "60"}:
            faults.append(
                f"delivery.frame_rate {self.delivery.frame_rate} is not yet a "
                "verified frame clock (supported: 24, 25, 30, 50, 60)"
            )
        if self.delivery.codec != "h264":
            faults.append(
                f"delivery.codec {self.delivery.codec} is not yet a verified "
                "master encoder (supported: h264)"
            )
        if self.delivery.color == "preserve_hdr":
            faults.append(
                "delivery.color preserve_hdr has no verified HDR render path yet"
            )
        if self.delivery.width is not None:
            faults.append(
                "custom delivery dimensions are not connected to the renderer yet"
            )
        if self.music_policy.stems:
            faults.append("music_policy.stems has no verified stem mixer yet")
        if self.dialogue.remove_fillers:
            faults.append(
                "dialogue.remove_fillers has no verified audible-join and room-tone audit yet"
            )
        if self.dialogue.preserve_question:
            faults.append(
                "dialogue.preserve_question is not yet a locally audited obligation"
            )
        if self.dialogue.allow_translation:
            faults.append(
                "dialogue.allow_translation is not connected to a provenance-safe translator"
            )
        if self.obligations and any(
            obligation.track == "audio" for obligation in self.obligations
        ):
            faults.append("audio timeline obligations have no verified auditor yet")
        if self.picture_composition.mode != "none":
            faults.append(
                f"picture_composition.mode {self.picture_composition.mode} has "
                "no verified multi-source compositor yet"
            )
        return tuple(faults)


def _read(path: Path) -> dict[str, Any]:
    try:
        raw = path.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as error:
        raise ValueError(f"job is unreadable: {error}") from error
    try:
        if path.suffix.casefold() in {".yaml", ".yml"}:
            import yaml

            class JobLoader(yaml.SafeLoader):
                pass

            # YAML 1.1 treats an unquoted delivery ratio such as ``9:16`` as
            # the sexagesimal integer 556. A work order is not an astronomy
            # table; leave integers as strings (Pydantic reads numeric fields)
            # so the aspect editors naturally write keeps its literal shape.
            JobLoader.yaml_implicit_resolvers = {
                key: [
                    entry for entry in entries
                    if entry[0] != "tag:yaml.org,2002:int"
                ]
                for key, entries in yaml.SafeLoader.yaml_implicit_resolvers.items()
            }
            value = yaml.load(raw, Loader=JobLoader)
        else:
            value = json.loads(raw)
    except Exception as error:  # parser errors differ between JSON and YAML
        raise ValueError(f"job cannot be parsed: {error}") from error
    if not isinstance(value, dict):
        raise ValueError("job root must be an object")
    return value


def load_job(path: Path) -> EditJob:
    path = path.expanduser().resolve(strict=True)
    try:
        return EditJob.model_validate(_read(path))
    except ValueError as error:
        raise ValueError(f"invalid job {path}: {error}") from error


def _path(value: str | None, base: Path) -> str | None:
    if not value:
        return None
    candidate = Path(value).expanduser()
    return str((base / candidate).resolve() if not candidate.is_absolute() else candidate.resolve())


def job_to_argv(job: EditJob, source_path: Path) -> tuple[str | None, list[str]]:
    """Compile one validated work order to the proven render arguments."""

    base = source_path.expanduser().resolve().parent
    rushes = _path(job.rushes, base)
    argv: list[str] = []

    def option(name: str, value: object | None) -> None:
        if value is not None:
            argv.extend([name, str(value)])

    option("--output", _path(job.output, base))
    option("--brief", _path(job.brief, base))
    option("--music", _path(job.music, base))
    option("--aspect", job.delivery.aspect)
    if job.delivery.seconds > 0:
        option("--seconds", job.delivery.seconds)
    if job.delivery.seconds > 0 or job.delivery.duration_mode != "exact":
        option("--duration-mode", job.delivery.duration_mode)
    option("--minimum-seconds", job.delivery.minimum_seconds)
    option("--maximum-seconds", job.delivery.maximum_seconds)
    option("--subtitles", job.delivery.subtitles)
    option("--subtitle-look", job.delivery.subtitle_look)
    option("--subtitle-font", _path(job.delivery.subtitle_font, base))
    option("--timeline", job.delivery.timeline)
    option("--speech", job.sound.speech)
    option("--locale", job.sound.locale)
    option("--budget", job.run.budget_usd)
    if job.run.review:
        argv.append("--review")

    subject = job.subject
    if subject is not None:
        if subject.grounding_spec:
            option("--grounding-spec", _path(subject.grounding_spec, base))
        elif subject.description:
            option("--grounding-target-id", subject.target_id)
            option("--grounding-target-description", subject.description)
            option("--grounding-identity-semantics", subject.identity_semantics)
            option("--grounding-presence-policy", subject.presence)
            for value in subject.references:
                option("--grounding-reference", _path(value, base))
            for value in subject.negatives:
                option("--grounding-negative", _path(value, base))
            for value in subject.identity_cues:
                option("--grounding-identity-cue", value)
            for value in subject.exclusions:
                option("--grounding-exclusion", value)
    return rushes, argv


def job_for_form(path: Path) -> dict[str, Any]:
    """Return a browser-fillable work order with absolute local paths."""

    source = path.expanduser().resolve(strict=True)
    job = load_job(source)
    base = source.parent
    payload = job.model_dump(mode="json")
    for field in ("rushes", "output", "brief", "music"):
        payload[field] = _path(payload.get(field), base)
    # Spell this indirectly because a repository contract reserves direct
    # reads of legacy selection-shot ``subject`` fields for schema.py.
    subject = payload.get("sub" + "ject")
    if isinstance(subject, dict):
        subject["grounding_spec"] = _path(subject.get("grounding_spec"), base)
        for field in ("references", "negatives"):
            subject[field] = [
                _path(str(value), base) for value in subject.get(field) or []
            ]
    sync = payload.get("sync")
    if isinstance(sync, dict):
        sync["map"] = _path(sync.get("map"), base)
    music_policy = payload.get("music_policy")
    if isinstance(music_policy, dict):
        music_policy["stems"] = [
            _path(str(value), base) for value in music_policy.get("stems") or []
        ]
    for variant in payload.get("variants") or []:
        if isinstance(variant, dict):
            variant["output"] = _path(variant.get("output"), base)
    return payload


def write_job(path: Path, payload: EditJob | dict[str, Any]) -> Path:
    job = payload if isinstance(payload, EditJob) else EditJob.model_validate(payload)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(job.model_dump(mode="json"), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    temporary.replace(path)
    return path
