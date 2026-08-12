"""Immutable, locally authoritative planning-state artifacts.

The state in this module is deliberately media- and provider-neutral.  It
records which known material spans remain eligible for a plan, which spans a
revision selected or retained as alternates, and the external obligations the
plan still has to satisfy.  It does not contain prompts, model confidence, or
provider-specific response fields.

Two boundaries make the artifact safe to use across planning stages:

* deltas compare-and-swap against the exact canonical hash of their base;
* revisions are published as complete, immutable directories and never
  overwrite an earlier revision.
"""

from __future__ import annotations

import hashlib
import ctypes
import errno
import json
import os
import re
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


PLANNING_STATE_VERSION = "planning-state-v1"
PLANNING_DELTA_VERSION = "planning-state-delta-v1"
SHA256_PATTERN = r"^[0-9a-f]{64}$"
_STAGE_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
_REVISION_PATTERN = re.compile(r"^rev-([0-9]+)$")

Eligibility = Literal["eligible", "unknown", "hard_invalid"]
Disposition = Literal[
    "available",
    "deferred",
    "selected",
    "alternate",
    "rejected",
    "unknown",
]


class PlanningStateError(ValueError):
    """A planning state, delta, or revision lineage is invalid."""


class PlanningStateConflictError(PlanningStateError):
    """A compare-and-swap base no longer names the current state."""


class PlanningStateLineageError(PlanningStateError):
    """A stored revision does not extend the immediately previous revision."""


class FrozenStrictModel(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        str_strip_whitespace=True,
        allow_inf_nan=False,
    )


def _require_string_json_keys(value: Any, path: str = "$") -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            if not isinstance(key, str):
                raise TypeError(f"canonical JSON object key at {path} must be a string")
            _require_string_json_keys(child, f"{path}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            _require_string_json_keys(child, f"{path}[{index}]")


def canonical_json(value: Any) -> str:
    """Return one deterministic UTF-8 JSON representation.

    ``allow_nan=False`` is part of the contract: NaN and infinity have no
    portable JSON meaning and therefore cannot participate in an authority
    hash or an on-disk revision.
    """

    if isinstance(value, BaseModel):
        value = value.model_dump(mode="json", exclude_none=False)
    _require_string_json_keys(value)
    return json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
        allow_nan=False,
    )


def canonical_sha256(value: Any) -> str:
    """Hash :func:`canonical_json` with SHA-256."""

    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def _unique_non_empty(
    values: tuple[str, ...] | None, field_name: str
) -> tuple[str, ...] | None:
    if values is None:
        return None
    if any(not value for value in values):
        raise ValueError(f"{field_name} values must be non-empty")
    if len(values) != len(set(values)):
        raise ValueError(f"{field_name} values must be unique")
    return values


class MaterialSpanRecord(FrozenStrictModel):
    """One immutable material interval plus its current planning status."""

    source_id: str = Field(min_length=1, max_length=512)
    span_id: str = Field(min_length=1, max_length=512)
    in_seconds: float = Field(ge=0.0, strict=True)
    out_seconds: float = Field(gt=0.0, strict=True)
    eligibility: Eligibility = "unknown"
    disposition: Disposition = "unknown"
    evidence: tuple[str, ...] = ()
    reason: str | None = Field(default=None, min_length=1)

    @field_validator("evidence")
    @classmethod
    def evidence_is_unique(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        return _unique_non_empty(value, "evidence") or ()

    @model_validator(mode="after")
    def interval_and_status_are_coherent(self) -> "MaterialSpanRecord":
        if self.out_seconds <= self.in_seconds:
            raise ValueError("span out_seconds must be greater than in_seconds")
        if self.eligibility == "hard_invalid" and self.disposition in {
            "available",
            "selected",
            "alternate",
        }:
            raise ValueError(
                "hard_invalid span cannot be available, selected, or alternate"
            )
        return self


class PlanningState(FrozenStrictModel):
    """One complete, immutable planning authority revision."""

    contract_version: Literal["planning-state-v1"]
    material_digest: str = Field(pattern=SHA256_PATTERN)
    revision: int = Field(ge=0, strict=True)
    parent_sha256: str | None = Field(default=None, pattern=SHA256_PATTERN)
    spans: tuple[MaterialSpanRecord, ...] = ()
    selected_span_ids: tuple[str, ...] = ()
    alternate_span_ids: tuple[str, ...] = ()
    story_obligations: tuple[str, ...] = ()
    coverage_obligations: tuple[str, ...] = ()
    music_cue_refs: tuple[str, ...] = ()
    grounding_target_refs: tuple[str, ...] = ()

    @field_validator(
        "selected_span_ids",
        "alternate_span_ids",
        "story_obligations",
        "coverage_obligations",
        "music_cue_refs",
        "grounding_target_refs",
    )
    @classmethod
    def references_are_unique(cls, value: tuple[str, ...], info) -> tuple[str, ...]:
        return _unique_non_empty(value, info.field_name) or ()

    @model_validator(mode="after")
    def validate_authority(self) -> "PlanningState":
        if self.revision == 0 and self.parent_sha256 is not None:
            raise ValueError("revision 0 must not have parent_sha256")
        if self.revision > 0 and self.parent_sha256 is None:
            raise ValueError("revision after 0 requires parent_sha256")

        span_ids = [span.span_id for span in self.spans]
        if len(span_ids) != len(set(span_ids)):
            raise ValueError("span_id values must be globally unique")
        known = set(span_ids)
        selected = set(self.selected_span_ids)
        alternates = set(self.alternate_span_ids)
        unknown_selected = selected - known
        unknown_alternates = alternates - known
        if unknown_selected:
            raise ValueError(
                "selected_span_ids contain unknown IDs: "
                + ", ".join(sorted(unknown_selected))
            )
        if unknown_alternates:
            raise ValueError(
                "alternate_span_ids contain unknown IDs: "
                + ", ".join(sorted(unknown_alternates))
            )
        overlap = selected & alternates
        if overlap:
            raise ValueError(
                "selected and alternate IDs must be disjoint: "
                + ", ".join(sorted(overlap))
            )

        selected_by_status = {
            span.span_id for span in self.spans if span.disposition == "selected"
        }
        alternate_by_status = {
            span.span_id for span in self.spans if span.disposition == "alternate"
        }
        if selected_by_status != selected:
            raise ValueError(
                "selected_span_ids must exactly match spans with selected disposition"
            )
        if alternate_by_status != alternates:
            raise ValueError(
                "alternate_span_ids must exactly match spans with alternate disposition"
            )

        hard_invalid = {
            span.span_id
            for span in self.spans
            if span.eligibility == "hard_invalid"
        }
        blocked = hard_invalid & (selected | alternates)
        if blocked:
            raise ValueError(
                "hard_invalid spans cannot be selected or alternate: "
                + ", ".join(sorted(blocked))
            )
        return self

    def canonical_json(self) -> str:
        return canonical_json(self)

    def sha256(self) -> str:
        return canonical_sha256(self)


class PlanningStateDelta(FrozenStrictModel):
    """A full-record patch applied only to its exact base state.

    Span identity and intervals are immutable through a delta.  A changed
    status therefore supplies the complete replacement record for a known
    ``span_id``; this avoids ambiguous partial fields such as an omitted
    reason versus a reason deliberately cleared to ``None``.
    """

    contract_version: Literal["planning-state-delta-v1"]
    base_sha256: str = Field(pattern=SHA256_PATTERN)
    span_updates: tuple[MaterialSpanRecord, ...] = ()
    selected_span_ids: tuple[str, ...] | None = None
    alternate_span_ids: tuple[str, ...] | None = None
    story_obligations: tuple[str, ...] | None = None
    coverage_obligations: tuple[str, ...] | None = None
    music_cue_refs: tuple[str, ...] | None = None
    grounding_target_refs: tuple[str, ...] | None = None

    @field_validator(
        "selected_span_ids",
        "alternate_span_ids",
        "story_obligations",
        "coverage_obligations",
        "music_cue_refs",
        "grounding_target_refs",
    )
    @classmethod
    def replacement_references_are_unique(
        cls, value: tuple[str, ...] | None, info
    ) -> tuple[str, ...] | None:
        return _unique_non_empty(value, info.field_name)

    @model_validator(mode="after")
    def updates_are_unique_and_non_empty(self) -> "PlanningStateDelta":
        update_ids = [span.span_id for span in self.span_updates]
        if len(update_ids) != len(set(update_ids)):
            raise ValueError("span_updates contain duplicate span_id values")
        replacements = (
            self.selected_span_ids,
            self.alternate_span_ids,
            self.story_obligations,
            self.coverage_obligations,
            self.music_cue_refs,
            self.grounding_target_refs,
        )
        if not self.span_updates and all(value is None for value in replacements):
            raise ValueError("planning delta must contain at least one change")
        return self


def apply_planning_delta(
    base: PlanningState, delta: PlanningStateDelta
) -> PlanningState:
    """Apply ``delta`` when its base hash still names ``base``.

    This is an in-memory compare-and-swap.  It never mutates ``base`` and
    always produces the immediately following revision with the base's hash
    as its parent.
    """

    if not isinstance(base, PlanningState):
        raise TypeError("base must be a PlanningState")
    if not isinstance(delta, PlanningStateDelta):
        raise TypeError("delta must be a PlanningStateDelta")
    # ``model_copy(update=...)`` deliberately skips Pydantic validation.
    # Re-validate at the authority boundary so a forged frozen-model instance
    # cannot bypass the same invariants as ordinary JSON input.
    base = PlanningState.model_validate(base.model_dump(mode="python"))
    delta = PlanningStateDelta.model_validate(delta.model_dump(mode="python"))

    base_sha256 = base.sha256()
    if delta.base_sha256 != base_sha256:
        raise PlanningStateConflictError(
            "planning delta base_sha256 does not match current state"
        )

    known = {span.span_id: span for span in base.spans}
    updates = {span.span_id: span for span in delta.span_updates}
    unknown = set(updates) - set(known)
    if unknown:
        raise PlanningStateLineageError(
            "span_updates contain unknown IDs: " + ", ".join(sorted(unknown))
        )
    for span_id, update in updates.items():
        previous = known[span_id]
        if update.source_id != previous.source_id:
            raise PlanningStateLineageError(
                f"span update {span_id!r} cannot change source_id"
            )
        if (
            update.in_seconds != previous.in_seconds
            or update.out_seconds != previous.out_seconds
        ):
            raise PlanningStateLineageError(
                f"span update {span_id!r} cannot change its interval"
            )
        if (
            previous.eligibility == "hard_invalid"
            and update.eligibility != "hard_invalid"
        ):
            raise PlanningStateLineageError(
                f"span update {span_id!r} cannot reverse hard_invalid "
                "without a new material lineage"
            )

    spans = tuple(updates.get(span.span_id, span) for span in base.spans)
    selected = (
        delta.selected_span_ids
        if delta.selected_span_ids is not None
        else base.selected_span_ids
    )
    alternates = (
        delta.alternate_span_ids
        if delta.alternate_span_ids is not None
        else base.alternate_span_ids
    )
    story = (
        delta.story_obligations
        if delta.story_obligations is not None
        else base.story_obligations
    )
    coverage = (
        delta.coverage_obligations
        if delta.coverage_obligations is not None
        else base.coverage_obligations
    )
    music = (
        delta.music_cue_refs
        if delta.music_cue_refs is not None
        else base.music_cue_refs
    )
    grounding = (
        delta.grounding_target_refs
        if delta.grounding_target_refs is not None
        else base.grounding_target_refs
    )

    if (
        spans == base.spans
        and selected == base.selected_span_ids
        and alternates == base.alternate_span_ids
        and story == base.story_obligations
        and coverage == base.coverage_obligations
        and music == base.music_cue_refs
        and grounding == base.grounding_target_refs
    ):
        raise PlanningStateError("planning delta makes no semantic change")

    return PlanningState(
        contract_version=PLANNING_STATE_VERSION,
        material_digest=base.material_digest,
        revision=base.revision + 1,
        parent_sha256=base_sha256,
        spans=spans,
        selected_span_ids=selected,
        alternate_span_ids=alternates,
        story_obligations=story,
        coverage_obligations=coverage,
        music_cue_refs=music,
        grounding_target_refs=grounding,
    )


def _stage_name(value: str) -> str:
    value = value.strip()
    if not _STAGE_PATTERN.fullmatch(value):
        raise ValueError(
            "planning stage must be one safe path component containing only "
            "letters, digits, dot, underscore, or hyphen"
        )
    return value


def _revision_numbers(stage_directory: Path) -> list[int]:
    numbers: list[int] = []
    for child in stage_directory.iterdir():
        matched = _REVISION_PATTERN.fullmatch(child.name)
        if matched and child.is_dir():
            numbers.append(int(matched.group(1)))
    return sorted(numbers)


def _load_stored_state(path: Path) -> PlanningState:
    try:
        return PlanningState.model_validate_json(path.read_text(encoding="utf-8"))
    except Exception as error:
        raise PlanningStateLineageError(
            f"stored parent state is invalid: {path}"
        ) from error


def _validate_storage_lineage(
    stage_directory: Path, state: PlanningState
) -> None:
    revisions = _revision_numbers(stage_directory)
    if state.revision == 0:
        if revisions:
            raise PlanningStateLineageError(
                "revision 0 cannot be appended to a stage with stored revisions"
            )
        return

    expected = list(range(state.revision))
    if revisions != expected:
        raise PlanningStateLineageError(
            f"revision {state.revision} requires contiguous stored revisions "
            f"0 through {state.revision - 1}"
        )
    parent_path = (
        stage_directory / f"rev-{state.revision - 1}" / "state.json"
    )
    if not parent_path.is_file():
        raise PlanningStateLineageError(
            f"revision {state.revision} has no stored parent state"
        )
    parent = _load_stored_state(parent_path)
    if parent.revision != state.revision - 1:
        raise PlanningStateLineageError("stored parent revision number is stale")
    if parent.sha256() != state.parent_sha256:
        raise PlanningStateLineageError(
            "parent_sha256 does not match the stored parent state"
        )
    if parent.material_digest != state.material_digest:
        raise PlanningStateLineageError(
            "material_digest cannot change within a planning lineage"
        )
    parent_material = tuple(
        (span.span_id, span.source_id, span.in_seconds, span.out_seconds)
        for span in parent.spans
    )
    current_material = tuple(
        (span.span_id, span.source_id, span.in_seconds, span.out_seconds)
        for span in state.spans
    )
    if parent_material != current_material:
        raise PlanningStateLineageError(
            "material span IDs, sources, intervals, and order cannot change "
            "within a planning lineage"
        )
    revived = [
        current.span_id
        for previous, current in zip(parent.spans, state.spans)
        if previous.eligibility == "hard_invalid"
        and current.eligibility != "hard_invalid"
    ]
    if revived:
        raise PlanningStateLineageError(
            "hard_invalid spans cannot be restored within a planning lineage: "
            + ", ".join(revived)
        )


def _write_complete_file(path: Path, text: str) -> None:
    with path.open("x", encoding="utf-8") as handle:
        handle.write(text)
        handle.flush()
        os.fsync(handle.fileno())


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _rename_no_replace(source: Path, destination: Path) -> None:
    """Atomically rename a directory while refusing an existing target.

    POSIX ``rename(2)`` may replace an empty destination directory, so an
    ``exists()`` check followed by :func:`os.rename` does not provide the
    never-overwrite property this store promises.  macOS and Linux both
    expose a kernel-level exclusive rename.  If a platform cannot provide
    that primitive, fail closed instead of publishing with a race window.
    """

    encoded_source = os.fsencode(source)
    encoded_destination = os.fsencode(destination)
    libc = ctypes.CDLL(None, use_errno=True)
    result: int
    if sys.platform == "darwin":
        renamex_np = getattr(libc, "renamex_np", None)
        if renamex_np is None:
            raise OSError(errno.ENOSYS, "atomic no-replace rename is unavailable")
        renamex_np.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_uint]
        renamex_np.restype = ctypes.c_int
        # Darwin's <stdio.h>: RENAME_EXCL refuses an existing destination.
        result = renamex_np(encoded_source, encoded_destination, 0x00000004)
    elif sys.platform.startswith("linux"):
        renameat2 = getattr(libc, "renameat2", None)
        if renameat2 is None:
            raise OSError(errno.ENOSYS, "atomic no-replace rename is unavailable")
        renameat2.argtypes = [
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_uint,
        ]
        renameat2.restype = ctypes.c_int
        # Linux's <stdio.h>: AT_FDCWD and RENAME_NOREPLACE.
        result = renameat2(
            -100,
            encoded_source,
            -100,
            encoded_destination,
            0x00000001,
        )
    elif os.name == "nt":
        # Windows ``os.rename`` already refuses to replace an existing path.
        os.rename(source, destination)
        return
    else:
        raise OSError(errno.ENOSYS, "atomic no-replace rename is unavailable")

    if result == 0:
        return
    error_number = ctypes.get_errno()
    if error_number in {errno.EEXIST, errno.ENOTEMPTY}:
        raise FileExistsError(
            error_number,
            os.strerror(error_number),
            str(destination),
        )
    raise OSError(error_number, os.strerror(error_number), str(destination))


def write_planning_revision(
    work_directory: Path,
    stage: str,
    state: PlanningState,
    *,
    request: Any,
    response: Any,
    validation: Any,
) -> Path:
    """Atomically publish one never-overwritten planning revision directory.

    The returned path is exactly
    ``work/planning/<stage>/rev-N`` when ``work_directory`` names ``work``.
    All four JSON files are completed and flushed in a hidden sibling
    directory before one atomic directory rename makes the revision visible.
    A stage-wide exclusive marker serialises cooperating writers so two
    concurrent CAS successors cannot both publish.
    """

    if not isinstance(state, PlanningState):
        raise TypeError("state must be a PlanningState")
    # See the corresponding check in ``apply_planning_delta``.  Persistence
    # is a second authority boundary and must not trust a Pydantic instance
    # merely because its class is correct.
    state = PlanningState.model_validate(state.model_dump(mode="python"))
    stage = _stage_name(stage)

    # Prove every sidecar is canonical-JSON serialisable before creating a
    # lock or staging directory.  A provider object must be projected to a
    # plain local record by its caller instead of leaking into this authority.
    documents = {
        "state.json": state.canonical_json() + "\n",
        "request.json": canonical_json(request) + "\n",
        "response.json": canonical_json(response) + "\n",
        "validation.json": canonical_json(validation) + "\n",
    }

    stage_directory = Path(work_directory) / "planning" / stage
    stage_directory.mkdir(parents=True, exist_ok=True)
    destination = stage_directory / f"rev-{state.revision}"
    lock_path = stage_directory / ".write.lock"
    try:
        lock_descriptor = os.open(
            lock_path,
            os.O_CREAT | os.O_EXCL | os.O_WRONLY,
            0o600,
        )
    except FileExistsError as error:
        raise PlanningStateConflictError(
            f"another writer owns planning stage {stage!r}"
        ) from error
    os.close(lock_descriptor)

    staging: Path | None = None
    try:
        if destination.exists():
            raise FileExistsError(
                f"planning revision already exists and will not be overwritten: "
                f"{destination}"
            )
        _validate_storage_lineage(stage_directory, state)
        staging = Path(
            tempfile.mkdtemp(
                prefix=f".rev-{state.revision}-",
                dir=stage_directory,
            )
        )
        for name, text in documents.items():
            _write_complete_file(staging / name, text)
        _fsync_directory(staging)
        if destination.exists():
            raise FileExistsError(
                f"planning revision appeared while writing: {destination}"
            )
        _rename_no_replace(staging, destination)
        staging = None
        _fsync_directory(stage_directory)
        return destination
    finally:
        if staging is not None and staging.exists():
            shutil.rmtree(staging)
        lock_path.unlink(missing_ok=True)


__all__ = [
    "Disposition",
    "Eligibility",
    "MaterialSpanRecord",
    "PLANNING_DELTA_VERSION",
    "PLANNING_STATE_VERSION",
    "PlanningState",
    "PlanningStateConflictError",
    "PlanningStateDelta",
    "PlanningStateError",
    "PlanningStateLineageError",
    "apply_planning_delta",
    "canonical_json",
    "canonical_sha256",
    "write_planning_revision",
]
