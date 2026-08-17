"""One command from a folder of rushes to a finished cut.

Every stage writes what it decided next to the output, because a run nobody
can inspect afterwards is a run nobody can argue with. The report is the
deliverable as much as the file is.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import shutil
import subprocess
import signal
import sys
import time
from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING, Any, Sequence

if TYPE_CHECKING:
    from montagewright.reference_grounding import ReferenceGroundingSpec

from montagewright.clipcard import (
    CARD_VERSION,
    action_beats,
    build_library,
    find_subject,
    load_card,
    snap_to_action_contract,
    subjects_from_card,
)
from montagewright import tracked_geometry
from montagewright.cost import BudgetSpent, Ledger
from montagewright.grounding import (
    analyse_track, beat_grid_from_payload, beat_grid_payload, load_beat_grid,
    read_runtime_beat_grid, shots_in,
)
from montagewright.measure.media import sha256_file
from montagewright.pipeline import (
    ReferenceShotsUnusable,
    _preflight_sam_checkpoint,
    probe,
    run,
)
from montagewright.planning_bridge import (
    material_planning_state,
    publish_planning_state,
    selection_planning_state,
)
from montagewright.planning_artifacts import (
    asked as _asked,
    decide as _decide,
    decided as _decided,
    latest_decision as _latest_decision,
    planning_contract as _planning_contract,
)
from montagewright.candidate_commitments import (
    CandidateCommitments,
    CommitmentError,
    resolve_candidate_commitments,
)
from montagewright.review import (
    Round,
    actionable_keys,
    adjudicate,
    review_cut,
    review_shots,
    should_continue,
)
from montagewright.planner import (
    MAX_OUTPUT_TOKENS,
    MODEL_ID,
    PROMPTS,
    THINKING_HIGH,
    MaterialItem,
    SelectionUnrenderable,
    _describe_one,
    _direction_schema,
    _selection_schema,
    _shot_count_bounds,
    audit_cached_selection,
    correct_candidate_options,
    decide_direction,
    replan_shots,
    repair_selection_motion_contracts,
    repair_selection_source_windows,
    normalize_selection,
    repair_single_look_hold_overflow,
    sequence_disagreements,
    select_shots,
)
from montagewright.schema import (
    EDL,
    Clip,
    ContentContract,
    delivered_camera_intent_of,
    looks_of,
    move_of_shot,
    reframe_of,
    subject_of,
)
from montagewright.spans import spans_of
from montagewright.uploads import (
    UploadCache,
    content_hash,
    default_cache_path,
    default_library,
)

ASPECTS = {"16:9": 16 / 9, "9:16": 9 / 16, "1:1": 1.0, "4:5": 4 / 5}
SAM_CHECKPOINT_NAME = "sam2.1_hiera_tiny.pt"

# Long enough that it is worth asking whether this is one take or many. Under
# it, scene detection costs more than it can save.
SPLIT_ABOVE_SECONDS = 90.0
VIDEO_SUFFIXES = {".mp4", ".mov", ".m4v", ".MP4", ".MOV"}


def prepare_grounding_spec_artifact(
    source: Path, destination: Path,
) -> tuple[Path, ReferenceGroundingSpec]:
    """Validate and make one self-contained, canonical grounding input.

    ``reference_grounding`` owns the schema and every semantic validation.
    This helper owns only run transport: reference images are copied beside
    the canonical JSON under content-addressed names, then the rewritten
    document is loaded a second time.  CLI and Web both call this function so
    an upload cannot acquire a more permissive interpretation than a path.
    """

    from montagewright.reference_grounding import (
        ReferenceGroundingError,
        load_grounding_spec,
    )

    def checked_load(path: Path) -> ReferenceGroundingSpec:
        try:
            return load_grounding_spec(path)
        except ReferenceGroundingError as error:
            raise ValueError(str(error)) from error

    source = source.expanduser().resolve()
    destination = destination.expanduser().resolve()
    if not source.is_file():
        raise ValueError(f"grounding spec is not there: {source}")

    spec = checked_load(source)
    canonical = spec.canonical_definition_json()
    if isinstance(canonical, bytes):
        canonical = canonical.decode("utf-8")
    payload = json.loads(canonical)
    definitions = payload.get("reference_images")
    references = tuple(spec.reference_images)
    if not isinstance(definitions, list) or len(definitions) != len(references):
        raise ValueError(
            "grounding spec canonical form changed its reference_images"
        )

    destination.parent.mkdir(parents=True, exist_ok=True)
    image_root = destination.parent / "reference-images"
    image_root.mkdir(parents=True, exist_ok=True)
    for definition, reference in zip(definitions, references, strict=True):
        image = spec.resolve_reference_path(reference).resolve()
        if not image.is_file():
            raise ValueError(f"reference image is not there: {image}")
        digest = str(reference.content_sha256).lower()
        suffix = Path(str(reference.path)).suffix.lower()
        stored = image_root / f"{digest}{suffix}"
        if image != stored:
            shutil.copyfile(image, stored)
        definition["path"] = stored.relative_to(destination.parent).as_posix()

    staged = destination.with_name(f".{destination.name}.tmp")
    staged.write_text(
        json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ),
        encoding="utf-8",
    )
    try:
        checked = checked_load(staged)
        rendered = checked.canonical_definition_json()
        if isinstance(rendered, bytes):
            rendered = rendered.decode("utf-8")
        staged.write_text(rendered, encoding="utf-8")
        staged.replace(destination)
    finally:
        staged.unlink(missing_ok=True)
    # Reload after the atomic rename so the runtime-only source path names the
    # durable artifact rather than the now-deleted staging file.
    return destination, checked_load(destination)


def _command_with_canonical_grounding(
    argv: list[str], grounding_spec: Path | None,
) -> list[str]:
    """Record the resumable command against the durable canonical artifact."""

    if grounding_spec is None:
        return list(argv)
    simple_flags = {
        "--grounding-target-id", "--grounding-target-description",
        "--grounding-reference", "--grounding-negative",
        "--grounding-identity-cue", "--grounding-exclusion",
    }
    rewritten: list[str] = []
    skip_value = False
    for argument in argv:
        if skip_value:
            skip_value = False
            continue
        if argument in simple_flags:
            skip_value = True
            continue
        if any(argument.startswith(f"{flag}=") for flag in simple_flags):
            continue
        rewritten.append(argument)
    for index, argument in enumerate(rewritten):
        if argument == "--grounding-spec" and index + 1 < len(rewritten):
            rewritten[index + 1] = str(grounding_spec)
            return rewritten
        if argument.startswith("--grounding-spec="):
            rewritten[index] = f"--grounding-spec={grounding_spec}"
            return rewritten
    rewritten += ["--grounding-spec", str(grounding_spec)]
    return rewritten


def _default_sam_checkpoint(
    search_roots: tuple[Path, ...] | None = None,
) -> Path | None:
    """Find the bundled SAM model without making callers know the repo layout."""

    roots = search_roots or (
        Path.cwd(),
        Path(__file__).resolve().parents[2],
    )
    for root in roots:
        candidate = root / "artifacts" / "models" / SAM_CHECKPOINT_NAME
        if candidate.is_file():
            return candidate.resolve()
    return None


def _sam_checkpoint_for(args: argparse.Namespace) -> Path | None:
    """SAM is the default; falling back must be explicit in the run log."""

    requested = getattr(args, "sam_checkpoint", None)
    disabled = bool(getattr(args, "no_sam_tracking", False))
    if requested is not None and disabled:
        raise SystemExit("choose either --sam-checkpoint or --no-sam-tracking")
    if disabled:
        print("SAM tracking disabled by --no-sam-tracking", flush=True)
        return None
    if requested is not None:
        checkpoint = requested.expanduser().resolve()
        if not checkpoint.is_file():
            raise SystemExit(f"SAM checkpoint is not there: {checkpoint}")
        print(f"SAM tracking: {checkpoint}", flush=True)
        return checkpoint
    checkpoint = _default_sam_checkpoint()
    if checkpoint is not None:
        print(f"SAM tracking: {checkpoint} (auto)", flush=True)
        return checkpoint
    print(
        "WARNING: SAM tracking is unavailable because "
        f"artifacts/models/{SAM_CHECKPOINT_NAME} was not found; moving "
        "subjects will use sparse Gemini samples. Pass --no-sam-tracking "
        "to acknowledge this fallback explicitly.",
        flush=True,
    )
    return None


def _client():
    from google import genai
    from google.genai import types
    from montagewright.environment import load_project_env

    load_project_env()
    key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not key:
        raise SystemExit("GEMINI_API_KEY is required")
    from montagewright.planner import _http_options

    return genai.Client(api_key=key, http_options=_http_options(types))


def _why(fault: Exception) -> str:
    """The reason a shot could not be delivered, in one line.

    Batching the repairs dropped it: the messages named the shot and the
    identity and said nothing about whether the frames had refused it or
    the tracker had never held it -- which are the two things anyone
    reading the log needs to tell apart.
    """

    said = str(fault)
    return (said.split(": ", 1)[-1] if ": " in said else said)[:150]


def _swap_for_alternate(
    shot: dict[str, Any],
    direction: dict[str, Any],
    *,
    taken: set[str],
    exhausted: set[str] | None = None,
) -> dict[str, Any] | None:
    """The other take the direction named for this same commitment.

    Every commitment came back with a primary and an alternate, sixteen
    options for eight commitments, and until now nothing had ever read the
    second one: a shot that failed on delivery ended the run while its
    replacement sat in the artifact, paid for.
    """

    commitment = str(shot.get("commitment_id") or "")
    if not commitment:
        return None
    here = str(shot.get("span_id") or "")
    for option in direction.get("candidate_options") or []:
        if str(option.get("commitment_id") or "") != commitment:
            continue
        span_id = str(option.get("span_id") or "")
        if not span_id or span_id == here or span_id in taken:
            continue
        # A span that already failed is not an alternate. Without this the
        # swap walked back to it the moment it stopped being "taken": two
        # candidates, one failing each round, traded places until the retry
        # budget ran out -- and every round paid to judge them both again.
        if exhausted is not None and span_id in exhausted:
            continue
        source_id = span_id.split(":")[0]
        return {
            **shot,
            "span_id": span_id,
            "source_id": source_id,
            "why": str(option.get("why") or shot.get("why") or ""),
            # The window belongs to the new span, and the old one's offset
            # meant nothing here. Everything else -- role, commitment, the
            # length its content proved -- is about the job, not the take.
            "start_offset_seconds": 0.0,
        }
    return None


def _confirm_material_identity_for_target(
    material: list[Any],
    sightings: dict[str, Any],
    spec: Any,
    target: str,
    *,
    masters: dict[str, Path],
    client: Any,
    cache: Any,
    ledger: Any,
    library: Path,
    work: Path,
    spread: bool = False,
    outcomes: dict[tuple[str, str], dict[str, str]] | None = None,
) -> dict[str, tuple[Any, ...]]:
    """Where one target was proved in each source, on the master's clock."""

    from montagewright.reference_grounding import (
        confirm_source_identity,
        confirmed_frame_from_validated_seed,
        decide_cross_asset_exact_frame_bboxes,
        identity_box_ratio_disagreement,
        prepare_source_identity_seed,
        read_source_confirmation_cache,
        read_source_confirmation_status,
        source_confirmation_cache_path,
        write_source_confirmation_cache,
    )

    def ratio_warning(frames: Sequence[Any]) -> str | None:
        return identity_box_ratio_disagreement(spec, target, frames)

    confirmed: dict[str, tuple[Any, ...]] = {}
    paid_before = float(getattr(ledger, "spent_usd", 0.0))
    total = len(material)
    prepared = []
    prepared_sources: list[tuple[Any, Any, Path, Path]] = []
    fallback: list[Any] = []
    for item in material:
        if not getattr(item, "carries_identity", True):
            # The screen already answered this one. Asking again costs money
            # to be told the same thing.
            continue
        discovery = sightings.get(item.source_id)
        # The master: these frames are decoded at 1440 and their boxes are
        # handed to a tracker that reads the master too.
        source = masters.get(item.source_id)
        if discovery is None or source is None:
            continue
        cache_path = source_confirmation_cache_path(
            Path(source), spec, target, library
        )
        remembered = read_source_confirmation_cache(cache_path, target)
        if remembered is not None:
            if outcomes is not None:
                outcomes[(item.source_id, target)] = {
                    "status": read_source_confirmation_status(cache_path, target)
                    or ("confirmed" if remembered else "uncertain"),
                    "reason": "source confirmation cache",
                }
                warning = ratio_warning(remembered)
                if warning:
                    outcomes[(item.source_id, target)]["ratio_disagreement"] = warning
            if remembered:
                confirmed[item.source_id] = remembered
            continue
        try:
            one = prepare_source_identity_seed(
                Path(source), spec, discovery, target,
                work / "identity-frames" / item.source_id,
                at_ms=(
                    tuple(
                        int(item.duration_seconds * 1000 * share)
                        for share in (0.1, 0.3, 0.5, 0.7, 0.9)
                    ) if spread else ()
                ),
            )
        except Exception as error:  # local preparation; singleton path reports it
            print(
                f"  {item.source_id} — seed preparation failed: "
                f"{type(error).__name__}: {error}"[:150],
                flush=True,
            )
            fallback.append(item)
            continue
        if one is None:
            fallback.append(item)
            continue
        prepared.append(one.item)
        prepared_sources.append((item, one, cache_path, Path(source)))

    if prepared:
        ledger.check()
        try:
            judged = decide_cross_asset_exact_frame_bboxes(
                spec, target, prepared,
                client=client, cache=cache, ledger=ledger,
            )
        except BudgetSpent:
            raise
        except Exception as error:  # noqa: BLE001 -- reported, not swallowed
            print(
                "  cross-source identity batch failed: "
                f"{type(error).__name__}: {error}"[:150],
                flush=True,
            )
            fallback.extend(item for item, _, _, _ in prepared_sources)
        else:
            batch_outcomes = judged[0].outcomes if judged is not None else ()
            for index, (item, seed, cache_path, source) in enumerate(
                prepared_sources
            ):
                outcome = (
                    batch_outcomes[index]
                    if index < len(batch_outcomes) else None
                )
                found = ()
                video_sha256 = ""
                semantic_answer = False
                decision = None
                if outcome is not None and outcome.evaluation is not None:
                    video_sha256 = outcome.evaluation.lineage.video_sha256
                    semantic_answer = True
                    decision = outcome.evaluation.decision
                    try:
                        frame = confirmed_frame_from_validated_seed(
                            seed, outcome.evaluation
                        )
                    except Exception:
                        frame = None
                    if frame is not None and not frame.seed_risk_flags:
                        found = (frame,)
                if found:
                    if outcomes is not None:
                        outcomes[(item.source_id, target)] = {
                            "status": "confirmed", "reason": "clean exact seed",
                        }
                        warning = ratio_warning(found)
                        if warning:
                            outcomes[(item.source_id, target)][
                                "ratio_disagreement"
                            ] = warning
                    confirmed[item.source_id] = found
                    write_source_confirmation_cache(
                        cache_path,
                        target,
                        video_sha256,
                        found,
                    )
                else:
                    if semantic_answer and decision is not None:
                        if decision.verdict != "matched_target":
                            # This verdict belongs to one exact frame, not the
                            # whole source.  An occluded recommended seed must
                            # not become a durable source-negative cache entry;
                            # let the mature path sample independent moments.
                            fallback.append(item)
                            continue
                    # A matched but risky seed needs independent anchors; a
                    # structural failure needs the mature singleton path.
                    fallback.append(item)

    for item in fallback:
        discovery = sightings.get(item.source_id)
        source = masters.get(item.source_id)
        if discovery is None or source is None:
            continue
        ledger.check()
        detail: dict[str, str] = {}
        try:
            found = confirm_source_identity(
                Path(source), spec, discovery, target,
                client=client,
                frames_dir=work / "identity-frames" / item.source_id,
                cache=cache, ledger=ledger, library=library,
                at_ms=(
                    tuple(
                        int(item.duration_seconds * 1000 * share)
                        for share in (0.1, 0.3, 0.5, 0.7, 0.9)
                    ) if spread else ()
                ),
                outcome=detail,
            )
        except BudgetSpent:
            raise
        except Exception as error:  # noqa: BLE001 -- reported, not swallowed
            print(
                f"  {item.source_id} — identity not confirmed: "
                f"{type(error).__name__}: {error}"[:150],
                flush=True,
            )
            found = ()
            detail = {
                "status": "provider_failure",
                "reason": f"{type(error).__name__}: {error}",
            }
        if outcomes is not None:
            outcomes[(item.source_id, target)] = detail or {
                "status": "uncertain", "reason": "no confirmed source frame",
            }
            warning = ratio_warning(found)
            if warning:
                outcomes[(item.source_id, target)]["ratio_disagreement"] = warning
        if found:
            confirmed[item.source_id] = found

    for index, item in enumerate(material, start=1):
        found = confirmed.get(item.source_id, ())
        print(
            f"  identity {index}/{total}  {item.source_id}  "
            + (f"{len(found)} confirmed" if found else "none"),
            flush=True,
        )
    print(
        f"identity confirmed on {len(confirmed)}/{len(material)} sources "
        f"(${float(getattr(ledger, 'spent_usd', 0.0)) - paid_before:.4f})",
        flush=True,
    )
    return confirmed


def _confirm_material_identity(
    material: list[Any],
    sightings: dict[str, Any],
    spec: Any,
    *,
    masters: dict[str, Path],
    client: Any,
    cache: Any,
    ledger: Any,
    library: Path,
    work: Path,
    spread: bool = False,
    outcomes: dict[tuple[str, str], dict[str, str]] | None = None,
) -> dict[str, dict[str, tuple[Any, ...]]]:
    """Return exact source confirmations keyed by both source and target."""

    required = tuple(dict.fromkeys(
        spec.identity_lock.framing.required_target_ids
        or [target.target_id for target in spec.identity_lock.identity.targets]
    ))
    confirmed: dict[str, dict[str, tuple[Any, ...]]] = {}
    for target_id in required:
        for source_id, frames in _confirm_material_identity_for_target(
            material,
            sightings,
            spec,
            target_id,
            masters=masters,
            client=client,
            cache=cache,
            ledger=ledger,
            library=library,
            work=work,
            spread=spread,
            outcomes=outcomes,
        ).items():
            confirmed.setdefault(source_id, {})[target_id] = frames
    return confirmed


def _identity_commitment_sources(commitments: Any) -> set[str]:
    """Sources whose primary/alternate picture promises the locked target."""

    return {
        option.span_id.split(":", 1)[0]
        for option in commitments.options
        if option.target_id != "none"
    }


def _commitments_without_exact_hard_negatives(
    commitments: Any,
    outcomes: dict[tuple[str, str], dict[str, str]],
    *,
    require_confirmation: set[tuple[str, str]] | frozenset[tuple[str, str]] = frozenset(),
) -> Any:
    """Remove only source/target pairs disproved by independent exact frames.

    `uncertain` and provider failures normally remain reviewable candidates.
    The exception is a source/target pair which Direction promoted after the
    cheap screen explicitly found it absent: that disagreement was allowed
    through only so master-resolution exact frames could settle it.  If those
    frames do not confirm it, the pair remains useful context but may not
    satisfy an identity-bearing commitment.
    """

    from montagewright.candidate_commitments import CandidateCommitments

    grouped: dict[str, list[Any]] = {}
    order: list[str] = []
    for option in commitments.options:
        if option.commitment_id not in grouped:
            grouped[option.commitment_id] = []
            order.append(option.commitment_id)
        grouped[option.commitment_id].append(option)
    kept: list[Any] = []
    warnings = list(commitments.warnings)
    for commitment_id in order:
        original = grouped[commitment_id]
        surviving = []
        for option in original:
            pair = (option.span_id.split(":", 1)[0], option.target_id)
            status = outcomes.get(pair, {}).get("status")
            if status == "hard_negative":
                continue
            if pair in require_confirmation and status != "confirmed":
                warnings.append(
                    f"{commitment_id} option {option.span_id} was promoted "
                    "after a negative proxy screen but exact master frames "
                    "did not confirm the target"
                )
                continue
            surviving.append(option)
        if not surviving:
            warnings.append(
                f"{commitment_id} has no exact-eligible identity option; kept "
                "as ungrounded and marked for review rather than ending the run"
            )
            surviving = [
                option.model_copy(update={
                    "target_id": "none",
                    "identity_status": "needs_review",
                    "identity_issue": (
                        "所有 exact-frame 候選均為 hard negative；保留未接地草稿，"
                        "必須人工替換或確認。"
                    ),
                })
                for option in original
            ]
        if not any(option.tier == "primary" for option in surviving):
            surviving[0] = surviving[0].model_copy(update={"tier": "primary"})
        kept.extend(surviving)
    payload = commitments.model_dump(mode="python")
    payload.update({"options": tuple(kept), "warnings": tuple(dict.fromkeys(warnings))})
    return CandidateCommitments.model_validate(payload)


def _annotate_selection_identity_evidence(
    selection: dict[str, Any], confirmed: dict[str, Any],
    outcomes: dict[tuple[str, str], dict[str, str]] | None = None,
) -> None:
    """Say what identity evidence exists before final-window tracking.

    Source confirmation is useful evidence but is not proof that the identity
    remains inside the exact selected window, and neither is a positive proxy
    screen.  Keeping these states distinct prevents an unfinished draft from
    reporting every target-bearing shot as fully confirmed.
    """

    notes = selection.setdefault("plan_disagreements", [])
    for index, shot in enumerate(selection.get("shots") or []):
        targets = tuple(dict.fromkeys(
            str(look.get("entity_id") or "").strip()
            for look in shot.get("looks") or []
            if str(look.get("entity_id") or "").strip() not in {"", "none"}
        ))
        if not targets:
            shot.setdefault("identity_status", "not_applicable")
            continue
        source_id = str(shot.get("source_id") or "")
        shot["identity_target_id"] = targets[0]
        shot["identity_target_ids"] = list(targets)
        source_confirmed = confirmed.get(source_id) or {}
        ratio_warnings = [
            str((outcomes or {}).get((source_id, target), {}).get(
                "ratio_disagreement", ""
            )).strip()
            for target in targets
        ]
        ratio_warnings = [one for one in ratio_warnings if one]
        if isinstance(source_confirmed, dict):
            missing = [
                target for target in targets
                if not source_confirmed.get(target)
            ]
        else:
            # Transitional source-only data can never prove which target its
            # frames belong to; inspect their embedded lineage and fail closed.
            missing = [
                target for target in targets
                if not any(
                    getattr(frame, "target_id", None) == target
                    for frame in source_confirmed
                )
            ]
        if not missing:
            shot["identity_status"] = "source_confirmed"
            shot["identity_issue"] = (
                "來源中的 exact frame 已確認；仍要等這個最終片段的 SAM "
                "連續性通過，才算最終追蹤已驗證。"
            )
        else:
            shot["identity_status"] = "unverified"
            shot["identity_issue"] = (
                "粗篩認為來源可能包含指定主體，但來源 exact frame 尚未證明；"
                "保留為可檢視草稿，不冒充已確認。"
            )
            note = (
                f"k{index:02d} {source_id} claims {', '.join(missing)} but has no "
                "confirmed source exact frame; final-window grounding must "
                "confirm it or leave the shot marked for review"
            )
            if note not in notes:
                notes.append(note)
        for warning in ratio_warnings:
            advisories = shot.setdefault("identity_advisories", [])
            if warning not in advisories:
                advisories.append(warning)
            shot["identity_issue"] = "; ".join(
                one for one in (str(shot.get("identity_issue") or ""), warning)
                if one
            )
            note = f"k{index:02d} {source_id}: {warning}"
            if note not in notes:
                notes.append(note)


def _annotate_selection_direction_motion(
    selection: dict[str, Any], commitments: Any
) -> None:
    """Carry Direction's camera advice beside Selection's final choice."""

    by_pair = {
        (option.commitment_id, option.span_id): option
        for option in commitments.options
    }
    notes = selection.setdefault("plan_disagreements", [])
    for index, shot in enumerate(selection.get("shots") or []):
        option = by_pair.get((
            str(shot.get("commitment_id") or ""),
            str(shot.get("span_id") or ""),
        ))
        if option is None:
            continue
        shot["direction_motion_advice"] = {
            "treatment": option.direction_treatment,
            "move": option.direction_suggested_move,
            "route": option.direction_camera_route,
            "reason": option.direction_motion_reason,
            "fallback": option.direction_fallback_treatment,
            "locally_feasible": list(option.feasible_treatments),
        }
        selected = str(shot.get("camera_intent") or "hold")
        agrees = selected == option.direction_treatment
        shot["agrees_with_direction"] = agrees
        if agrees:
            shot.setdefault("direction_disagreement_reason", "")
            continue
        reason = str(shot.get("direction_disagreement_reason") or "").strip()
        if not reason:
            reason = str(shot.get("why") or "Selection chose after viewing")
            shot["direction_disagreement_reason"] = reason
        note = (
            f"k{index:02d}: Direction advised "
            f"{option.direction_treatment}, Selection requested {selected}: "
            f"{reason}"
        )
        if note not in notes:
            notes.append(note)


def _project_track_confirmed(selection: dict[str, Any], report: Any) -> None:
    """Project final per-target SAM proof back onto the reviewable shot state."""

    grounding = getattr(report, "reference_grounding", {}) or {}
    for index, shot in enumerate(selection.get("shots") or []):
        targets = tuple(dict.fromkeys(
            str(look.get("entity_id") or "").strip()
            for look in shot.get("looks") or []
            if str(look.get("entity_id") or "").strip() not in {"", "none"}
        ))
        if not targets:
            continue
        record = grounding.get(f"k{index:02d}") or {}
        per_target = record.get("targets") if isinstance(record, dict) else {}
        if not isinstance(per_target, dict):
            per_target = {}
        statuses = {
            target: str((per_target.get(target) or {}).get("status") or "")
            for target in targets
        }
        shot["identity_track_statuses"] = statuses
        if statuses and all(
            status == "sam_geometry_validated" for status in statuses.values()
        ):
            shot["identity_status"] = "track_confirmed"
            shot["identity_issue"] = ""


def _merge_identity_windows(
    windows: list[tuple[float, float]], *, tolerance: float = 1e-3
) -> tuple[tuple[float, float], ...]:
    """Merge overlaps but never bridge an interval where a target was absent."""

    merged: list[list[float]] = []
    for starts, ends in sorted(windows):
        if ends <= starts:
            continue
        if merged and starts <= merged[-1][1] + tolerance:
            merged[-1][1] = max(merged[-1][1], ends)
        else:
            merged.append([starts, ends])
    return tuple((round(one[0], 3), round(one[1], 3)) for one in merged)


def _spans_inside_identity_windows(
    spans: tuple[Any, ...], windows: tuple[tuple[float, float], ...]
) -> tuple[Any, ...]:
    """Intersect spans with connected sightings, preserving every dry gap."""

    surviving: list[Any] = []
    for span in spans:
        overlaps = _merge_identity_windows([
            (max(span.starts_seconds, starts), min(span.ends_seconds, ends))
            for starts, ends in windows
            if span.starts_seconds < ends and span.ends_seconds > starts
        ])
        usable = [one for one in overlaps if one[1] - one[0] >= 0.5]
        for part_index, (opens, closes) in enumerate(usable):
            unchanged = (
                len(usable) == 1
                and opens <= span.starts_seconds + 1e-3
                and closes >= span.ends_seconds - 1e-3
            )
            surviving.append(span if unchanged else replace(
                span,
                span_id=(
                    span.span_id if part_index == 0
                    else f"{span.span_id}:identity{part_index:02d}"
                ),
                starts_seconds=round(opens, 3),
                ends_seconds=round(closes, 3),
            ))
    return tuple(surviving)


def _replace_identity_material(item: Any, **updates: Any) -> Any:
    """Update new identity fields while retaining lightweight test adapters."""

    return replace(item, **{
        name: value for name, value in updates.items() if hasattr(item, name)
    })


def _screen_material_identity(
    material: list[Any],
    spec: Any,
    *,
    client: Any,
    cache: Any,
    ledger: Any,
    library: Path,
) -> tuple[list[Any], dict[str, str], dict[str, Any]]:
    """Keep only material that can still contain the locked identity.

    Runs on the proxy, which is what the planning stages watch anyway and is
    already uploaded -- measured against the rushes this was written for, a
    640-wide proxy answers "two camera rings or three" at 0.98 confidence,
    so the screen does not need the master. The per-shot check still decodes
    the original at 1440 for the frames it authorises a track from; this only
    decides what selection is allowed to see.

    Absent means the source goes; present narrows nothing away that the card
    already offered but drops the spans no sighting overlaps. Uncertain is
    kept, because a screen that discards on doubt would quietly delete
    material and report a smaller edit as a smaller pile of rushes.
    """

    from montagewright.reference_grounding import remembered_discovery

    required = tuple(
        spec.identity_lock.framing.required_target_ids
        or [
            target.target_id
            for target in spec.identity_lock.identity.targets
        ]
    )
    if not required:
        return material, {}, {}

    kept: list[Any] = []
    aside: dict[str, str] = {}
    context: dict[str, str] = {}
    found: dict[str, Any] = {}
    print(
        f"identity screen: {len(material)} sources against "
        f"{', '.join(required)}",
        flush=True,
    )
    paid = 0
    total = len(material)
    for index, item in enumerate(material, start=1):
        proxy = getattr(item, "proxy", None)
        if proxy is None or not Path(proxy).exists():
            kept.append(item)
            continue
        ledger.check()
        try:
            screened = remembered_discovery(
                proxy, spec,
                client=client, cache=cache, ledger=ledger,
                library=library, target_ids=required,
            )
        except BudgetSpent:
            raise
        except Exception as error:  # noqa: BLE001 -- reported, not swallowed
            # One source nobody could answer for is not a reason to abandon
            # the screen, and it is certainly not a reason to lose the film:
            # keeping it means selection may still offer it and the per-shot
            # check still has to prove it before anything is tracked.
            print(
                f"  {item.source_id} — not screened: "
                f"{type(error).__name__}: {error}"[:160],
                flush=True,
            )
            kept.append(item)
            continue
        if screened is None:
            kept.append(item)
            continue
        discovery, usage = screened
        paid += 1 if usage is not None else 0
        found[item.source_id] = discovery
        # Seventy-four of these is minutes of nothing on screen, which reads
        # as a hang to anyone watching the run rather than the terminal.
        if usage is not None:
            print(f"  screen {index}/{total}  {item.source_id}", flush=True)
        absent = [
            summary for summary in discovery.target_summaries
            if summary.target_id in required and summary.verdict == "absent"
        ]
        absent_targets = tuple(sorted({one.target_id for one in absent}))
        # Sightings are in milliseconds from the first decoded frame, which
        # is the clock the card's segments are on too.
        windows_by_target = tuple(
            (target_id, _merge_identity_windows([
                (candidate.start_ms / 1000.0, candidate.end_ms / 1000.0)
                for candidate in discovery.candidates
                if candidate.target_id == target_id
                and candidate.identity_status != "hard_negative"
            ]))
            for target_id in required
        )
        seen = _merge_identity_windows([
            window
            for _, target_windows in windows_by_target
            for window in target_windows
        ])
        unresolved = set(required) - set(absent_targets) - {
            target_id for target_id, target_windows in windows_by_target
            if target_windows
        }
        identity_update = {
            "identity_windows_by_target": windows_by_target,
            "identity_absent_targets": absent_targets,
        }
        if not seen:
            if unresolved:
                kept.append(_replace_identity_material(item, **identity_update))
                continue
            context[item.source_id] = "; ".join(
                f"{one.target_id} is not in this source: {one.reason}"
                for one in absent
            ) or "the locked identities are absent from this source"
            kept.append(_replace_identity_material(
                item, carries_identity=False, **identity_update
            ))
            continue
        # Any overlap at all was enough to keep a span, and selection is
        # free to cut anywhere inside one -- so a span that clipped a
        # sighting by a fifth of a second was offered whole, the cut landed
        # in the part where the target is not, and the shot died four
        # stages later with the frames judged and no target in them. Keep
        # the part of the span the identity was actually seen in.
        surviving = _spans_inside_identity_windows(item.spans, seen)
        if not surviving:
            if unresolved:
                kept.append(_replace_identity_material(item, **identity_update))
                continue
            # Seen, but never in a stretch anything can be cut from. Same
            # treatment: it is still material, it is just not the subject.
            context[item.source_id] = (
                "the locked identity was seen in this source but never "
                "inside a usable span"
            )
            kept.append(_replace_identity_material(
                item, carries_identity=False, **identity_update
            ))
            continue
        kept.append(_replace_identity_material(
            item, spans=surviving, carries_identity=True, **identity_update
        ))
    print(
        f"  {len(kept) - len(context)} sources carry the identity, "
        f"{len(context)} kept for context only, {len(aside)} set aside "
        f"({paid} newly screened, ${ledger.spent_usd:.4f} so far)",
        flush=True,
    )
    return kept, aside, found


def _focus_note(source: Path, library: Path, duration: float) -> str:
    """One line about this take's focus, or nothing if it holds.

    Measured locally off the original, cached by content hash, and never
    sent as a verdict: the planner is told which seconds the lens was on and
    decides what to do about it, exactly as it does with camera motion.
    """

    try:
        from montagewright.focus import cached as focus_for, describe

        return describe(focus_for(source, library), 0.0, duration)
    except Exception:
        return ""


def _travel(source: Path, target_aspect: float) -> tuple[float, float]:
    """Horizontal and vertical room to move, as fractions of the frame.

    Not the proxy, for the same reason `push_room` is not: this looked like a
    question about shape and is a question about resolution. A crop only has
    to be as tall as the delivery, so what is left over is room to move -- and
    a 640-wide proxy has nothing left over and would report that no clip
    anywhere can be tilted.
    """

    from montagewright.executor import delivery_size
    from montagewright.measure.media import probe_video
    from montagewright.reframe import travel_room

    try:
        shape = probe_video(source).video
        wide, tall = int(shape.display_width), int(shape.display_height)
    except Exception:
        return 0.0, 0.0
    if not wide or not tall:
        return 0.0, 0.0
    out_w, out_h = delivery_size(target_aspect)
    return travel_room(
        source_width=wide, source_height=tall, target_aspect=target_aspect,
        output_width=out_w, output_height=out_h,
    )


def _push_room(proxy: Path, target_aspect: float) -> float:
    """How far this source can be pushed into, as a zoom factor.

    Read off the file that will actually be cut, so it is a fact about this
    clip rather than a rule about clips. A 4K take at 9:16 has room for
    about 1.5x; a 1080 one has none, and saying so is what stops a push
    being asked for where it cannot be given.

    Not the proxy: that is 640 pixels wide and would report that nothing
    anywhere can be pushed into.
    """

    from montagewright.executor import delivery_size
    from montagewright.measure.media import probe_video
    from montagewright.reframe import zoom_budget

    try:
        shape = probe_video(proxy).video
        wide = int(shape.display_width)
        tall = int(shape.display_height)
    except Exception:
        return 1.0
    if not wide or not tall:
        return 1.0
    out_w, out_h = delivery_size(target_aspect)
    budget = zoom_budget(
        source_width=wide, source_height=tall, source_aspect=wide / tall,
        target_aspect=target_aspect, output_width=out_w, output_height=out_h,
    )
    return round(1.0 / max(budget, 1e-6), 2)


def _make_proxy(
    source: Path, destination: Path, *, library: Path | None = None
) -> Path:
    """A small copy for the model to watch.

    Sending 4K masters would cost more than the rest of the run put together
    and tell the model nothing extra: it is judging what is in the frame, not
    how sharp it is.

    A proxy is a pure function of the bytes it was made from, so it is kept
    where the cards it feeds are kept -- named for those bytes, shared across
    runs. It used to live in the output directory, which meant a second cut
    of the same rushes re-encoded seventy-four 4K files before it could ask
    the first question. The run still gets one under the source's own name,
    because everything downstream looks it up that way; it is just a link
    now.
    """

    if destination.exists():
        return destination
    destination.parent.mkdir(parents=True, exist_ok=True)

    if library is not None:
        from montagewright.uploads import content_hash

        kept = library / "proxies" / f"{content_hash(source)[:20]}.mp4"
        if kept.exists():
            try:
                destination.hardlink_to(kept)
            except OSError:
                shutil.copy2(kept, destination)
            return destination
        kept.parent.mkdir(parents=True, exist_ok=True)
        _encode_proxy(source, kept)
        try:
            destination.hardlink_to(kept)
        except OSError:
            shutil.copy2(kept, destination)
        return destination

    _encode_proxy(source, destination)
    return destination


def _encode_proxy(source: Path, destination: Path) -> None:
    """Shrink the frame, keep the clock.

    This forced 15fps for a while, which nothing asked for. Gemini samples
    video at one frame a second whatever it is given, so the extra frames
    were never looked at; SAM tracks the original rather than this file, at
    its own rate; and the only thing left watching a proxy at full speed is
    the web preview, which is happier at the source rate anyway.

    What the resampling did cost was a clock that no longer matched. At
    15fps a duration has to land on a multiple of 1/15, so a 12.012s take
    came out 12.133s -- and cards describe this file while the edit cuts the
    original, which left two clocks a tenth of a second apart for no reason.
    Dropping the flag makes them the same to the millisecond, for about nine
    percent more bytes and no extra encoding time.

    The width is capped rather than set, because `scale=640` is a demand and
    not a limit: handed a 320x240 clip it produced a 640x480 one, larger than
    the file it came from and blurrier than the picture it describes. Nothing
    is gained by enlarging a proxy -- the API caps each video frame at 70
    tokens whatever it is sent, so the extra pixels are discarded before the
    model ever sees them.

    `-2` rounds the height to the nearest even number, which H.264 requires,
    so the aspect can shift by up to one pixel. 16:9 and 4:3 divide cleanly
    and come out exact; the worst measured case is a 3840x1600 source at
    0.25%, since a short proxy gives that one pixel more to be worth. It only
    reaches anything through the card's own subject boxes, and only when
    those are used without a client to ask again -- the grounding and
    tracking path never touches this file.
    """

    subprocess.run(
        [
            "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
            "-i", str(source), "-vf", "scale='min(640,iw)':-2",
            "-c:v", "libx264", "-crf", "30", "-preset", "veryfast",
            "-c:a", "aac", "-b:a", "64k", str(destination),
        ],
        check=True,
    )


def _tee_output(destination: Path) -> None:
    """Send everything printed to the terminal and to a file.

    Line buffered, because the point is to be readable while the run is
    still going.
    """

    destination.parent.mkdir(parents=True, exist_ok=True)
    handle = destination.open("a", encoding="utf-8", buffering=1)

    class Both:
        def __init__(self, first, second):
            self.first, self.second = first, second

        def write(self, text):
            self.first.write(text)
            try:
                self.second.write(text)
            except (OSError, ValueError):
                pass
            return len(text)

        def flush(self):
            self.first.flush()
            try:
                self.second.flush()
            except (OSError, ValueError):
                pass

    sys.stdout = Both(sys.stdout, handle)
    sys.stderr = Both(sys.stderr, handle)


def _make_findable(output: Path) -> None:
    """Let the interface list a cut written anywhere.

    It only scans its own runs folder, so `--output ~/cut` -- which is what
    the README tells people to type -- produced a film, a report and two
    timelines that nothing could open. Rather than teach every path in the
    interface that a run might live elsewhere, the runs folder gets a link
    pointing at it, and everything downstream carries on believing the
    layout it already believes.
    """

    from montagewright.webapp import RUNS_ROOT

    root = RUNS_ROOT.expanduser()
    if output.is_relative_to(root):
        return
    try:
        root.mkdir(parents=True, exist_ok=True)
        stem = output.parent.name if output.name == "out" else output.name
        folder = root / (stem or "cut")
        count = 2
        while folder.exists() and not (folder / "out").is_symlink():
            folder = root / f"{stem}-{count}"
            count += 1
        folder.mkdir(parents=True, exist_ok=True)
        link = folder / "out"
        if link.is_symlink():
            link.unlink()
        if not link.exists():
            link.symlink_to(output, target_is_directory=True)
    except OSError:
        # Not being listable is not a reason to refuse to cut.
        pass


def command_render(args: argparse.Namespace) -> int:
    rushes = args.rushes.expanduser().resolve()
    output = args.output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)

    grounding_spec = getattr(args, "grounding_spec", None)
    simple_description = str(
        getattr(args, "grounding_target_description", "") or ""
    ).strip()
    simple_references = tuple(
        Path(path) for path in (getattr(args, "grounding_reference", None) or [])
    )
    if grounding_spec is not None and simple_description:
        raise SystemExit(
            "use either --grounding-spec or the simple grounding target fields"
        )
    if simple_description:
        from montagewright.reference_grounding import build_reference_grounding_spec

        try:
            built = build_reference_grounding_spec(
                output / "work" / "simple-grounding-input.json",
                target_id=(
                    str(getattr(args, "grounding_target_id", "") or "").strip()
                    or "target.primary"
                ),
                target_description=simple_description,
                positive_images=simple_references,
                identity_cues=tuple(
                    getattr(args, "grounding_identity_cue", None) or ()
                ),
                stable_exclusions=tuple(
                    getattr(args, "grounding_exclusion", None) or ()
                ),
                negative_images=tuple(
                    Path(one)
                    for one in (getattr(args, "grounding_negative", None) or ())
                ),
                created_by="cli_user",
            )
        except (OSError, ValueError) as error:
            raise SystemExit(f"invalid grounding references: {error}") from error
        grounding_spec = Path(str(built.source_path))
    elif simple_references:
        raise SystemExit(
            "--grounding-reference requires --grounding-target-description"
        )
    args.grounding_spec = grounding_spec
    args.reference_grounding_spec = None
    if grounding_spec is not None:
        try:
            (
                args.grounding_spec,
                args.reference_grounding_spec,
            ) = prepare_grounding_spec_artifact(
                grounding_spec, output / "work" / "grounding-spec.json"
            )
        except (OSError, ValueError) as error:
            raise SystemExit(f"invalid grounding spec: {error}") from error

    # What produced this, written before anything is attempted. The
    # interface lists every cut in the runs folder and could only see the
    # ones it had started itself, so a render from the command line left a
    # report, a film and two timelines that nothing could open. Written at
    # the end it would have covered only the runs that finished -- and the
    # ones worth finding are the others: three died partway today, each
    # after paying for cards, direction and selection, each resumable from
    # what was already on disk, and none of them openable.
    # Everything printed also goes beside the output, so a run started here
    # can be read in the interface. The note said which command produced the
    # run and the interface could offer to resume it -- but with no log there
    # was nothing on the page except the word "中斷", which reads as broken
    # rather than as unfinished.
    _tee_output(output / "run.log")

    # The tracker was opt-in even when the model was already in this repo.
    # That made CLI renders silently less accurate than Web UI renders. Keep
    # one resolved value for the command record and every review round.
    args.sam_checkpoint = _sam_checkpoint_for(args)
    if args.sam_checkpoint is not None:
        try:
            args.sam_checkpoint = _preflight_sam_checkpoint(
                args.sam_checkpoint
            )
        except (ImportError, OSError, RuntimeError) as error:
            raise SystemExit(
                f"SAM local preflight failed before paid planning: {error}"
            ) from error

    recorded_argv = _command_with_canonical_grounding(
        list(getattr(args, "_argv", sys.argv[1:])),
        getattr(args, "grounding_spec", None),
    )
    reference_grounding_spec = args.reference_grounding_spec
    grounding_sha256 = (
        reference_grounding_spec.definition_sha256()
        if reference_grounding_spec is not None else None
    )
    (output / "command.json").write_text(
        json.dumps({
            "source": str(rushes),
            "sam_tracking": bool(args.sam_checkpoint),
            "sam_checkpoint": (
                str(args.sam_checkpoint) if args.sam_checkpoint else None
            ),
            "grounding_spec": (
                str(args.grounding_spec) if args.grounding_spec else None
            ),
            "grounding_spec_sha256": grounding_sha256,
            "command": [sys.executable, "-u", "-m", "montagewright.cli"]
            + recorded_argv,
        }, ensure_ascii=False),
        encoding="utf-8",
    )
    _make_findable(output)

    work = output / "work"
    # Cards and transcripts describe the material, so they belong to the
    # material rather than to one attempt at cutting it. Keeping them beside
    # the output meant every new run over the same rushes paid for them again
    # -- forty-four cents of cards before anything was decided.
    library = (args.library or default_library()).expanduser()
    client = _client()
    cache = UploadCache.load(args.upload_cache or default_cache_path())
    ledger = Ledger(
        cap_usd=args.budget,
        model_id=MODEL_ID,
        journal_path=output / "spend-events.jsonl",
    )

    sources_paths = sorted(
        path for path in rushes.iterdir() if path.suffix in VIDEO_SUFFIXES
    )
    if not sources_paths:
        raise SystemExit(f"no video files in {rushes}")
    print(f"{len(sources_paths)} clips in {rushes.name}", flush=True)

    # Something already cut is one file holding many takes. Handing it over
    # whole means one card for five minutes, one transcript, and a planner
    # choosing windows out of a single source as though the cuts inside it
    # were not there -- so a long file is opened along the boundaries it
    # already has. A continuous take comes back as itself.
    rushes_paths: list[Path] = []
    for path in sources_paths:
        spans = (
            shots_in(path)
            if _duration(path) >= SPLIT_ABOVE_SECONDS
            else [(0.0, 0.0)]
        )
        if len(spans) < 2:
            rushes_paths.append(path)
            continue
        print(
            f"{path.name}: already cut, opening into {len(spans)} shots",
            flush=True,
        )
        pieces = work / "shots"
        pieces.mkdir(parents=True, exist_ok=True)
        for index, (start, end) in enumerate(spans):
            piece = pieces / f"{path.stem}-{index:02d}{path.suffix}"
            if not piece.exists():
                subprocess.run(
                    [
                        "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
                        "-ss", f"{start:.3f}", "-to", f"{end:.3f}",
                        "-i", str(path), "-c:v", "libx264", "-crf", "18",
                        "-preset", "veryfast", "-c:a", "aac", str(piece),
                    ],
                    check=True,
                )
            rushes_paths.append(piece)
    sources_paths = rushes_paths

    # A subset, chosen the same way every time. Random would mean a fresh set
    # of cards on every run, which is the cost this exists to avoid; evenly
    # spaced means the sample is not all of one setup, which taking the first
    # N would be -- rushes arrive in shooting order.
    if args.sample and 0 < args.sample < len(sources_paths):
        step = len(sources_paths) / args.sample
        sources_paths = [
            sources_paths[min(len(sources_paths) - 1, int(i * step))]
            for i in range(args.sample)
        ]
        print(
            f"sampling {len(sources_paths)} of {len(rushes_paths)} clips",
            flush=True,
        )

    proxies = {
        path.stem: _make_proxy(
            path, work / "proxies" / f"{path.stem}.mp4", library=library
        )
        for path in sources_paths
    }
    # The originals, by source id. How far a shot can be pushed into is a
    # fact about the file that will be cut, and the proxy is 640 pixels wide
    # -- asking it says every clip has no room at all.
    originals = {path.stem: path for path in sources_paths}
    def wrote(index: int, total: int, source_id: str) -> None:
        print(f"  card {index}/{total}  {source_id}", flush=True)

    # The half the model cannot see. Measured off the original rather than
    # the proxy -- shake survives scaling but the proxy was re-encoded, and
    # this is about the file that will be cut. Content addressed, so seventy
    # four clips are measured once ever, about three seconds each.
    from montagewright.motion import cached as motion_for

    def motion_of(source_id: str):
        original = originals.get(source_id)
        if original is None:
            return None
        try:
            return motion_for(original, _duration(original), library)
        except Exception as error:
            print(f"  {source_id} — no motion measured: {error}"[:120], flush=True)
            return None

    print(f"measuring camera motion across {len(proxies)} clips", flush=True)
    try:
        cards, stats = build_library(
            proxies, library / "cards", client=client, cache=cache,
            progress=wrote, motion_of=motion_of, ledger=ledger,
        )
    except BudgetSpent as error:
        # Say it here rather than let the traceback speak: the rushes that
        # were described are cached under their own content hashes, so
        # resuming after topping up costs only the ones still missing.
        print(
            f"\nstopped before the library was complete: {error}\n"
            "described clips are cached; resume once the budget allows",
            flush=True,
        )
        raise
    print(
        f"cards: {stats['written']} written, {stats['reused']} reused, "
        f"{stats['failed']} failed  (${ledger.spent_usd:.4f})",
        flush=True,
    )
    for line in stats.get("failures", [])[:5]:
        print(f"  card failed — {line[:140]}", flush=True)

    # Only where the speech is the content. A transcript costs a call and a
    # minute per clip, and on b-roll it answers a question nobody asked --
    # so the card, which already watched the clip with its audio, says which
    # ones need one rather than a flag somebody has to remember.
    transcripts: dict[str, dict] = {}
    speaking = [
        source_id for source_id in proxies
        if (load_card(cards[source_id]) if source_id in cards else {})
        and (load_card(cards[source_id]) or {}).get("speech") == "content"
    ]
    if speaking and args.speech != "never":
        from montagewright.transcript import describe as transcribe
        from montagewright.transcript import load as load_transcript
        from montagewright.transcript import save as save_transcript

        print(
            f"speech: {len(speaking)} clips carry it as content",
            flush=True,
        )
        for source_id in speaking:
            destination = (
                library / "transcripts"
                / f"{content_hash(proxies[source_id])[:20]}.json"
            )
            card = load_transcript(destination)
            if card is None:
                # Before the call, not after: a transcript on a long clip is
                # one of the more expensive things here, and the cap is meant
                # to stop work rather than to describe it afterwards.
                ledger.check()
                try:
                    card, usage = transcribe(
                        proxies[source_id], client=client,
                        locale=args.locale, cache=cache,
                        # The model watches the proxy, which is already
                        # uploaded; the recogniser runs here and has no
                        # reason to listen to a 64 kbps re-encode of a file
                        # sitting next to it.
                        audio=originals.get(source_id),
                        ledger=ledger,
                    )
                except BudgetSpent:
                    # Money running out says nothing about this clip.  Treating
                    # it as missing speech freezes a partial, false inventory
                    # and lets later paid stages keep trying.
                    raise
                except Exception as error:
                    print(
                        f"  {source_id} — no transcript: "
                        f"{type(error).__name__}: {error}"[:150],
                        flush=True,
                    )
                    continue
                save_transcript(card, destination)
            transcripts[source_id] = card
        print(f"  transcribed, running total ${ledger.spent_usd:.4f}", flush=True)

    # Measured once per source rather than twice per field, and off the
    # original: how far a crop can travel is a fact about resolution.
    _room = {
        source_id: _travel(originals.get(source_id, proxy), ASPECTS[args.aspect])
        for source_id, proxy in proxies.items()
    }
    material = []
    # Which takes were set aside before anything was planned, and why. The
    # card gives a reason and it was being dropped on the floor, so a run
    # that quietly worked from sixty-six of seventy-four clips looked
    # identical to one that had all of them -- and "why didn't it use the
    # good coin shot" had no answer anywhere in the output.
    set_aside: dict[str, str] = {}
    confirmed_identities: dict[str, Any] = {}
    identity_confirmation_outcomes: dict[
        tuple[str, str], dict[str, str]
    ] = {}
    sightings: dict[str, Any] = {}
    for source_id, proxy in proxies.items():
        card = load_card(cards[source_id]) if source_id in cards else None
        if card is not None and not card.get("usable", True):
            set_aside[source_id] = str(
                card.get("unusable_reason") or "no reason given"
            )
            continue
        # The card says where each subject is; a previous run's tracker has
        # measured how wide the thing the crop follows actually is. Price the
        # object that will be cropped, not the one the description framed.
        subject_boxes = tracked_geometry.applied(
            subjects_from_card(card or {}), cards.get(source_id),
        )
        material.append(
            MaterialItem(
                source_id=source_id,
                duration_seconds=_duration(proxy),
                summary=(card or {}).get("summary", ""),
                proxy=proxy,
                composition=(card or {}).get("composition", ""),
                # Derived, not asked. Whether the camera moved is a fact
                # about the pixels and was being answered by a model reading
                # one frame a second -- which is the rate at which a moving
                # camera and a still one look the same. It was optional too,
                # so a missing answer became "still" on the field that
                # decides whether a digital move gets stacked on a take that
                # already has one.
                camera_moves=any(
                    one.state == "moving" for one in (motion_of(source_id) or [])
                ),
                camera_motion=str((card or {}).get("camera_motion", "") or ""),
                shot_size=str((card or {}).get("shot_size", "") or ""),
                facing=str((card or {}).get("facing", "") or ""),
                focus_note=_focus_note(
                    originals.get(source_id, proxy), library, _duration(proxy)
                ),
                spans=tuple(
                    spans_of(card, source_id, _duration(proxy))
                ),
                push_room=_push_room(
                    originals.get(source_id, proxy), ASPECTS[args.aspect]
                ),
                pan_room=_room[source_id][0],
                tilt_room=_room[source_id][1],
                action=tuple(
                    f"`{beat.beat_id}` {beat.what} "
                    f"{beat.starts_seconds:.1f}-{beat.ends_seconds:.1f}s"
                    for beat in action_beats(card or {})[:4]
                ),
                action_ids=tuple(
                    beat.beat_id
                    for beat in action_beats(card or {})[:4]
                ),
                action_windows=tuple(
                    (beat.beat_id, beat.starts_seconds, beat.ends_seconds)
                    for beat in action_beats(card or {})[:4]
                ),
                needs=tuple(
                    f"{entry.get('what')}（{entry.get('why', '')[:60]}）"
                    for entry in (card or {}).get("needs", [])
                ),
                subjects=tuple(
                    _subject_line(box, _aspect(proxy), ASPECTS[args.aspect])
                    for box in subject_boxes
                ),
                subject_geometry=tuple(
                    (
                        box.label, box.entity_id, box.centre_x, box.centre_y,
                        box.width, box.height,
                    )
                    for box in subject_boxes
                ),
                # The label a look will name, beside the moment it was seen
                # and what the camera did over the take. Two facts already
                # on the card and in the measurement; the planner is the
                # first place that can put them together.
                sightings=tuple(
                    (box.label, box.at_seconds)
                    for box in subjects_from_card(card or {})
                ),
                motion=tuple(motion_of(source_id) or ()),
                crop_width=min(1.0, ASPECTS[args.aspect] / _aspect(proxy)),
                speech=_speech_lines(source_id, transcripts.get(source_id)),
                audio_spans=tuple(
                    (
                        span_id,
                        float(span["in_seconds"]),
                        float(span["out_seconds"]),
                    )
                    for span_id, span in (
                        _audio_spans_for_source(
                            source_id, transcripts[source_id]
                        )
                        if transcripts.get(source_id)
                        else {}
                    ).items()
                ),
            )
        )

    # Which sources actually contain the locked identity is a question about
    # the material, and it belongs here -- beside "is this take usable" --
    # rather than after a whole film has been planned around them. The
    # methodology this project started from says it in one line: pick the
    # right object first, then talk about grounding. Selection reads text
    # cards and watches 640-wide proxies with the reference images attached,
    # which is enough to choose a shot and not enough to tell two rear
    # cameras from three; the run that made this necessary planned eight
    # shots, laid them on the music, and only then discovered that its
    # close-up of "the rear camera module" was a different model. Its
    # alternate was too.
    if args.reference_grounding_spec is not None:
        material, identity_aside, sightings = _screen_material_identity(
            material,
            args.reference_grounding_spec,
            client=client,
            cache=cache,
            ledger=ledger,
            library=library,
        )
        set_aside.update(identity_aside)
        # Exact boxes are deliberately deferred until direction has bought a
        # bounded primary/alternate pool.  Screening protects recall; paying
        # to locate every positive source before knowing whether the edit can
        # use it was the largest avoidable identity cost.

    if set_aside:
        print(
            f"set aside: {len(set_aside)} clips the cards called unusable",
            flush=True,
        )
        for source_id, why in list(set_aside.items())[:5]:
            print(f"  {source_id} — {why[:110]}", flush=True)

    from montagewright.brief import load_brief

    brief_document = load_brief(args.brief)
    # The copy manifest is data for the graphics track, not instructions to
    # the edit planner. The creative prose keeps the exact legacy behaviour
    # when no manifest exists.
    brief = brief_document.creative_brief
    if brief_document.approved_copy:
        approved_copy = work / "approved-copy.json"
        approved_copy.write_text(
            json.dumps({
                "brief_sha256": brief_document.sha256,
                "facts": [
                    fact.model_dump(mode="json")
                    for fact in brief_document.approved_copy
                ],
            }, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    else:
        # A reused run directory must not retain copy approval from an older
        # brief after the fenced manifest has been removed.
        (work / "approved-copy.json").unlink(missing_ok=True)
    brief_candidates = work / "brief-candidates.json"
    if brief_document.candidates or brief_document.instructions:
        brief_candidates.write_text(
            json.dumps(
                brief_document.candidates_json(), ensure_ascii=False, indent=2
            ),
            encoding="utf-8",
        )
    else:
        brief_candidates.unlink(missing_ok=True)

    # Music is one planning fact, measured once and shared by every stage.
    # Previously direction heard the track, selection received only a prose
    # memory of it, and the exact BeatGrid did not exist until both decisions
    # had already been paid for.  Measuring it here lets content selection
    # refer to the same stable cue IDs that local rhythm resolution executes.
    if args.music_map:
        grid = load_beat_grid(args.music_map)
    elif args.music:
        grid = None
        music_key = _asked(
            "runtime-music-analysis-v1",
            sha256_file(Path(args.music).expanduser().resolve(strict=True)),
        )
        remembered_music = _decided(work, "music-analysis", music_key)
        if remembered_music is not None:
            try:
                grid = beat_grid_from_payload(remembered_music)
                print("music analysis: reused from the last attempt", flush=True)
            except (KeyError, TypeError, ValueError):
                remembered_music = None
        if grid is None:
            print("no music map given; measuring the track", flush=True)
            grid = analyse_track(args.music)
            _decide(work, "music-analysis", music_key, beat_grid_payload(grid))
    else:
        grid = None
        print("no music; lengths will be led by content", flush=True)

    grounding_target_refs = tuple(
        target.target_id
        for target in (
            args.reference_grounding_spec.identity_lock.identity.targets
            if args.reference_grounding_spec is not None
            else ()
        )
    )
    planning_state = material_planning_state(
        material,
        {
            source_id: load_card(path)
            for source_id, path in cards.items()
            if path.exists()
        },
        story_obligations=tuple(
            instruction.text for instruction in brief_document.instructions
        ),
        coverage_obligations=tuple(
            f"recognizably_present:{target_id}"
            for target_id in grounding_target_refs
        ),
        music_cue_refs=tuple(
            cue.cue_id
            for cue in (grid.cues if grid is not None else ())
            if cue.kind in {"section_boundary", "downbeat", "accent"}
        ),
        grounding_target_refs=grounding_target_refs,
    )
    publish_planning_state(
        work,
        planning_state,
        request={
            "stage": "material_inventory",
            "brief_sha256": brief_document.sha256,
            "aspect": args.aspect,
            "target_seconds": float(args.seconds or 0.0),
        },
        response={"material_digest": planning_state.material_digest},
        validation={
            "valid": True,
            "available": sum(
                one.disposition == "available" for one in planning_state.spans
            ),
            "deferred": sum(
                one.disposition == "deferred" for one in planning_state.spans
            ),
        },
    )
    # What was decided has to be keyed on everything it was decided from.
    # This was source ids, brief, aspect and the music path -- so a card
    # rewritten with better segments, a span boundary moved, or a schema
    # changed left the same key, and the next run reused a selection made
    # against material that no longer exists in that shape. The card version
    # already moves when the card's shape does, and the span ids move when
    # its answers do; both belong here.
    # Hash the request the model actually sees, not a hand-picked subset of
    # fields. This includes card/transcript text through ``_describe_one`` and
    # the exact proxy bytes through their content hashes.
    catalogue = _asked(*(
        f"{item.source_id}|{CARD_VERSION}|"
        f"{content_hash(item.proxy) if item.proxy else 'no-proxy'}|"
        + ",".join(
            f"{one.span_id}:{one.starts_seconds:.3f}-{one.ends_seconds:.3f}"
            for one in item.spans
        ) + "|"
        f"{_describe_one(item)}"
        for item in sorted(material, key=lambda one: one.source_id)
    ))
    music_key = (
        content_hash(args.music_map)
        if args.music_map is not None and args.music_map.exists()
        else content_hash(args.music)
        if args.music is not None and args.music.exists()
        else "no-music"
    )
    direction_contract = _planning_contract(
        "direction_zh-TW.txt", _direction_schema(
            [span.span_id for item in material for span in item.spans],
            list(grounding_target_refs),
            list(dict.fromkeys(
                action_id for item in material for action_id in item.action_ids
            )),
        )
    )
    asked = _asked(
        catalogue, brief, args.aspect, music_key,
        f"seconds={args.seconds or 0}",
        f"duration_mode={args.duration_mode}", direction_contract,
        (
            args.reference_grounding_spec.definition_sha256()
            if args.reference_grounding_spec is not None
            else "no-reference-grounding"
        ),
    )
    direction = _decided(work, "direction", asked)
    migrated_direction_key = False
    if direction is None:
        # v5 replaced natural-language subject labels with source-scoped
        # v01/v02 ids. A paid Direction written immediately before that
        # migration belongs to this same run/brief/material but naturally has
        # the previous schema key. Re-open only that narrow artifact shape;
        # resolve_candidate_commitments still validates every span, action,
        # label migration and geometry before it can be used. This is not a
        # general stale-cache bypass.
        try:
            saved_direction = json.loads(
                (work / "direction.json").read_text(encoding="utf-8")
            )
            legacy_key = str(saved_direction.get("key") or "")
            legacy_value = _decided(work, "direction", legacy_key)
            legacy_options = (
                legacy_value.get("candidate_options") or []
                if legacy_value is not None else []
            )
            if (
                legacy_value is not None
                and legacy_options
                and all("required_visuals" in one for one in legacy_options)
                and all("required_evidence" not in one for one in legacy_options)
            ):
                direction = legacy_value
                migrated_direction_key = True
                print(
                    "direction: migrating the paid label-based visual contract "
                    "to stable local ids",
                    flush=True,
                )
        except (OSError, json.JSONDecodeError, AttributeError):
            pass
    if direction is None:
        ledger.check()
        direction, usage_direction = decide_direction(
            material, brief=brief, aspect=args.aspect, music=args.music,
            music_grid=grid,
            seconds=args.seconds, duration_mode=args.duration_mode,
            cache=cache, client=client, ledger=ledger,
            grounding_spec=args.reference_grounding_spec,
        )
        _decide(work, "direction", asked, direction)
    else:
        print("direction: reused from the last attempt", flush=True)
    print(
        f"direction: {direction['target_seconds']:.0f}s {direction['aspect']}, "
        f"{len(direction.get('unusable', []))} ruled out",
        flush=True,
    )
    # The direction's rulings belong on the same list as the cards'. Both
    # answer "why is this take not in the film", selection is handed only
    # what survives both, and only one of them was being written down -- so
    # a run that ruled out five clips reported none set aside and seventy-four
    # in play, and the interface had no answer for where they went.
    #
    # The reasons are better than the cards' as well: a card sees one clip
    # and can only say it failed, while the direction has seen all of them
    # and can say which take supersedes this one.
    # Only what the direction called broken leaves the run. A take it merely
    # ranked below another is still material -- the comparison travels with
    # it into selection instead of deleting it, because the two are not the
    # same claim and only one of them is about the whole file.
    from montagewright.planner import _beaten_and_broken

    beaten, broken = _beaten_and_broken(direction)
    # The screen is a filter, not a judge. It reads a 640-pixel proxy at a
    # frame a second, and when direction -- which watched the same clip --
    # promises the locked product from a source the screen called absent,
    # the disagreement belongs to the per-shot check that decodes the master
    # at 1440 and looks at the actual frames. Refusing it here argued with a
    # model that was right about a table of three handsets, twice, at the
    # price of a paid correction each time, and then ended the run.
    promoted: list[str] = []
    promoted_pairs: set[tuple[str, str]] = set()
    if args.reference_grounding_spec is not None and grounding_target_refs:
        wanted_pairs = {
            (
                str(option.get("span_id") or "").split(":")[0],
                str(option.get("target_id") or "none"),
            )
            for option in (direction.get("candidate_options") or [])
            if str(option.get("target_id") or "none") in set(grounding_target_refs)
        }
        promoted_pairs = {
            (source_id, target_id)
            for source_id, target_id in wanted_pairs
            for item in material
            if item.source_id == source_id
            and (
                not item.carries_identity
                or target_id in set(item.identity_absent_targets)
            )
        }
        promoted = sorted({source_id for source_id, _ in promoted_pairs})
        if promoted:
            material = [
                replace(
                    item,
                    carries_identity=True,
                    identity_absent_targets=tuple(
                        target_id for target_id in item.identity_absent_targets
                        if (item.source_id, target_id) not in promoted_pairs
                    ),
                    identity_windows_by_target=tuple(
                        (
                            target_id,
                            () if (item.source_id, target_id) in promoted_pairs
                            else windows,
                        )
                        for target_id, windows in item.identity_windows_by_target
                    ),
                ) if item.source_id in promoted else item
                for item in material
            ]
            print(
                "  direction says the identity is in "
                + ", ".join(sorted(promoted))
                + ", which the screen did not see; the per-shot check will "
                "settle it",
                flush=True,
            )
    def bind_commitments(answer):
        resolved_commitments = resolve_candidate_commitments(
            answer,
            material,
            material_digest=planning_state.material_digest,
            aspect=args.aspect,
            target_seconds=float(args.seconds or answer["target_seconds"]),
            grounding_target_ids=grounding_target_refs,
            grounding_sha256=(
                args.reference_grounding_spec.definition_sha256()
                if args.reference_grounding_spec is not None else None
            ),
            excluded_source_ids=tuple(sorted(broken)),
        )
        offered_count = sum(
            len(item.spans) for item in material if item.source_id not in broken
        )
        needed, _ = _shot_count_bounds(answer, offered_count)
        unique = len({
            option.commitment_id for option in resolved_commitments.options
        })
        if needed is not None and unique < needed:
            # Say which unit is short and what closes the gap. Told only
            # that it "supplied 7 per-shot commitments" against a pacing
            # needing 8, the correction added a second option to a slot it
            # already had and dropped another slot entirely -- ending with
            # the same seven options and six distinct commitments. The
            # array is called candidate_options, so adding an option looks
            # like adding a commitment unless the difference is spelled
            # out.
            warning = (
                f"the pacing prefers at least {needed} shots, so it prefers at "
                f"least {needed} different commitment_id values, and there "
                f"are {unique}: "
                + ", ".join(sorted({
                    option.commitment_id
                    for option in resolved_commitments.options
                }))
                + f". Add {needed - unique} more commitment_id(s) -- new "
                "shot slots with their own purpose, each with exactly one "
                "primary. Adding another option to a commitment_id that "
                "already exists is an alternate for that same shot and does "
                "not raise this count. Delivering the available commitments "
                "with an explicit duration advisory."
            )
            payload = resolved_commitments.model_dump(mode="python")
            payload["warnings"] = tuple(dict.fromkeys([
                *resolved_commitments.warnings, warning,
            ]))
            resolved_commitments = CandidateCommitments.model_validate(payload)
        return resolved_commitments

    # Provider-valid JSON can still make a claim the local material cannot
    # prove. Allow two bounded corrections; never weaken the local contract or
    # spin indefinitely when the provider repeats a structurally invalid plan.
    correction_fault: CommitmentError | None = None
    for correction in range(3):
        try:
            commitments = bind_commitments(direction)
            if migrated_direction_key:
                _decide(work, "direction", asked, direction)
            break
        except CommitmentError as error:
            correction_fault = error
            if correction == 2:
                raise
            ledger.check()
            correction_key = _asked(
                json.dumps(direction, ensure_ascii=False, sort_keys=True),
                planning_state.material_digest,
                str(error),
                "candidate-correction-text-v1",
                (
                    args.reference_grounding_spec.definition_sha256()
                    if args.reference_grounding_spec is not None
                    else "no-reference-grounding"
                ),
            )
            corrected = _decided(
                work, f"commitment-correction-{correction}", correction_key
            )
            if corrected is None:
                direction, usage_direction = correct_candidate_options(
                    direction,
                    material,
                    fault=str(error),
                    grounding_target_ids=tuple(grounding_target_refs),
                    excluded_source_ids=frozenset(broken),
                    client=client,
                    ledger=ledger,
                )
                _decide(
                    work, f"commitment-correction-{correction}",
                    correction_key, direction,
                )
            else:
                direction = corrected
            beaten, broken = _beaten_and_broken(direction)
    else:  # pragma: no cover - the bounded loop either binds or raises.
        raise correction_fault or CommitmentError("commitment correction failed")

    # Direction has now reduced the screen-positive pool to the sources that
    # can actually serve a primary or alternate commitment. Prove only that
    # pool. Context-only options name target=none and never buy an identity
    # box; a failed primary can still fall through to its already-confirmed
    # alternate without starting another planning call.
    if args.reference_grounding_spec is not None:
        identity_sources = _identity_commitment_sources(commitments)
        normal = [
            item for item in material
            if item.source_id in identity_sources
            and item.source_id not in set(promoted)
        ]
        promoted_material = [
            item for item in material if item.source_id in set(promoted)
        ]
        confirmed_identities.update(_confirm_material_identity(
            normal, sightings, args.reference_grounding_spec,
            masters=originals,
            client=client, cache=cache, ledger=ledger, library=library,
            work=work,
            outcomes=identity_confirmation_outcomes,
        ))
        confirmed_identities.update(_confirm_material_identity(
            promoted_material, sightings, args.reference_grounding_spec,
            masters=originals,
            client=client, cache=cache, ledger=ledger, library=library,
            work=work, spread=True,
            outcomes=identity_confirmation_outcomes,
        ))
        commitments = _commitments_without_exact_hard_negatives(
            commitments, identity_confirmation_outcomes,
            require_confirmation=promoted_pairs,
        )
    # Keep the paid full Direction immutable. Candidate corrections have
    # their own content-addressed artifacts above and are never allowed to
    # overwrite the decision that watched all rushes and heard the music.
    publish_planning_state(
        work,
        planning_state,
        stage=f"commitments-{commitments.sha256()[:16]}",
        request={
            "stage": "candidate_commitments",
            "direction_key": asked,
            "material_digest": planning_state.material_digest,
        },
        response=commitments.model_dump(mode="json"),
        validation={
            "valid": True,
            "warnings": list(commitments.warnings),
        },
    )
    for entry in direction.get("unusable", []) or []:
        source_id = str(entry.get("source_id", ""))
        if source_id in broken:
            set_aside[source_id] = str(entry.get("reason") or "no reason given")
    if beaten:
        print(
            f"  {len(beaten)} more the direction ranked below another take, "
            "kept and passed on",
            flush=True,
        )

    offered_ids = [
        span.span_id
        for item in material
        if item.source_id not in broken
        for span in item.spans
    ]
    min_shots, max_shots = _shot_count_bounds(direction, len(offered_ids))
    selection_contract = _planning_contract(
        "selection_zh-TW.txt",
        _selection_schema(
            offered_ids, min_shots=min_shots, max_shots=max_shots,
            graphic_candidate_ids=[
                one.candidate_id
                for one in brief_document.graphics_candidates()
            ],
            audio_span_ids=list(_audio_spans(transcripts)),
            grounding_target_ids=(
                [
                    target.target_id
                    for target in args.reference_grounding_spec.identity_lock.identity.targets
                ]
                if args.reference_grounding_spec is not None else []
            ),
            commitment_ids=list(dict.fromkeys(
                option.commitment_id for option in commitments.options
            )),
            action_ids=list(dict.fromkeys(
                action_id for item in material for action_id in item.action_ids
            )),
        ),
    )
    chose = _asked(
        asked,
        json.dumps(direction, sort_keys=True, ensure_ascii=False),
        commitments.sha256(),
        selection_contract,
        # Local semantic validators are part of the executable answer's
        # meaning even though JSON Schema cannot encode their cross-field
        # rules. Bump this when those rules change so a paid answer accepted
        # by an older binary is audited again instead of bypassing the new
        # Selection repair loop on resume.
        "selection-local-contract-v7-canonical-camera-duration",
    )
    provider_selection = _decided(work, "selection", chose)
    selection_to_repair: dict[str, Any] | None = None
    if provider_selection is None:
        # Migrate the immediately preceding contract without throwing away a
        # paid edit. The v7 audit below decides whether it can be reused or is
        # the starting point for a scoped repair; successful output is saved
        # under v7 and this compatibility read disappears naturally.
        legacy_chose = _asked(
            asked,
            json.dumps(direction, sort_keys=True, ensure_ascii=False),
            commitments.sha256(),
            selection_contract,
            "selection-local-contract-v8-direction-bound-action",
        )
        provider_selection = _decided(work, "selection", legacy_chose)
    if provider_selection is not None:
        # Keep the paid answer on disk as the audit record, but make the one
        # unambiguous monotonic duration repair on a copy before deciding that
        # the whole 18-shot edit needs another paid Selection pass.
        executable_cached_selection = copy.deepcopy(provider_selection)
        # Cache and fresh answers must enter the same local contract graph.
        # Newly added immutable action bounds live on commitments, not on an
        # older provider artifact; project them before the source-window
        # solver runs or resume will repeat an EDL failure that fresh planning
        # already knows how to avoid.
        cached_repairs = normalize_selection(
            executable_cached_selection, material, commitments=commitments
        )
        cached_selection_faults = audit_cached_selection(
            executable_cached_selection,
            material,
            direction,
            commitments=commitments,
            grounding_spec=args.reference_grounding_spec,
            duration_mode=args.duration_mode,
        )
        if cached_selection_faults:
            print(
                "selection: cached answer failed the current local audit; "
                "asking Selection to repair it\n  - "
                + "\n  - ".join(cached_selection_faults),
                flush=True,
            )
            selection_to_repair = executable_cached_selection
            provider_selection = None
        else:
            provider_selection = executable_cached_selection
            if cached_repairs:
                print(
                    "selection: fitted executable shot timing locally; "
                    "all shots and commitments are unchanged",
                    flush=True,
                )
    if provider_selection is None:
        saved_attempt = (
            _decided(work, "selection-attempt", chose)
            or _latest_decision(work, "selection-attempt")
        )
        if saved_attempt is not None and selection_to_repair is None:
            candidate = saved_attempt.get("selection")
            if isinstance(candidate, dict):
                selection_to_repair = copy.deepcopy(candidate)
                print(
                    "selection: recovered the latest paid provider attempt "
                    "for local validation before asking again",
                    flush=True,
                )
    if provider_selection is None:
        # SelectionUnrenderable preserves the last paid editorial answer for
        # Web inspection.  It used to be write-only from the CLI's point of
        # view, so every resume paid Gemini to recreate the same 18 shots --
        # even when a newer local contract could now normalize and accept the
        # saved answer. Re-audit the content-addressed draft first and only
        # ask the provider when substantive faults remain.
        blocked_draft = (
            _decided(work, "invalid-selection-draft", chose)
            or _latest_decision(work, "invalid-selection-draft")
        )
        if blocked_draft is not None:
            recovered = copy.deepcopy(blocked_draft)
            for local_only in (
                "invalid_selection_faults", "delivery_status", "draft_only",
            ):
                recovered.pop(local_only, None)
            normalize_selection(recovered, material, commitments=commitments)
            recovered_faults = audit_cached_selection(
                recovered,
                material,
                direction,
                commitments=commitments,
                grounding_spec=args.reference_grounding_spec,
                duration_mode=args.duration_mode,
            )
            if not recovered_faults:
                provider_selection = recovered
                _decide(work, "selection", chose, provider_selection)
                print(
                    "selection: recovered the paid draft after current local "
                    "contract validation; no provider repair needed",
                    flush=True,
                )
            elif selection_to_repair is None:
                selection_to_repair = recovered
                print(
                    "selection: the saved paid draft still needs a scoped "
                    "editorial decision after local normalization\n  - "
                    + "\n  - ".join(recovered_faults),
                    flush=True,
                )
    if provider_selection is None:
        ledger.check()
        def record_selection_attempt(
            draft: dict[str, Any], faults: tuple[str, ...], attempt: int,
        ) -> None:
            _decide(work, "selection-attempt", chose, {
                "attempt": attempt,
                "selection": draft,
                "faults": list(faults),
            })

        try:
            provider_selection, usage_selection = select_shots(
                material, direction, brief=brief, cache=cache, client=client,
                ledger=ledger,
                graphic_candidates=brief_document.graphics_candidates(),
                grounding_spec=args.reference_grounding_spec,
                music_grid=grid,
                commitments=commitments,
                duration_mode=args.duration_mode,
                initial_selection=selection_to_repair,
                attempt_recorder=record_selection_attempt,
            )
        except SelectionUnrenderable as error:
            # Keep the paid editorial answer visible without confusing it
            # with the executable Selection cache. Resume must still repair
            # it, while Web can show exactly where this run stopped.
            draft = copy.deepcopy(error.draft)
            _annotate_selection_identity_evidence(
                draft, confirmed_identities, identity_confirmation_outcomes
            )
            _annotate_selection_direction_motion(draft, commitments)
            draft["invalid_selection_faults"] = list(error.faults)
            draft["delivery_status"] = "release_blocked"
            draft["draft_only"] = True
            _decide(work, "invalid-selection-draft", chose, draft)
            raise
        _annotate_selection_identity_evidence(
            provider_selection, confirmed_identities, identity_confirmation_outcomes
        )
        _decide(work, "selection", chose, provider_selection)
    else:
        print("selection: reused from the last attempt", flush=True)
        _annotate_selection_identity_evidence(
            provider_selection, confirmed_identities, identity_confirmation_outcomes
        )
    _annotate_selection_direction_motion(provider_selection, commitments)
    resolved_selection_key = _asked(
        chose,
        json.dumps(provider_selection, sort_keys=True, ensure_ascii=False),
        "resolved-selection-v2-camera-rest-fit-identity-needs-review-hold",
    )
    selection = _decided(
        work, "resolved-selection", resolved_selection_key
    )
    if selection is None:
        # The provider answer is an audit record.  Local execution repairs
        # live in their own artifact so they can resume without rewriting or
        # pretending the paid answer said something it did not.
        selection = copy.deepcopy(provider_selection)
    else:
        print("selection: reused locally resolved execution plan", flush=True)
    cached_sequence_faults = sequence_disagreements(selection.get("shots") or [])
    if cached_sequence_faults:
        raise RuntimeError(
            "cached selection repeats overlapping adjacent source windows; "
            "refusing to render: " + "; ".join(cached_sequence_faults)
        )
    travelling = sum(
        1 for shot in selection["shots"] if str(shot.get("frame", "")) == "travels"
    )
    source_motion = sum(
        1 for shot in selection["shots"]
        if delivered_camera_intent_of(shot) == "use_source_motion"
    )
    deliberate_holds = sum(
        1 for shot in selection["shots"]
        if delivered_camera_intent_of(shot) == "hold"
    )
    print(
        f"selection: {len(selection['shots'])} shots; {travelling} digital "
        f"crop moves, {source_motion} using source camera motion, "
        f"{deliberate_holds} deliberate holds",
        flush=True,
    )
    planning_state = selection_planning_state(
        planning_state, selection.get("shots") or []
    )
    publish_planning_state(
        work,
        planning_state,
        allow_incomplete_rollover=True,
        request={
            "stage": "selection",
            "direction_key": asked,
            "selection_key": chose,
        },
        response={
            "selected_span_ids": list(planning_state.selected_span_ids),
            "alternate_span_ids": list(planning_state.alternate_span_ids),
        },
        validation={
            "valid": True,
            "pool_size": len(planning_state.spans),
            "selected_count": len(planning_state.selected_span_ids),
        },
    )
    # A plan whose prose and whose structure describe different shots is not
    # a rendering problem -- both halves came from the same answer, so it
    # says the planner was of two minds and the film will follow the looks.
    disagreed: list[str] = list(selection.get("frame_disagreements") or [])
    for note in disagreed:
        print(f"  {note}", flush=True)

    edl, snaps = _edl_from_selection(
        selection, rushes, cards, transcripts=transcripts, library=library,
        material=material,
    )
    camera_rest_repairs = _fit_camera_rests_to_shot(selection, edl)
    if camera_rest_repairs:
        selection.setdefault("duration_repairs", []).extend(camera_rest_repairs)
        for note in camera_rest_repairs:
            print(f"  {note}", flush=True)
        _decide(work, "resolved-selection", resolved_selection_key, selection)
        edl, snaps = _edl_from_selection(
            selection, rushes, cards, transcripts=transcripts, library=library,
            material=material,
        )
    from montagewright.planning_release import resolve_preferred_camera_durations

    edl, camera_resolutions = resolve_preferred_camera_durations(
        edl, duration_mode=args.duration_mode
    )
    for note in camera_resolutions:
        print(f"  camera duration resolution: {note}", flush=True)
    if snaps:
        print(f"cut on action: {len(snaps)} in-points moved", flush=True)
    found = {path.stem: path for path in sources_paths}
    audio_source_ids = {audio.source_id for audio in edl.audio_clips}
    sources = {
        shot["source_id"]: probe(shot["source_id"], found[shot["source_id"]])
        for shot in selection["shots"]
        if shot["source_id"] in found
    }
    sources.update({
        source_id: probe(source_id, found[source_id])
        for source_id in audio_source_ids
        if source_id in found and source_id not in sources
    })
    rhythm_context = _rhythm_context(selection, cards, material=material)

    aspect = ASPECTS[args.aspect]

    def cut(edl, sources, rhythm_context):
        """Render this plan. One definition, because a replan renders again.

        It was written out twice, and only the first copy learned to keep the
        voice -- so a run that revised anything delivered the same film with
        the speech thrown away and nothing but the bed left.
        """

        return run(
            edl,
            sources,
            grid,
            output,
            target_aspect=aspect,
            intent=(
                direction["direction"]
                + "\n節奏密度：目標約 "
                + f"{direction.get('target_shot_count', len(edl.clips))} 顆，"
                + f"典型 {direction.get('typical_shot_seconds', 0):.1f}s，"
                + "純靜態通常不超過 "
                + f"{direction.get('max_static_seconds', 0):.1f}s。"
                + str(direction.get("pacing_reason", ""))
            ),
            brief=brief,
            rhythm_context=rhythm_context,
            target_seconds=float(direction["target_seconds"]),
            duration_mode=args.duration_mode,
            max_static_seconds=float(direction.get("max_static_seconds") or 0.0),
            music=args.music,
            cards=cards,
            checkpoint=args.sam_checkpoint,
            ledger=ledger,
            keep_voice=bool(transcripts),
            under_speech=str(direction.get("music_under_speech") or "duck"),
            client=client,
            transcripts=transcripts,
            grounding_spec=args.reference_grounding_spec,
            grounding_memory=library / "reference-grounding",
            confirmed_identities=confirmed_identities,
            # The pass that decides how long each shot runs should be able
            # to see the shots. They are the material items selection chose,
            # in the order it chose them.
            rhythm_shots=[
                item for shot in selection["shots"]
                for item in material
                if item.source_id == shot.get("source_id")
            ],
            source_motion_measurements={
                item.source_id: item.motion for item in material
            },
            upload_cache=cache,
        )

    # A shot whose identity cannot be proved is one review item, not a reason
    # to throw away the draft.  Do not silently replace it with another model
    # choice: that made repeated repairs expensive and, worse, repeatedly
    # selected the same attractive lookalike.  Keep the editor's timing, make
    # the picture a safe untracked hold, and let the report/Web editor say
    # exactly which shot needs a human replacement.
    identity_swaps: list[str] = []
    for attempt in range(3):
        try:
            result, plan, report, resolved = cut(edl, sources, rhythm_context)
            break
        except ReferenceShotsUnusable as unproved:
            if attempt == 2:
                raise
            repaired = 0
            for fault in unproved.faults:
                index = next(
                    (
                        at for at, shot in enumerate(selection["shots"])
                        if f"k{at:02d}" == fault.clip_id
                    ),
                    None,
                )
                if index is None:
                    continue
                shot = selection["shots"][index]
                failed_looks = [
                    look for look in shot.get("looks") or []
                    if look.get("entity_id") == fault.entity_id
                ]
                if not failed_looks:
                    continue
                issue = _why(fault)
                shot["identity_status"] = "needs_review"
                shot.setdefault("identity_target_id", fault.entity_id)
                prior_issue = str(shot.get("identity_issue") or "")
                shot["identity_issue"] = "; ".join(
                    one for one in (prior_issue, issue) if one
                )
                shot["delivered_camera_intent"] = "hold"
                shot["frame"] = "settles"
                for look in failed_looks:
                    look["entity_id"] = "none"
                identity_swaps.append(
                    f"{fault.clip_id}: {shot.get('span_id')} could not prove "
                    f"{fault.entity_id} ({issue}); kept as an untracked draft "
                    "shot and marked needs_review for manual replacement"
                )
                repaired += 1
                print(f"  {identity_swaps[-1]}", flush=True)
            if not repaired:
                raise
            _decide(
                work, "resolved-selection", resolved_selection_key, selection
            )
            edl, snaps = _edl_from_selection(
                selection, rushes, cards, transcripts=transcripts, library=library,
                material=material,
            )
            sources = {
                one["source_id"]: probe(one["source_id"], found[one["source_id"]])
                for one in selection["shots"]
                if one["source_id"] in found
            }
            sources.update({
                source_id: probe(source_id, found[source_id])
                for source_id in {audio.source_id for audio in edl.audio_clips}
                if source_id in found and source_id not in sources
            })
            rhythm_context = _rhythm_context(selection, cards, material=material)
    else:  # pragma: no cover - the bounded loop either cuts or raises.
        raise RuntimeError("identity recovery did not converge")
    _project_track_confirmed(selection, report)
    _decide(work, "resolved-selection", resolved_selection_key, selection)
    report.plan_disagreements.extend(identity_swaps)
    if grid is not None:
        from montagewright.measure.storage import write_json

        write_json(
            work / "graphics-beat-grid.json",
            beat_grid_payload(
                grid.as_heard(plan.music_from_seconds, plan.music_spans)
            ),
        )
    # The direction set a length; somebody has to compare it with what came
    # out. Three layers each made a defensible call last run and delivered
    # 17.9 seconds against 30, with nothing in the report saying so.
    report.target_seconds = float(direction["target_seconds"])
    # Held from before the report existed. Kept rather than printed: these
    # went to stdout and nowhere else, so the one that mattered -- a shot
    # naming a subject its own window never reaches -- was on screen while
    # the same shot was replanned twice into the same failure.
    report.plan_disagreements.extend(
        note for note in disagreed if note not in report.plan_disagreements
    )

    rounds: list[Round] = []
    shot_verdicts: dict[str, dict] = {}
    # Carried across iterations because both gates need it: the one at the
    # top of the loop decides whether a replanned cut gets looked at again,
    # and it would otherwise re-read a verdict from before the replan and
    # stop on it.
    undelivered = 0
    stopped = "review not requested"
    # Whatever happens in here, the account of the run still gets written.
    # It used to be written once at the very end, so a crash anywhere after
    # the render discarded everything the run had already decided and paid
    # for: one left a finished film, a review round and three replans on
    # disk with no report, and the interface reads that as a run that
    # produced nothing but a video.
    #
    # "The report is a deliverable as much as the file is" was already the
    # rule here. A report that exists only when nothing went wrong is a
    # trophy. The error is named in the report and printed in full, so this
    # records a failure rather than swallowing one.
    crashed: BaseException | None = None
    try:
      if args.review:
          # Every round renders before it reviews, so a stopping condition
          # always leaves a finished film rather than a half-planned one.
          while True:
              keep_going, stopped = should_continue(
                  rounds, ledger=ledger, undelivered=undelivered
              )
              if not keep_going:
                  break
              # Two questions, two viewings. The shots say whether each one did
              # what its plan promised; the cut says whether the eight of them
              # are a film. Only the second was ever asked, and it is the one
              # that cannot see a clipped wordmark going past in three seconds.
              try:
                  shot_verdicts = review_shots(
                      {
                          f"k{index:02d}": path
                          for index, path in enumerate(result.segment_paths)
                      },
                      selection["shots"],
                      seconds={
                          clip_id: entry["seconds"]
                          for clip_id, entry in report.rhythm_decisions.items()
                      },
                      degradations=report.degradations,
                      client=client,
                      cache=cache,
                      ledger=ledger,
                  )
              except BudgetSpent as error:
                  stopped = str(error)
                  break
              missed = [
                  f"{clip_id}: {entry.get('note', '')}"
                  for clip_id, entry in shot_verdicts.items()
                  if not entry.get("delivered", True)
              ]
              print(
                  f"shots: {len(shot_verdicts) - len(missed)}/"
                  f"{len(shot_verdicts)} delivered what they planned",
                  flush=True,
              )
              for line in missed:
                  print(f"  {line[:110]}", flush=True)
              try:
                  verdict = review_cut(
                      result.preview,
                      brief=brief,
                      direction=direction["direction"],
                      wanted_seconds=report.target_seconds or 0.0,
                      delivered_seconds=report.delivered_seconds or 0.0,
                      already=rounds,
                      client=client,
                      cache=cache,
                      ledger=ledger,
                  )
              except BudgetSpent as error:
                  stopped = str(error)
                  break
              rounds.append(
                  Round(
                      index=len(rounds) + 1,
                      verdict=verdict,
                      actionable=actionable_keys(verdict),
                  )
              )
              print(
                  f"review {len(rounds)}: {verdict.verdict} "
                  f"({len(verdict.issues)} issues) — {verdict.overall[:70]}",
                  flush=True,
              )
              audio_issues = [
                  issue for issue in verdict.issues
                  if issue.severity in {"major", "blocking"}
                  and issue.issue_type == "audio_content"
              ]
              if audio_issues:
                  # A wrong sentence is not repaired by replacing the image
                  # at its timecode. Re-run the joint selection pass so the
                  # canonical transcript span and the B-roll covering it can
                  # change together, then render before the next review.
                  feedback = "\n".join(
                      f"- {issue.description}；請改成：{issue.fix}"
                      for issue in audio_issues
                  )
                  try:
                      ledger.check()
                      selection, _ = select_shots(
                          material,
                          direction,
                          brief=(
                              brief
                              + "\n\n## 上一版聲音審核未通過\n"
                              + feedback
                          ),
                          cache=cache,
                          client=client,
                          ledger=ledger,
                          graphic_candidates=brief_document.graphics_candidates(),
                          grounding_spec=args.reference_grounding_spec,
                          music_grid=grid,
                          commitments=commitments,
                      )
                      edl, snaps = _edl_from_selection(
                          selection, rushes, cards, transcripts=transcripts,
                          library=library, material=material,
                      )
                      sources = {
                          shot["source_id"]: probe(
                              shot["source_id"], found[shot["source_id"]]
                          )
                          for shot in selection["shots"]
                          if shot["source_id"] in found
                      }
                      sources.update({
                          audio.source_id: probe(
                              audio.source_id, found[audio.source_id]
                          )
                          for audio in edl.audio_clips
                          if audio.source_id in found
                          and audio.source_id not in sources
                      })
                      rhythm_context = _rhythm_context(
                          selection, cards, material=material
                      )
                      result, plan, report, resolved = cut(
                          edl, sources, rhythm_context
                      )
                  except BudgetSpent as error:
                      stopped = str(error)
                      break
                  report.target_seconds = float(direction["target_seconds"])
                  undelivered = len(audio_issues)
                  continue
              # Worked out before the gate, not after it. The shot reviewer's
              # findings used to be computed on the far side of an early
              # return, so an approving film reviewer threw them away
              # unread -- and the loop it gates is the only thing that can
              # act on them.
              # The reviewer says what it saw; the disagreement says what
              # made it inevitable. A shot told only "the pan does not pan"
              # was replanned twice into the same pan, because the reason it
              # could not pan -- the far end of the sweep is six seconds
              # past where this shot cuts away -- was on stdout and nowhere
              # the planner could read.
              said, mandatory = _replan_diagnostics(
                  report.plan_disagreements,
                  report.degradations,
                  shot_verdicts,
              )
              failing = [
                  (
                      index,
                      shot,
                      "；".join(
                          [shot_verdicts[f"k{index:02d}"].get("note", "")]
                          + said.get(f"k{index:02d}", [])
                      ),
                  )
                  for index, shot in enumerate(selection["shots"])
                  if f"k{index:02d}" in mandatory
              ]
              # A whole-cut fault the per-shot pass did not raise still names
              # a shot: the reviewer's timecode maps to whatever is on screen
              # then. Without this a "revise" verdict over dust on a screen or
              # a cropped reference object found nothing to replan and shipped
              # the fault. Only major and blocking issues -- a minor note does
              # not spend a round -- and the shot's own duration decides which
              # one 0:10 fell in.
              by_index = {row[0]: row for row in failing}
              for issue in verdict.issues:
                  if issue.severity not in {"major", "blocking"}:
                      continue
                  named = issue.clip_id
                  if not named and issue.at_seconds is not None:
                      named = _shot_at_second(
                          issue.at_seconds, report.rhythm_decisions
                      )
                  if not named or not named.startswith("k"):
                      continue
                  try:
                      idx = int(named[1:])
                  except ValueError:
                      continue
                  if not 0 <= idx < len(selection["shots"]):
                      continue
                  note = f"整片審核指出：{issue.description}。建議：{issue.fix}"
                  if idx in by_index:
                      i, shot, old = by_index[idx]
                      by_index[idx] = (i, shot, f"{old}；{note}")
                  else:
                      by_index[idx] = (idx, selection["shots"][idx], note)
              failing = [by_index[i] for i in sorted(by_index)]
              undelivered = len(failing)
              keep_going, stopped = should_continue(
                  rounds, ledger=ledger, undelivered=undelivered
              )
              if not keep_going:
                  break
              if not failing:
                  # The cut reviewer wants a change nobody can point at a shot.
                  stopped = (
                      "revision asked for, but no shot was named as undelivered"
                  )
                  break
              sequence_context = "\n".join(
                  f"第 {index + 1} 顆（span={shot.get('span_id')}，"
                  f"source={shot['source_id']}，"
                  f"intent={shot.get('camera_intent', 'hold')}，"
                  f"約 {shot.get('seconds_needed', 0)} 秒）："
                  f"{shot.get('why', '')}"
                  for index, shot in enumerate(selection["shots"])
              )
              try:
                  ledger.check()
                  replanned, usage = replan_shots(
                      failing,
                      material,
                      direction,
                      brief=brief,
                      context=sequence_context,
                      cache=cache,
                      client=client,
                      ledger=ledger,
                      grounding_spec=args.reference_grounding_spec,
                      commitments=commitments,
                  )
              except BudgetSpent as error:
                  stopped = str(error)
                  break
              fresh = replanned.get("shots", [])
              if len(fresh) != len(failing):
                  stopped = (
                      f"replan returned {len(fresh)} shots for "
                      f"{len(failing)} that needed one"
                  )
                  break
              expected_replacements = {
                  f"k{index:02d}": (index, old, note)
                  for index, old, note in failing
              }
              fresh_by_id = {
                  str(new.get("replace_clip_id", "")): new for new in fresh
              }
              if set(fresh_by_id) != set(expected_replacements):
                  stopped = "replan did not identify each replacement clip exactly once"
                  break
              candidate = list(selection["shots"])
              for clip_id, (index, _, _) in expected_replacements.items():
                  new = fresh_by_id[clip_id]
                  candidate[index] = new
              sequence_notes = sequence_disagreements(candidate)
              if sequence_notes:
                  # One paid retry with the structural fault made explicit.
                  # Shipping a repeated window is worse than keeping the last
                  # reviewed cut, but a single correction usually resolves it.
                  try:
                      ledger.check()
                      replanned, usage = replan_shots(
                          failing,
                          material,
                          direction,
                          brief=brief,
                          context=(
                              sequence_context
                              + "\n\n上次替換不可接受："
                              + "；".join(sequence_notes)
                          ),
                          cache=cache,
                          client=client,
                          ledger=ledger,
                          grounding_spec=args.reference_grounding_spec,
                          commitments=commitments,
                      )
                  except BudgetSpent as error:
                      stopped = str(error)
                      break
                  fresh = replanned.get("shots", [])
                  if len(fresh) != len(failing):
                      stopped = "sequence repair returned the wrong shot count"
                      break
                  fresh_by_id = {
                      str(new.get("replace_clip_id", "")): new for new in fresh
                  }
                  if set(fresh_by_id) != set(expected_replacements):
                      stopped = "sequence repair did not identify every clip"
                      break
                  candidate = list(selection["shots"])
                  for clip_id, (index, _, _) in expected_replacements.items():
                      new = fresh_by_id[clip_id]
                      candidate[index] = new
                  sequence_notes = sequence_disagreements(candidate)
                  if sequence_notes:
                      stopped = "replan repeated an adjacent source window twice"
                      disagreed.extend(sequence_notes)
                      report.plan_disagreements.extend(sequence_notes)
                      break
              for clip_id, (index, old, _) in expected_replacements.items():
                  new = fresh_by_id[clip_id]
                  print(
                      f"  replan k{index:02d}: {old['source_id']} "
                      f"{move_of_shot(old)} → {new['source_id']} "
                      f"{move_of_shot(new)} — {new.get('why', '')[:70]}",
                      flush=True,
                  )
                  selection["shots"][index] = new
              for note in replanned.get("frame_disagreements") or []:
                  print(f"  {note}", flush=True)
              disagreed.extend(
                  (replanned.get("frame_disagreements") or []) + sequence_notes
              )
              # Everything downstream is rebuilt from the amended selection, so
              # the next round renders a different film rather than re-reading
              # the same one.
              edl, snaps = _edl_from_selection(
                  selection, rushes, cards, transcripts=transcripts,
                  library=library, material=material,
              )
              sources = {
                  shot["source_id"]: probe(
                      shot["source_id"], found[shot["source_id"]]
                  )
                  for shot in selection["shots"]
                  if shot["source_id"] in found
              }
              sources.update({
                  audio.source_id: probe(
                      audio.source_id, found[audio.source_id]
                  )
                  for audio in edl.audio_clips
                  if audio.source_id in found
                  and audio.source_id not in sources
              })
              rhythm_context = _rhythm_context(
                  selection, cards, material=material
              )
              try:
                  ledger.check()
                  result, plan, report, resolved = cut(
                      edl, sources, rhythm_context
                  )
              except BudgetSpent as error:
                  stopped = str(error)
                  break
              report.target_seconds = float(direction["target_seconds"])
              report.plan_disagreements.extend(
                  note for note in disagreed
                  if note not in report.plan_disagreements
              )
          if rounds:
              report.degradations = adjudicate(
                  report.degradations, rounds[-1].verdict, shot_verdicts
              )

    except BudgetSpent as error:
        stopped = str(error)
        print(f"\nstopped: {stopped}", flush=True)
    except Exception as error:  # noqa: BLE001 -- re-raised below, after the report
        import traceback

        crashed = error
        stopped = f"{type(error).__name__}: {error}"[:400]
        traceback.print_exc()

    final_selected_spans = tuple(dict.fromkeys(
        str(shot.get("span_id") or "")
        for shot in selection.get("shots") or [] if shot.get("span_id")
    ))
    if crashed is None and final_selected_spans != planning_state.selected_span_ids:
        planning_state = selection_planning_state(
            planning_state, selection.get("shots") or []
        )
        publish_planning_state(
            work,
            planning_state,
            stage="edit",
            allow_incomplete_rollover=True,
            request={
                "stage": "reviewed_selection",
                "selection_key": chose,
                "review_rounds": len(rounds),
            },
            response={
                "selected_span_ids": list(planning_state.selected_span_ids),
                "alternate_span_ids": list(planning_state.alternate_span_ids),
            },
            validation={
                "valid": True,
                "reviewed": True,
                "selected_count": len(planning_state.selected_span_ids),
            },
        )

    _write_report(
        output,
        snaps=snaps,
        set_aside=set_aside,
        material_ids=[item.source_id for item in material],
        rounds=rounds,
        shot_verdicts=shot_verdicts,
        stopped_because=stopped,
        direction=direction,
        selection=selection,
        report=report,
        plan=plan,
        result=result,
        usages=[],
    )
    # The same selection call may identify useful on-screen Brief moments.
    # Materialise those as editable drafts for the Web/CLI graphics backend;
    # never overwrite a plan the editor has already touched, and never let
    # the model's choice promote ordinary prose into approved copy.
    graphics_path = work / "graphics.json"
    if crashed is None and plan is not None and not graphics_path.exists():
        from montagewright.brief import initial_graphics_plan

        initial_graphics = initial_graphics_plan(
            brief_document,
            selection,
            shot_durations=[segment.duration_seconds for segment in plan.segments],
        )
        if initial_graphics.cues:
            graphics_path.write_text(
                json.dumps(
                    initial_graphics.model_dump(mode="json"),
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )
    if crashed is not None:
        print(
            "\nthe report was written before this run gave up, so what it "
            "did decide is on disk",
            flush=True,
        )
    print(f"\n{report.summary()}", flush=True)
    for stage, usd in sorted(
        report.spend().get("by_stage", {}).items(), key=lambda kv: -kv[1]
    ):
        print(f"  {stage:12s} ${usd:.4f}", flush=True)
    # The words, once there is a cut for them to sit on. Written from the
    # same lines the timeline shows, so what was corrected in the interface
    # is what gets burned.
    if transcripts and args.subtitles != "none":
        from montagewright.transcript import (
            against_audio_assignments, against_windows, to_srt,
            windows_against_segments, words_against_audio_assignments,
            words_against_windows,
        )

        # The render plan is the picture's clock. Selection can differ after
        # action snapping, beat grounding or source clamping; rebuilding the
        # subtitle clock from it made captions drift exactly on those cuts.
        subtitle_windows = windows_against_segments(plan.segments)
        if plan.audio_assignments:
            said = against_audio_assignments(plan.audio_assignments, transcripts)
            subtitle_words = words_against_audio_assignments(
                plan.audio_assignments, transcripts
            )
        else:
            said = against_windows(subtitle_windows, transcripts)
            subtitle_words = words_against_windows(subtitle_windows, transcripts)
        try:
            from montagewright.subtitles import as_cues

            wide, tall = plan.output_size
            said = as_cues(
                said, args.aspect, wide, tall, words=subtitle_words,
            )
        except Exception:
            pass
        if said:
            (output / "subtitles.srt").write_text(
                to_srt(said, with_speaker=True), encoding="utf-8"
            )
            print(f"subtitles   {output / 'subtitles.srt'}", flush=True)
            if args.subtitles == "burn":
                from montagewright import subtitles as typeset
                from montagewright.subtitles import NoFontHere, burn

                if args.subtitle_font:
                    typeset.CHOSEN = str(args.subtitle_font.expanduser())

                try:
                    burned = burn(
                        result.deliverable, said,
                        output / "deliverable-subtitled.mp4",
                        aspect=args.aspect, work=work / "subs",
                        style=typeset.look(args.subtitle_look),
                        words=subtitle_words,
                    )
                    print(f"burned in   {burned}", flush=True)
                except NoFontHere as error:
                    # The cut is finished either way; say what is missing
                    # rather than failing a render over a font.
                    print(f"not burned  {error}", flush=True)

    print(f"deliverable {result.deliverable}", flush=True)
    print(f"preview     {result.preview}", flush=True)

    # Off unless asked for. A rendered file is what most runs want; a
    # timeline is for the run where somebody intends to open it and disagree
    # with one shot, and writing one every time is clutter in every other.
    if args.timeline != "none":
        from montagewright.timeline import to_fcpxml, to_xmeml

        payload = json.loads(
            (output / "report.json").read_text(encoding="utf-8")
        )
        width, height = plan.output_size
        for flavour, suffix, build in (
            ("premiere", "xml", to_xmeml),
            ("finalcut", "fcpxml", to_fcpxml),
        ):
            if args.timeline in {flavour, "both"}:
                path = output / f"timeline.{suffix}"
                laid_bed = output / "bed-as-laid.m4a"
                laid_voice = output / "voice-as-laid.m4a"
                path.write_text(
                    build(plan, payload, name=output.name,
                          width=width, height=height,
                          music=laid_bed if laid_bed.exists() else args.music,
                          voice=laid_voice if laid_voice.exists() else None,
                          graphics=(
                              output / "graphics-overlay.mov"
                              if (output / "graphics-overlay.mov").exists()
                              else None
                          )),
                    encoding="utf-8",
                )
                print(f"timeline    {path}", flush=True)

    # Everything that could still be delivered has been. The run failed all
    # the same, and says so to whoever asked.
    if crashed is not None:
        raise crashed
    return 0


def _suffix(rushes: Path, stem: str) -> str:
    for suffix in VIDEO_SUFFIXES:
        if (rushes / f"{stem}{suffix}").exists():
            return suffix
    return ".mp4"


def _speech_lines(
    source_id: str | dict, card: dict | None = None, limit: int = 40
) -> tuple[str, ...]:
    """The soundbites, as the planner needs to read them.

    A window out of an interview is chosen because of a sentence, so the
    sentence has to be visible when the choice is made -- with who said it,
    because a shot of the person not talking is the obvious way to get this
    wrong, and with its seconds, because they are what the shot's length is.
    """

    # Keep the small helper source-compatible for callers that only need the
    # prose list; production supplies the source ID so Gemini can return a
    # stable canonical span reference.
    if isinstance(source_id, dict):
        card, source_id = source_id, "source"
    if not card:
        return ()
    return tuple(
        f"`{span_id}` "
        f"{span['in_seconds']:.1f}-{span['out_seconds']:.1f}s"
        f"（{span['speaker'] or '未標'}）{span['text']}"
        + ("〔連續多行〕" if len(span["line_ids"]) > 1 else "")
        for span_id, span in _audio_spans_for_source(
            str(source_id), card, limit=limit
        ).items()
    )


def _audio_spans_for_source(
    source_id: str, card: dict, *, limit: int = 40,
    max_gap_seconds: float = 1.2, max_span_seconds: float = 14.0,
) -> dict[str, dict]:
    """Canonical source-contiguous soundbites available to the planner.

    Apple ASR may split one thought across several lines.  Exposing only the
    lines makes the planner either cut the sentence short or lay several
    independently timed clips.  Alongside every original line, expose one
    deterministic run for adjacent lines by the same speaker.  The run is a
    single source window: Gemini may choose it, but cannot reorder, rewrite,
    or splice non-adjacent words.
    """

    from montagewright.transcript import lines_of

    lines = lines_of(card)[:limit]
    made: dict[str, dict] = {}

    def add(first: int, last: int) -> None:
        chosen = lines[first:last + 1]
        span_id = (
            f"{source_id}:t{first:02d}"
            if first == last else f"{source_id}:t{first:02d}-t{last:02d}"
        )
        made[span_id] = {
            "source_id": source_id,
            "in_seconds": chosen[0].starts_seconds,
            "out_seconds": chosen[-1].ends_seconds,
            "speaker": chosen[0].speaker,
            "text": " ".join(one.text.strip() for one in chosen if one.text.strip()),
            "line_ids": [f"{source_id}:t{index:02d}" for index in range(first, last + 1)],
            "kind": "line" if first == last else "continuous_turn",
        }

    for index in range(len(lines)):
        add(index, index)

    run_start = 0
    while run_start < len(lines):
        run_end = run_start
        while run_end + 1 < len(lines):
            current, following = lines[run_end], lines[run_end + 1]
            same_speaker = (current.speaker or "") == (following.speaker or "")
            small_gap = (
                following.starts_seconds - current.ends_seconds
                <= max_gap_seconds
            )
            within_cap = (
                following.ends_seconds - lines[run_start].starts_seconds
                <= max_span_seconds
            )
            if not (same_speaker and small_gap and within_cap):
                break
            run_end += 1
        if run_end > run_start:
            add(run_start, run_end)
        run_start = run_end + 1

    return made


def _audio_spans(cards: dict[str, dict]) -> dict[str, dict]:
    """Canonical transcript spans Gemini may place on the edit timeline."""

    made: dict[str, dict] = {}
    for source_id, card in cards.items():
        made.update(_audio_spans_for_source(source_id, card))
    return made


def _subject_line(box, source_aspect: float, target_aspect: float) -> str:
    """A subject, how much of it the delivery frame can hold, and whether it
    moves.

    Selection was marking a wordmark "must be whole" on a 16:9 plate being
    delivered 9:16, where the widest possible crop covers a third of it. The
    planner was not being careless -- it had no way to know, so the promise
    was unkeepable at the moment it was made. Told the fraction, it can move
    the camera across the subject, pick a tighter shot of the same thing, or
    accept a partial view on purpose.

    Whether it moves is the same argument one step later. A frame asked to
    follow something that stands still is a hold, and the executor was
    working that out after the shot had been planned, substituting, and
    recording a degradation -- forty-five of them in one run, which is not a
    ladder of exceptions any more, it is the normal path. The card already
    answers it; the planner simply was not being told.
    """

    fits = min(1.0, (target_aspect / source_aspect) / max(box.width, 1e-9))
    facts = []
    if fits < 0.995:
        facts.append(f"交付比例裡最多只能露出 {fits * 100:.0f}%")
    facts.append(
        "會在畫面裡移動" if getattr(box, "moves", False)
        else "整段停在原地，跟著它走等於定鏡"
    )
    return f"{box.label}（{'、'.join(facts)}）"


def _shot_at_second(seconds: float, rhythm: dict[str, dict]) -> str | None:
    """Which shot covers this moment on the timeline.

    The whole-cut reviewer reports a problem at a time -- dust on a screen at
    0:10, a reference object cropped at 0:10 -- and often names no shot, so
    `clip_id` comes back null. The loop that could fix it only acts on shots
    the per-shot reviewer marked undelivered, so a real fault found by the
    one pass that watches the finished film was written down and dropped,
    and the run stopped saying "revision asked for, but no shot was named".
    The timeline knows which shot is on screen at 0:10; this is that lookup,
    from the durations already recorded per shot.
    """

    if not rhythm:
        return None
    cursor = 0.0
    for clip_id in sorted(rhythm):
        length = float(rhythm[clip_id].get("seconds") or 0.0)
        if cursor <= seconds < cursor + length:
            return clip_id
        cursor += length
    # Past the last cut -- rounding, or a note on the final frame -- belongs
    # to the last shot rather than to nothing.
    return max(rhythm)


def _replan_diagnostics(
    plan_disagreements: list[str],
    degradations: list,
    shot_verdicts: dict[str, dict],
) -> tuple[dict[str, list[str]], set[str]]:
    """Separate evidence for a replan from reasons to start one.

    Measurements are context, not failures.  A crop showing 90% of a subject
    can be worth mentioning to the shot reviewer and still exceed the 85%
    contract.  Treating every degradation as mandatory replaced most of a
    cut twice.  Structural contradictions remain mandatory, as do a shot
    reviewer saying the plan was not delivered or explicitly adjudicating a
    fallback as needing a replan.
    """

    said: dict[str, list[str]] = {}
    mandatory: set[str] = set()
    for note in plan_disagreements:
        clip_id = note.split(" ", 1)[0]
        if not clip_id.startswith("k"):
            continue
        said.setdefault(clip_id, []).append(note)
        mandatory.add(clip_id)
    for step in degradations:
        clip_id = str(step.clip_id or "")
        if not clip_id:
            continue
        measured = ", ".join(
            f"{name} {value}" for name, value in (step.measured or {}).items()
        )
        said.setdefault(clip_id, []).append(
            f"本機量到：{step.trigger}"
            + (f"（{measured}）" if measured else "")
        )
        if getattr(step, "adjudication", "unadjudicated") == "replan":
            mandatory.add(clip_id)
    for clip_id, verdict in shot_verdicts.items():
        if (
            not verdict.get("delivered", True)
            or verdict.get("degradation_verdict") == "replan"
        ):
            mandatory.add(clip_id)
    return said, mandatory


def _duration(path: Path) -> float:
    completed = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "json", str(path)],
        capture_output=True, text=True, check=True,
    )
    return float(json.loads(completed.stdout)["format"]["duration"])


def _aspect(path: Path) -> float:
    completed = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream=width,height", "-of", "json", str(path)],
        capture_output=True, text=True, check=True,
    )
    stream = json.loads(completed.stdout)["streams"][0]
    return float(stream["width"]) / float(stream["height"])


def _rhythm_context(
    selection: dict, cards: dict[str, Path], *, material: list[Any] | None = None,
) -> dict[str, dict]:
    """Why each shot exists, what moves in it, how much frame it holds.

    A length is a judgement about purpose, and the rhythm pass was being
    asked to make it from a description and an energy label.
    """

    context: dict[str, dict] = {}
    for index, shot in enumerate(selection["shots"]):
        clip_id = f"k{index:02d}"
        card = (
            load_card(cards[shot["source_id"]])
            if shot["source_id"] in cards
            else None
        )
        entry: dict[str, object] = {
            "why": shot.get("why", ""),
            "camera_intent": shot.get("camera_intent", "hold"),
            "source_motion": shot.get("source_motion_role", "locked"),
        }
        if card is not None:
            first_look = (looks_of(shot) or [None])[0]
            box = find_subject(
                card,
                subject_of(shot),
                entity_id=(first_look.entity_id if first_look else None),
            )
            if box is not None:
                entry["subject_share"] = round(box.width * box.height, 4)
        selected_action = str(shot.get("action_id") or "none")
        if selected_action != "none" and material is not None:
            from montagewright.planner import resolve_action_boundary

            boundary = resolve_action_boundary(shot, tuple(material))
            if boundary is not None:
                first, last, _, _ = boundary
                entry["action"] = (
                    f"{selected_action}: selected source action "
                    f"{first:.1f}-{last:.1f}s"
                )
        elif card is not None:
            local_action = selected_action.rsplit(":", 1)[-1]
            beats = [
                beat for beat in action_beats(card)
                if selected_action != "none" and beat.beat_id == local_action
            ]
            if beats:
                entry["action"] = "；".join(
                    f"{beat.beat_id}: {beat.what} {beat.starts_seconds:.1f}-"
                    f"{beat.ends_seconds:.1f}s"
                    for beat in beats[:3]
                )
        context[clip_id] = entry
    return context


def _look_boxes(card: dict, reframe) -> list[tuple[float, float, float]]:
    """Each look's measured place, or nothing if any of them is unknown.

    All or none: a half-known path gives a travel distance that is wrong in
    a way nobody can see, and a floor computed from it would be confidently
    too small. Unknown falls back to the move's declared minimum.
    """

    out: list[tuple[float, float, float]] = []
    for look in reframe.looks:
        box = find_subject(card, look.at, entity_id=look.entity_id)
        if box is None:
            return []
        # The crop this framing asks for, as a share of the frame. `fill`
        # closes toward the subject; the rest take what there is.
        width = min(1.0, max(0.2, box.height / 0.66)) if look.framing == "fill" else 1.0
        out.append((box.centre_x, box.centre_y, width))
    return out


def _fit_camera_rests_to_shot(
    selection: dict, edl: EDL,
) -> list[str]:
    """Fit declared multi-look rests without deleting the chosen move.

    Selection chooses both a shot duration and preferred dwell at each look.
    Once card geometry is attached, the local solver knows the actual travel
    time as well.  A small conflict between those two preferences should not
    kill an otherwise executable edit: preserve the journey and the minimum
    readable settle at every real stop, then share the remaining time in the
    same proportions Selection requested.  If even those minimum settles do
    not fit, leave the shot untouched so the structural gate still rejects it.
    """

    from montagewright.capabilities import SETTLE_SECONDS
    from montagewright.grounding import _floor_for
    from montagewright.schema import looks_of

    repairs: list[str] = []
    shots = list(selection.get("shots") or [])
    for index, (shot, clip) in enumerate(zip(shots, edl.clips)):
        reframe = clip.reframe
        if reframe is None or len(reframe.looks) < 2:
            continue
        duration = max(
            0.0, float(clip.approx_out_seconds) - float(clip.approx_in_seconds)
        )
        floor = _floor_for(clip)
        if floor <= duration + 1e-6:
            continue
        raw_looks = list(shot.get("looks") or [])
        stop_indices = [
            at for at, look in enumerate(raw_looks)
            if str(look.get("presentation_intent") or "")
            != "transition_pass"
        ]
        if len(stop_indices) < 2:
            continue
        declared = sum(
            max(0.0, float(raw_looks[at].get("seconds") or 0.0))
            for at in stop_indices
        )
        # _floor_for is travel plus the declared rests. Geometry remains the
        # same while only rest durations are shortened.
        travel = max(0.0, floor - declared)
        available_rests = duration - travel
        minimum_rests = SETTLE_SECONDS * len(stop_indices)
        if available_rests < minimum_rests - 1e-6:
            continue
        flexible = [
            max(
                0.0,
                float(raw_looks[at].get("seconds") or 0.0) - SETTLE_SECONDS,
            )
            for at in stop_indices
        ]
        extra = max(0.0, available_rests - minimum_rests)
        weight = sum(flexible)
        before = [float(raw_looks[at].get("seconds") or 0.0) for at in stop_indices]
        for position, at in enumerate(stop_indices):
            share = (
                extra * flexible[position] / weight
                if weight > 1e-9
                else extra / len(stop_indices)
            )
            raw_looks[at]["seconds"] = SETTLE_SECONDS + share
        trial_reframe = reframe.model_copy(update={"looks": looks_of(shot)})
        trial_clip = clip.model_copy(update={"reframe": trial_reframe})
        fitted = _floor_for(trial_clip)
        if fitted > duration + 1e-5:
            for at, seconds in zip(stop_indices, before):
                raw_looks[at]["seconds"] = seconds
            continue
        repairs.append(
            f"k{index:02d}: shortened look rests from "
            f"{sum(before):.2f}s to {available_rests:.2f}s so "
            f"{reframe.camera_move} keeps its travel and settles inside "
            f"the {duration:.2f}s shot"
        )
    return repairs


def _edl_from_selection(
    selection: dict, rushes: Path, cards: dict[str, Path], *,
    transcripts: dict[str, dict] | None = None,
    library: Path | None = None,
    material: list[Any] | None = None,
) -> tuple[EDL, dict[str, str]]:
    clips = []
    snaps: dict[str, str] = {}
    material_by_source = {
        str(item.source_id): item for item in (material or [])
    }

    def focus_of(source_id: str) -> list:
        """How this take's focus behaves, measured once per file ever."""

        if library is None:
            return []
        original = rushes / f"{source_id}.MP4"
        if not original.exists():
            found = next(rushes.glob(f"{source_id}.*"), None)
            if found is None:
                return []
            original = found
        try:
            from montagewright.focus import cached as focus_for

            return focus_for(original, library)
        except Exception:
            # A reading nobody could take is not a reason to refuse the cut.
            return []

    for index, shot in enumerate(selection["shots"]):
        # Text and UI have to survive whole or they say nothing. The first
        # run cropped a Galaxy Unpacked wordmark down to "y Unpacked", which
        # is worse than not using the shot. That, and everything else a
        # reframe is made of, is decided in one place now -- the rebuild had
        # its own copy and the copy was missing a field.
        reframe = reframe_of(shot)
        clip_id = f"k{index:02d}"
        start = float(shot["start_seconds"])
        # What selection said this shot needs to do its job. It used to be a
        # flat four seconds for every shot, which meant the layer that chose
        # the material had no say in how long it ran and the layer that chose
        # the length started from a constant.
        # Pre-span cached selections did not carry this field at all and keep
        # their historical four-second placeholder. A present but malformed or
        # zero field is a current contract failure and must never become 4s.
        legacy_missing_duration = (
            "seconds_needed" not in shot and "span_id" not in shot
        )
        wanted = (
            4.0 if legacy_missing_duration
            else float(shot.get("seconds_needed") or 0.0)
        )
        if wanted <= 0.0:
            raise ValueError(
                f"{clip_id} has no positive resolved seconds_needed; invalid "
                "MM:SS must be repaired at Selection, not changed into 4s"
            )
        item = material_by_source.get(str(shot["source_id"]))
        card = load_card(cards[shot["source_id"]]) if shot["source_id"] in cards else None
        action_contract = None
        selected_action = str(shot.get("action_id") or "none")
        # Legacy direct callers omitted the treatment.  Current paid and
        # cached Selection answers are schema/audit gated and must carry it.
        action_treatment = str(
            shot.get("action_treatment")
            or ("complete_here" if selected_action != "none" else "none")
        )
        window = _usable_window(shot)
        if item is not None:
            from montagewright.planner import resolve_named_span

            resolved_span = resolve_named_span(shot, tuple(material or ()))
            if resolved_span is not None:
                window = (
                    float(resolved_span.starts_seconds),
                    float(resolved_span.ends_seconds),
                )
        resolved_action = None
        if selected_action != "none" and item is not None:
            from montagewright.planner import resolve_action_boundary

            resolved_action = resolve_action_boundary(
                shot, tuple(material or ())
            )
            if resolved_action is not None:
                _, _, usable_from, usable_to = resolved_action
                window = (usable_from, usable_to)
        if window is not None:
            # The card said where this take is worth cutting into and until
            # now that answer only ever reached a line of prompt text. A
            # planner reading "可用區間 4.2–8.5s" and choosing 2.0 was not
            # contradicted by anything, so the shot began on the camera still
            # swinging -- which is the case the field was added for.
            first, last = window
            room = max(0.0, last - first)
            wanted = min(wanted, room) if room > 0 else wanted
            start = min(max(start, first), max(first, last - wanted))
        action_card = None
        if selected_action != "none" and resolved_action is not None:
            local_action = selected_action.rsplit(":", 1)[-1]
            action_start, action_end, _, _ = resolved_action
            # Selection and EDL must execute the same immutable local
            # contract.  Reopening and reparsing the card here created a
            # second authority that could disagree after Selection had
            # already passed.
            action_card = {"action": [{
                "id": local_action,
                "what": "selected source action",
                "from": action_start,
                "to": action_end,
            }]}
        elif selected_action == "none":
            action_card = card
        elif item is None:
            # Compatibility for direct/legacy EDL callers that predate the
            # material catalogue. Production render paths always provide it.
            action_card = card
        if action_card is not None:
            note = None
            if action_treatment == "complete_here":
                start, action_contract, note = snap_to_action_contract(
                    action_card, start, wanted, action_id=selected_action, within=window,
                    focus=focus_of(str(shot["source_id"])),
                )
                if selected_action == "none" or action_contract is None:
                    raise ValueError(
                        f"{clip_id} selected action {selected_action!r}, but it "
                        "cannot be resolved and completed inside this source window"
                    )
                minimum = action_contract.minimum_duration_from(start)
                if wanted + 1e-3 < minimum:
                    raise ValueError(
                        f"{clip_id} gives {wanted:.2f}s to complete action "
                        f"{selected_action!r}, which needs {minimum:.2f}s; "
                        "Selection must repair this before Rhythm"
                    )
            elif action_treatment == "after_completion":
                local_action = selected_action.rsplit(":", 1)[-1]
                beat = next(
                    (
                        one for one in action_beats(action_card)
                        if one.beat_id == local_action
                    ),
                    None,
                )
                if beat is None or selected_action == "none":
                    raise ValueError(
                        f"{clip_id} cannot resolve after_completion for action "
                        f"{selected_action!r}"
                    )
                start = beat.ends_seconds
                if window is not None:
                    first, last = window
                    if start < first - 1e-3 or start + wanted > last + 1e-3:
                        raise ValueError(
                            f"{clip_id} cannot fit {wanted:.2f}s after action "
                            f"{selected_action!r} inside this source window"
                        )
                note = (
                    f"entered after {selected_action} completed at "
                    f"{beat.ends_seconds:.2f}s"
                )
            elif action_treatment == "intentional_cut":
                local_action = selected_action.rsplit(":", 1)[-1]
                beat = next(
                    (
                        one for one in action_beats(action_card)
                        if one.beat_id == local_action
                    ),
                    None,
                )
                if beat is None or selected_action == "none":
                    raise ValueError(
                        f"{clip_id} cannot resolve intentional_cut for action "
                        f"{selected_action!r}"
                    )
                if start + wanted <= beat.starts_seconds + 1e-3:
                    raise ValueError(
                        f"{clip_id} marks {selected_action!r} intentional_cut, "
                        "but its source window ends before that action begins"
                    )
                if start >= beat.ends_seconds - 1e-3 or start + wanted >= beat.ends_seconds - 1e-3:
                    raise ValueError(
                        f"{clip_id} marks {selected_action!r} intentional_cut, "
                        "but its source window does not actually cut before "
                        "the action completes"
                    )
                note = (
                    f"intentionally cuts {selected_action} before its "
                    f"{beat.ends_seconds:.2f}s completion"
                )
            elif action_treatment != "none" or selected_action != "none":
                raise ValueError(
                    f"{clip_id} has inconsistent action_id/action_treatment"
                )
            if note:
                snaps[clip_id] = note
        elif action_treatment != "none" or selected_action != "none":
            raise ValueError(
                f"{clip_id} cannot resolve selected action {selected_action!r} "
                "from either its card or local material action windows"
            )
        if item is not None:
            from montagewright.planner import material_look_boxes

            reframe = reframe.model_copy(
                update={"look_boxes": material_look_boxes(item, reframe)}
            )
        elif card is not None:
            # Where the card already measured each look. This is what makes
            # "how long does this shot need" a fact about this shot rather
            # than a constant per move: the distance between two watches on
            # one table is not the distance between two models on a stage,
            # and neither is 2.5 seconds.
            reframe = reframe.model_copy(
                update={"look_boxes": _look_boxes(card, reframe)}
            )
        coverage_claim = shot.get("coverage_claim_seconds")
        source_motion_contract = None
        if item is not None:
            from montagewright.coverage import (
                source_motion_contract_for, visual_supported_max,
            )

            coverage_claim = visual_supported_max(
                item,
                role=str(shot.get("picture_role") or "primary_action"),
                source_start=start,
                available_seconds=wanted,
                motion_role=(
                    str(shot.get("source_motion_role") or "")
                    if delivered_camera_intent_of(shot) == "use_source_motion"
                    else ""
                ),
                presentation_intent=next((
                    str(look.get("presentation_intent") or "")
                    for look in shot.get("looks") or []
                    if look.get("presentation_intent")
                ), ""),
                target_id=next((
                    str(look.get("entity_id") or "none")
                    for look in shot.get("looks") or []
                    if str(look.get("entity_id") or "none") != "none"
                ), "none"),
            )
            if delivered_camera_intent_of(shot) == "use_source_motion":
                source_motion_contract = source_motion_contract_for(
                    item,
                    source_start=start,
                    source_end=start + wanted,
                    motion_role=str(shot.get("source_motion_role") or ""),
                )
        content_contract = None
        content_policy = str(shot.get("content_policy") or "")
        content_minimum = float(shot.get("content_min_seconds") or 0.0)
        commitment_id = str(shot.get("commitment_id") or "")
        content_purpose = str(shot.get("content_purpose") or "")
        if (
            content_policy
            and content_minimum > 0.0
            and commitment_id
            and content_purpose
        ):
            content_contract = ContentContract(
                commitment_id=commitment_id,
                purpose=content_purpose,
                policy=content_policy,
                minimum_seconds=content_minimum,
            )
        clips.append(
            Clip(
                clip_id=clip_id,
                source_id=shot["source_id"],
                approx_in_seconds=start,
                approx_out_seconds=start + wanted,
                in_looks_like=subject_of(shot),
                energy_intent=shot.get("energy", "medium"),
                audio_role=shot.get("audio_role", "auto"),
                audio_completion=shot.get("audio_completion", "none"),
                picture_role=shot.get("picture_role", "primary_action"),
                coverage_claim_seconds=coverage_claim,
                reframe=reframe,
                # Carried on the clip so the layers after this one can see
                # it. Rhythm stretches shots to land on beats and the
                # executor asks only whether the time exists in the file;
                # neither could tell a second of usable take from a second
                # of somebody resetting a prop.
                # Every moment this take offers, by name, so the rhythm pass
                # can point at one and grounding can put it on a beat.
                moments={
                    one.beat_id: one.starts_seconds
                    for one in action_beats(action_card or {})
                    if action_treatment == "complete_here"
                    and str(shot.get("action_id") or "none") != "none"
                    and one.beat_id
                    == str(shot.get("action_id")).rsplit(":", 1)[-1]
                },
                action_contracts=(
                    [action_contract] if action_contract is not None else []
                ),
                content_contracts=(
                    [content_contract] if content_contract is not None else []
                ),
                source_motion_contracts=(
                    [source_motion_contract]
                    if source_motion_contract is not None else []
                ),
                usable_from_seconds=(window[0] if window else 0.0),
                usable_to_seconds=(window[1] if window else 0.0),
            )
        )
    from montagewright.schema import AudioClip

    available_audio = _audio_spans(transcripts or {})
    audio_clips = []
    speaker_sync: dict[int, tuple[float, str]] = {}
    for assignment in selection.get("audio_assignments") or []:
        span_id = str(assignment.get("audio_span_id") or "")
        span = available_audio.get(span_id)
        shot_index = int(assignment.get("starts_at_shot_index", -1))
        if span is None or not 0 <= shot_index < len(clips):
            raise ValueError(f"cannot resolve audio assignment {span_id!r}")
        picture = clips[shot_index]
        if picture.picture_role == "speaker":
            if picture.source_id != str(span["source_id"]):
                raise ValueError(
                    f"speaker picture {picture.clip_id} uses {picture.source_id} "
                    f"but its narrative audio uses {span['source_id']}"
                )
            offset = float(assignment.get("offset_seconds") or 0.0)
            source_in = float(span["in_seconds"]) - offset
            if source_in < 0:
                raise ValueError(
                    f"speaker picture {picture.clip_id} cannot begin "
                    f"{abs(source_in):.3f}s before its source"
                )
            speaker_sync[shot_index] = (source_in, span_id)
        audio_clips.append(AudioClip(
            audio_id=str(assignment.get("audio_id") or f"a{len(audio_clips):02d}"),
            source_id=str(span["source_id"]),
            in_seconds=float(span["in_seconds"]),
            out_seconds=float(span["out_seconds"]),
            starts_at_clip_id=clips[shot_index].clip_id,
            offset_seconds=float(assignment.get("offset_seconds") or 0.0),
            role="narrative",
            completion=str(assignment.get("completion") or "complete_thought"),
            gain_db=float(assignment.get("gain_db") or 0.0),
            why=str(assignment.get("why") or ""),
        ))

    for index, (source_in, span_id) in speaker_sync.items():
        clip = clips[index]
        duration = clip.approx_out_seconds - clip.approx_in_seconds
        clips[index] = clip.model_copy(update={
            "approx_in_seconds": source_in,
            "approx_out_seconds": source_in + duration,
        })
        snaps[clip.clip_id] = (
            f"speaker picture source clock aligned to {span_id} at "
            f"{source_in:.3f}s"
        )
    return EDL(
        project_id=rushes.name, clips=clips, audio_clips=audio_clips
    ), snaps


def _usable_window(shot: dict) -> tuple[float, float] | None:
    """The span this shot was cut from, as bounds.

    Read off the shot rather than off the card, because by the time this runs
    the planner has already named a span and `expand_spans` has written its
    edges here. Going back to the card would mean picking one of several
    stretches again, on less information than the plan had.
    """

    first = float(shot.get("usable_from_seconds", 0.0) or 0.0)
    last = float(shot.get("usable_to_seconds", 0.0) or 0.0)
    return (first, last) if last > first else None


def _delivery_selection(
    selection: dict[str, Any], reference_grounding: dict[str, dict],
    degradations: Sequence[Any] = (),
) -> tuple[dict[str, Any], str]:
    """Project immutable planning evidence into honest delivery status."""

    delivery_selection = copy.deepcopy(selection)
    unresolved_by_clip: dict[str, list[str]] = {}
    for step in degradations:
        if getattr(step, "severity", "advisory") != "blocking_shot":
            continue
        adjudication = getattr(step, "adjudication", "unadjudicated")
        if adjudication == "accept":
            continue
        clip_id = str(getattr(step, "clip_id", "") or "")
        if not clip_id:
            continue
        reason = str(
            getattr(step, "adjudication_reason", "")
            or getattr(step, "trigger", "camera treatment was not delivered")
        )
        if adjudication == "unadjudicated":
            reason = "尚未逐顆驗收：" + reason
        unresolved_by_clip.setdefault(clip_id, []).append(reason)
    for index, shot in enumerate(delivery_selection.get("shots") or []):
        if (
            reference_grounding.get(f"k{index:02d}", {}).get("status")
            == "sam_geometry_validated"
        ):
            shot["identity_status"] = "track_validated"
            shot["identity_issue"] = "; ".join(
                str(one) for one in shot.get("identity_advisories") or []
                if str(one)
            )
        clip_id = f"k{index:02d}"
        if clip_id in unresolved_by_clip:
            shot["delivery_status"] = "needs_review"
            shot["delivery_issue"] = "; ".join(unresolved_by_clip[clip_id])
    delivery_status = (
        "needs_review"
        if unresolved_by_clip or any(
                shot.get("identity_status") in {"needs_review", "unverified"}
                for shot in delivery_selection.get("shots") or []
            )
        else "ready"
    )
    return delivery_selection, delivery_status


def _write_report(output: Path, **parts) -> None:
    """The account of what was decided and what it cost."""

    report = parts["report"]
    plan = parts.get("plan")
    delivery_selection, delivery_status = _delivery_selection(
        parts["selection"], report.reference_grounding, report.degradations
    )
    payload = {
        "direction": parts["direction"],
        "selection": delivery_selection,
        "delivery_status": delivery_status,
        # Which part of the track was used. Decided by the rhythm pass and
        # otherwise invisible: the bed sounding right or wrong is the most
        # audible thing in a cut and nothing recorded where it came from.
        "edl": {
            "music_from_seconds": getattr(plan, "music_from_seconds", 0.0),
            "music_spans": list(getattr(plan, "music_spans", []) or []),
        },
        # Against how many asked for a beat, not how many cuts there are. A
        # speech cut where every length is content-led was reading as 0/13,
        # which is what missing every beat would look like.
        "cuts_on_music": (
            f"{report.aligned_cuts}/"
            f"{sum(1 for e in report.rhythm_decisions.values() if e.get('cut_on_beat'))}"
            if report.rhythm_decisions
            else f"{report.aligned_cuts}/{report.total_cuts}"
        ),
        "shots_following": report.following_shots,
        "shots_held": report.static_shots,
        "source_motion": report.source_motion,
        "source_motion_details": report.source_motion_details,
        "digital_motion": report.digital_motion,
        "motion": {
            clip_id: {
                "source": report.source_motion.get(clip_id, "locked"),
                "digital": report.digital_motion.get(clip_id, "hold"),
                "composite": (
                    "stacked"
                    if report.source_motion.get(clip_id, "locked") != "locked"
                    and report.digital_motion.get(clip_id, "hold") != "hold"
                    else "source_only"
                    if report.source_motion.get(clip_id, "locked") != "locked"
                    else "digital_only"
                    if report.digital_motion.get(clip_id, "hold") != "hold"
                    else "still"
                ),
                "camera_intent": (
                    parts["selection"]["shots"][int(clip_id[1:])].get(
                        "camera_intent", "hold"
                    )
                    if clip_id.startswith("k")
                    and clip_id[1:].isdigit()
                    and int(clip_id[1:]) < len(parts["selection"]["shots"])
                    else "hold"
                ),
                "requested_camera_intent": (
                    parts["selection"]["shots"][int(clip_id[1:])].get(
                        "camera_intent", "hold"
                    )
                    if clip_id.startswith("k")
                    and clip_id[1:].isdigit()
                    and int(clip_id[1:]) < len(parts["selection"]["shots"])
                    else "hold"
                ),
                "delivered_camera_intent": (
                    delivered_camera_intent_of(
                        parts["selection"]["shots"][int(clip_id[1:])]
                    )
                    if clip_id.startswith("k")
                    and clip_id[1:].isdigit()
                    and int(clip_id[1:]) < len(parts["selection"]["shots"])
                    else "hold"
                ),
            }
            for clip_id in sorted(
                set(report.source_motion) | set(report.digital_motion)
            )
        },
        "upscales": {k: round(v, 3) for k, v in report.upscales.items()},
        "subject_notes": report.subject_notes,
        "subject_tracks": report.subject_tracks,
        "reference_grounding": report.reference_grounding,
        "plan_disagreements": report.plan_disagreements,
        "degradations": [
            {
                "clip_id": step.clip_id,
                "ladder": step.ladder_other or step.ladder,
                "trigger": step.trigger,
                "measured": step.measured,
                "adjudication": step.adjudication,
                "severity": step.severity,
                "attempt_id": step.attempt_id,
                # Who settled it and on what grounds. Without this the report
                # says "replan" and nothing about why, which is the same
                # position the reviewer was in before they could see the shot.
                "adjudication_reason": step.adjudication_reason,
            }
            for step in report.degradations
        ],
        "tokens": {
            "input": report.input_tokens
            + sum(u.input_tokens for u in parts["usages"]),
            "output": report.output_tokens
            + sum(u.output_tokens + u.thought_tokens for u in parts["usages"]),
        },
        "duration_seconds": round(parts["result"].duration_seconds, 3),
        "target_seconds": report.target_seconds,
        "duration_shortfall_seconds": report.duration_shortfall,
        "coverage_seconds": report.coverage_seconds,
        "unsupported_seconds": report.unsupported_seconds,
        "coverage_details": report.coverage_details,
        "moves_too_short": report.moves_too_short,
        "spend": report.spend(),
        "spend_all_attempts": (
            report.ledger.cumulative_summary()
            if report.ledger is not None else report.spend()
        ),
        "cut_on_action": parts.get("snaps", {}),
        "set_aside": parts.get("set_aside", {}),
        # Everything that was on the table. Without it there is no way to
        # say which clips were simply passed over, which is how most of a
        # folder does not reach the film.
        "material_ids": parts.get("material_ids", []),
        "rhythm": report.rhythm_decisions,
        # Per shot, against the plan that asked for it. The whole-cut verdict
        # below answers a different question and has never once caught a
        # composition fault, because at thirty seconds the next shot arrives
        # before the fault registers.
        "shots": parts.get("shot_verdicts", {}),
        "review": {
            "stopped_because": parts.get("stopped_because"),
            "rounds": [
                {
                    "verdict": entry.verdict.verdict,
                    "overall": entry.verdict.overall,
                    "issues": [
                        {
                            "clip_id": issue.clip_id,
                            "at_seconds": issue.at_seconds,
                            "type": issue.issue_type,
                            "severity": issue.severity,
                            "description": issue.description,
                            "fix": issue.fix,
                        }
                        for issue in entry.verdict.issues
                    ],
                }
                for entry in parts.get("rounds", [])
            ],
        },
    }
    (output / "report.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8"
    )


def command_transcribe(args: argparse.Namespace) -> int:
    """Subtitle a video without touching the edit.

    This is the transcript half on its own, because subtitling something
    already finished is a real job and has nothing to do with cutting. The
    same card feeds the editorial passes when there is an edit.
    """

    from montagewright.transcript import describe, lines_of, load, save, to_srt
    from montagewright.uploads import content_hash

    client = _client()
    cache = UploadCache.load(args.upload_cache or default_cache_path())
    ledger = Ledger(cap_usd=args.budget, model_id=MODEL_ID)

    sources = (
        sorted(p for p in args.source.iterdir() if p.suffix in VIDEO_SUFFIXES)
        if args.source.is_dir()
        else [args.source]
    )
    if not sources:
        raise SystemExit(f"no video files in {args.source}")

    output = (args.output or args.source.parent).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)

    library = (args.library or default_library()).expanduser()
    work = output / "work"

    for source in sources:
        # The same proxy a render would make, keyed the same way. Both
        # commands transcribe the same material and this one kept its answer
        # beside its own output under the file's name, so rendering the same
        # folder afterwards found nothing and paid for all of it again -- at
        # twenty cents a clip, on the most expensive call in the tool.
        proxy = _make_proxy(
            source, work / "proxies" / f"{source.stem}.mp4", library=library
        )
        destination = (
            library / "transcripts" / f"{content_hash(proxy)[:20]}.json"
        )
        card = load(destination)
        if card is None:
            card, usage = describe(
                proxy, client=client, locale=args.locale, cache=cache,
                # Match the render path: Gemini watches the reusable proxy,
                # while local Apple Speech listens to the source master.
                # A standalone subtitle command must not silently measure a
                # second, lossy audio encode for the same file.
                audio=source,
                ledger=ledger,
            )
            save(card, destination)
        lines = lines_of(card)
        (output / f"{source.stem}.srt").write_text(
            to_srt(lines), encoding="utf-8"
        )
        changed = sum(1 for line in lines if line.corrected)
        print(
            f"{source.name}: {len(lines)} lines, {changed} corrected, "
            f"{card.get('language')} — {output / (source.stem + '.srt')}",
            flush=True,
        )
        for note in card.get("uncertain", [])[:3]:
            print(f"  unsure — {str(note)[:120]}", flush=True)

    print(f"transcript spend ${ledger.spent_usd:.4f}", flush=True)
    return 0


def command_timeline(args: argparse.Namespace) -> int:
    """Write a timeline for a cut that already exists.

    Choosing the flavour before the run and finding out afterwards that you
    wanted one meant running the whole thing again. Everything a timeline
    needs is in the report and the material, and rebuilding the plan costs
    nothing -- the subject positions come out of the cards.
    """

    from montagewright.clipcard import card_map
    from montagewright.executor import plan_render
    from montagewright.pipeline import Report, follow_subjects, probe, read_crops
    from montagewright.timeline import to_fcpxml, to_xmeml

    output = args.output.expanduser().resolve()
    report = json.loads((output / "report.json").read_text(encoding="utf-8"))
    original_shots = report.get("selection", {}).get("shots", [])
    shots = original_shots
    rhythm = report.get("rhythm", {})
    current = {}
    current_path = output / "work" / "current-timeline.json"
    if current_path.exists():
        current = json.loads(current_path.read_text(encoding="utf-8"))
        if current.get("version") not in {
            "montagewright-current-timeline-v1",
            "montagewright-current-timeline-v2",
        }:
            raise SystemExit("current-timeline.json has an unknown version")
        blocks = current.get("shots") or []
        shots = [
            {
                **original_shots[int(one["selection_index"])],
                "start_seconds": float(one["in_seconds"]),
            }
            for one in blocks
        ]
        rhythm = {
            f"k{index:02d}": {"seconds": float(one["seconds"])}
            for index, one in enumerate(blocks)
        }
        projected = dict(report)
        projected_selection = dict(report.get("selection") or {})
        projected_selection["shots"] = shots
        projected["selection"] = projected_selection
        projected["rhythm"] = rhythm
        report = projected
    aspect = ASPECTS.get(report.get("direction", {}).get("aspect", "9:16"), 9 / 16)
    library = (args.library or default_library()).expanduser()
    cards = card_map(output / "work" / "proxies", library / "cards")

    hunting = [output / "work" / "shots"] + (
        [args.rushes.expanduser()] if args.rushes else []
    )
    clips, sources = [], {}
    for index, shot in enumerate(shots):
        source_id = shot["source_id"]
        if source_id not in sources:
            match = next(
                (
                    path
                    for folder in hunting if folder.exists()
                    for path in folder.iterdir()
                    if path.stem == source_id
                ),
                None,
            )
            if match is None:
                raise SystemExit(
                    f"cannot find {source_id}; pass --rushes with its folder"
                )
            sources[source_id] = probe(source_id, match)
        start = float(shot.get("start_seconds", 0.0))
        clips.append(
            Clip(
                clip_id=f"k{index:02d}", source_id=source_id,
                approx_in_seconds=start,
                approx_out_seconds=start
                + float(rhythm.get(f"k{index:02d}", {}).get("seconds", 0.0)),
                in_looks_like=subject_of(shot),
                energy_intent=shot.get("energy", "medium"),
                reframe=reframe_of(shot),
            )
        )
    edl = EDL(project_id=output.name, clips=clips)
    # What the render actually did, when the run left a record of it. This
    # used to rebuild the paths from the cards with no client and no
    # checkpoint, which for a held frame is the same arithmetic and for
    # anything that followed a subject is not: that path came out of a mask
    # propagation nothing here can repeat. The timeline is the one output
    # where being approximately right is worst -- a wrong number in a report
    # is a wrong number, an edit list that disagrees with the film opens as a
    # different cut and the person opening it cannot tell.
    paths = read_crops(output / "work" / "crops.json")
    if paths:
        print(f"crops       {len(paths)} read from the render", flush=True)
    else:
        print(
            "crops       no record kept by this run; rebuilding from the "
            "cards, so followed shots will differ from the film",
            flush=True,
        )
        paths = follow_subjects(
            edl, sources, target_aspect=aspect, report=Report(),
            cards=cards, checkpoint=None, client=None,
        )
    plan = plan_render(
        edl, sources, target_aspect=aspect, crop_paths=paths,
        output_size=(
            tuple(current["output_size"])
            if current.get("output_size") else None
        ),
        output_fps=int(current.get("output_fps") or 30),
    )
    width, height = plan.output_size
    for flavour, suffix, build in (
        ("premiere", "xml", to_xmeml), ("finalcut", "fcpxml", to_fcpxml)
    ):
        if args.flavour in {flavour, "both"}:
            path = output / f"timeline.{suffix}"
            path.write_text(
                build(plan, report, name=output.name,
                      width=width, height=height,
                      music=(
                          output / "bed-as-laid.m4a"
                          if (output / "bed-as-laid.m4a").exists() else None
                      ),
                      voice=(
                          output / "voice-as-laid.m4a"
                          if (output / "voice-as-laid.m4a").exists() else None
                      ),
                      graphics=(
                          output / "graphics-overlay.mov"
                          if (output / "graphics-overlay.mov").exists()
                          else None
                      )),
                encoding="utf-8",
            )
            print(f"timeline    {path}", flush=True)
    return 0


def command_graphics(args: argparse.Namespace) -> int:
    """Inspect and finish the same graphics track the Web editor uses."""

    from montagewright.graphics import (
        CopyFact,
        GraphicsPlan,
        _layout_frames,
        burn_graphics,
        render_graphics_overlay,
        compile_graphic,
        validate_brief_authority,
        validate_for_render,
    )
    from montagewright.measure.media import probe_video
    from montagewright.measure.storage import write_json
    from montagewright.renderer import probe_duration

    output = args.output.expanduser().resolve()
    plan_path = output / "work" / "graphics.json"
    if not plan_path.exists():
        raise SystemExit(f"no graphics track at {plan_path}")
    plan = GraphicsPlan.model_validate_json(plan_path.read_text(encoding="utf-8"))
    clean = next(
        (output / name for name in ("deliverable.mp4", "picture.mp4")
         if (output / name).exists()),
        None,
    )

    if args.action == "inspect":
        for cue in plan.cues:
            primary = plan.fact(cue.primary_fact_id)
            print(
                f"{cue.graphic_id:18s} {cue.status:8s} "
                f"{cue.at_seconds:7.2f}-{cue.at_seconds + cue.duration_seconds:7.2f} "
                f"{cue.kind:14s} {cue.style.preset:20s} {primary.exact_text}",
                flush=True,
            )
        return 0

    if args.action == "approve":
        if not args.graphic_id:
            raise SystemExit("graphics approve requires --graphic-id")
        try:
            cue = next(one for one in plan.cues if one.graphic_id == args.graphic_id)
        except StopIteration:
            raise SystemExit(f"unknown graphic id {args.graphic_id}") from None
        facts = list(plan.facts)
        updates: dict[str, str] = {}
        for which, fact_id in (
            ("primary", cue.primary_fact_id),
            ("secondary", cue.secondary_fact_id),
        ):
            if not fact_id:
                continue
            source = plan.fact(fact_id)
            approved_id = f"user.{cue.graphic_id}.{which}"
            approved = CopyFact(
                fact_id=approved_id,
                exact_text=source.exact_text,
                source_kind="user",
                source_reference=f"cli-review:{source.fact_id}",
                source_sha256=source.source_sha256,
                text_sha256=hashlib.sha256(
                    source.exact_text.encode("utf-8")
                ).hexdigest(),
                allowed_kinds=[cue.kind],
                confidence=source.confidence,
                approved=True,
                approved_by="human_review",
            )
            facts = [one for one in facts if one.fact_id != approved_id]
            facts.append(approved)
            updates[f"{which}_fact_id"] = approved_id
        approved_cue = cue.model_copy(update={**updates, "status": "approved"})
        plan = GraphicsPlan(
            version=plan.version,
            revision=plan.revision + 1,
            brand=plan.brand,
            facts=facts,
            cues=[approved_cue if one.graphic_id == cue.graphic_id else one
                  for one in plan.cues],
        )
        write_json(plan_path, plan)
        print(f"approved    {cue.graphic_id}", flush=True)
        return 0

    if clean is None:
        raise SystemExit("this output has no finished clean picture")
    duration = probe_duration(clean)
    problems = validate_for_render(plan, duration_seconds=duration)
    approved_manifest = output / "work" / "approved-copy.json"
    authority = []
    if approved_manifest.exists():
        authority = [
            CopyFact.model_validate(one)
            for one in json.loads(
                approved_manifest.read_text(encoding="utf-8")
            ).get("facts", [])
        ]
    validate_brief_authority(plan, authority)
    if problems:
        for problem in problems:
            print(f"invalid     {problem}", flush=True)
        return 2
    if args.action == "validate":
        print(
            f"valid       {sum(one.status == 'approved' for one in plan.cues)} "
            "approved graphics",
            flush=True,
        )
        return 0

    beat_grid = read_runtime_beat_grid(output / "work" / "graphics-beat-grid.json")
    if args.action == "preview":
        if not args.graphic_id:
            raise SystemExit("graphics preview requires --graphic-id")
        try:
            cue = next(one for one in plan.cues if one.graphic_id == args.graphic_id)
        except StopIteration:
            raise SystemExit(f"unknown graphic id {args.graphic_id}") from None
        shape = probe_video(clean).video
        width, height = int(shape.display_width), int(shape.display_height)
        rate = shape.average_frame_rate or shape.real_frame_rate or 30
        from montagewright.graphics import resolve_graphic_window
        start, _ = resolve_graphic_window(
            cue, beat_grid=beat_grid, output_fps=rate,
            timeline_duration=duration,
        )
        preview_dir = output / "work" / "graphics-cli-preview"
        frames = _layout_frames(
            clean, cue.model_copy(update={"at_seconds": start}),
            cache_dir=preview_dir / "frames",
        )
        made, _ = compile_graphic(
            cue, plan, width=width, height=height,
            into=preview_dir / f"{cue.graphic_id}.png", frames=frames,
            beat_grid=beat_grid,
            output_fps=rate,
            timeline_duration=duration,
        )
        print(f"preview     {made.path}", flush=True)
        return 0

    destination = output / "deliverable-graphics.mp4"
    made = burn_graphics(
        clean, plan, destination,
        work=output / "work" / "graphics-render",
        beat_grid=beat_grid,
    )
    overlay = render_graphics_overlay(
        clean, plan, output / "graphics-overlay.mov",
        work=output / "work" / "graphics-render",
        beat_grid=beat_grid,
    )
    for stale_timeline in (output / "timeline.xml", output / "timeline.fcpxml"):
        stale_timeline.unlink(missing_ok=True)
    print(f"graphics    {made}", flush=True)
    print(f"overlay     {overlay}", flush=True)
    return 0


def _write_run_state(path: Path, state: str) -> None:
    """Say who is doing this and whether they are still here.

    The pid is the whole point: "running" written by a process that has
    since died is exactly the stale claim the interface was already
    guarding against by ignoring the field.
    """

    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps({
                "state": state, "pid": os.getpid(), "at": time.time(),
            }),
            encoding="utf-8",
        )
    except OSError:
        # Not being watchable is not a reason to refuse to cut.
        pass


def main(argv: list[str] | None = None) -> int:
    from montagewright.environment import load_project_env

    load_project_env()
    parser = argparse.ArgumentParser(prog="montagewright")
    sub = parser.add_subparsers(dest="command", required=True)

    render = sub.add_parser("render", help="Cut a folder of rushes into a film")
    render.add_argument("rushes", type=Path)
    render.add_argument("--brief", type=Path)
    render.add_argument(
        "--grounding-spec",
        type=Path,
        help="validated reference-identity JSON. A canonical, self-contained "
             "copy is kept at work/grounding-spec.json for this run.",
    )
    render.add_argument(
        "--grounding-target-id", default="target.primary",
        help="stable ID for the simple reference-image mode",
    )
    render.add_argument(
        "--grounding-target-description", default="",
        help="what exact identity the supplied reference images represent",
    )
    render.add_argument(
        "--grounding-reference", type=Path, action="append", default=[],
        help="positive reference image; repeat for more views",
    )
    render.add_argument(
        "--grounding-negative", type=Path, action="append", default=[],
        help="image of a lookalike that must never be substituted; repeat "
             "for more. The spec has carried these since it was written and "
             "no entry point offered them, so telling the difference between "
             "two similar things rested entirely on prose.",
    )
    render.add_argument(
        "--grounding-identity-cue", action="append", default=[],
        help="stable visible identity cue; repeat as needed",
    )
    render.add_argument(
        "--grounding-exclusion", action="append", default=[],
        help="lookalike or depiction that must not be substituted",
    )
    render.add_argument("--music", type=Path)
    render.add_argument("--music-map", type=Path)
    render.add_argument("--aspect", choices=sorted(ASPECTS), default="9:16")
    render.add_argument(
        "--seconds", type=float, default=0.0,
        help="how long the finished cut should be. Without it the direction "
             "pass picks a length, which is the right default when nobody "
             "has a slot to fill and the wrong one when they do -- writing "
             "'make it 15 seconds' in the brief is a request, not a number.",
    )
    render.add_argument(
        "--duration-mode", choices=("exact", "preferred"), default="exact",
        help="whether --seconds is a hard delivery specification or a preferred "
             "maximum that may resolve shorter when verified content is insufficient",
    )
    render.add_argument(
        "--sample", type=int, default=0, metavar="N",
        help="cut from N of the clips instead of all of them. For trying a "
             "change without paying to describe a whole shoot: seventy-four "
             "cards is about a dollar and twelve is fifteen cents. The same "
             "N always picks the same clips, spread across the folder rather "
             "than taken off the front, so the cards stay cached between "
             "runs and the sample is not all one setup.",
    )
    render.add_argument(
        "--sam-checkpoint", type=Path,
        help="SAM 2.1 checkpoint. By default Montagewright discovers "
             "artifacts/models/sam2.1_hiera_tiny.pt and uses it.",
    )
    render.add_argument(
        "--no-sam-tracking", action="store_true",
        help="Explicitly disable SAM and use sparse Gemini positions.",
    )
    render.add_argument(
        "--budget",
        type=float,
        default=5.0,
        help="Total spend ceiling in USD. Reaching it stops the run with the "
        "best cut so far rather than making the next step cheaper.",
    )
    render.add_argument(
        "--upload-cache",
        type=Path,
        help="Where uploaded-media URIs are remembered. Defaults to a shared "
        "location, because the key is the file's content and a per-run store "
        "re-uploads material that has not changed.",
    )
    render.add_argument(
        "--review",
        action="store_true",
        help="Watch the finished cut and report what it would change.",
    )
    render.add_argument("--output", type=Path, required=True)
    render.add_argument(
        "--timeline", choices=["none", "premiere", "finalcut", "both"],
        default="none",
        help="also write an editable timeline (off unless asked for)",
    )
    render.add_argument(
        "--subtitles", choices=["none", "sidecar", "burn"], default="sidecar",
        help=(
            "sidecar writes an SRT beside the cut; burn also puts the words "
            "on the picture, inside the safe area for the delivery aspect"
        ),
    )
    render.add_argument(
        "--subtitle-look", choices=["plain", "speakers", "spoken", "plate"],
        default="plain",
        help="how burned subtitles look: plain, a colour per speaker, "
             "filling as it is said, or on a plate",
    )
    render.add_argument(
        "--subtitle-font", type=Path,
        help="a font file to set the subtitles in; the system is asked if "
             "this is not given",
    )
    render.add_argument(
        "--speech", choices=["auto", "never"], default="auto",
        help="transcribe clips whose card calls the speech content",
    )
    render.add_argument("--locale", default="zh-TW")
    render.add_argument(
        "--library", type=Path,
        help="where cards and transcripts live; shared across runs",
    )
    render.set_defaults(handler=command_render)

    speak = sub.add_parser(
        "transcribe", help="Subtitle a video, or a folder of them"
    )
    speak.add_argument("source", type=Path)
    speak.add_argument("--locale", default="zh-TW")
    speak.add_argument("--output", type=Path)
    speak.add_argument("--budget", type=float, default=5.0)
    speak.add_argument("--upload-cache", type=Path)
    speak.add_argument(
        "--library", type=Path,
        help="where cards and transcripts live; shared across runs",
    )
    speak.set_defaults(handler=command_transcribe)

    lay = sub.add_parser(
        "timeline", help="Write a timeline for a cut that already exists"
    )
    lay.add_argument("output", type=Path)
    lay.add_argument(
        "--flavour", choices=["premiere", "finalcut", "both"], default="both"
    )
    lay.add_argument(
        "--rushes", type=Path,
        help="where the sources are, if they are not under the output",
    )
    lay.add_argument("--library", type=Path)
    lay.set_defaults(handler=command_timeline)

    graphics = sub.add_parser(
        "graphics", help="Inspect, approve, validate, preview or render graphics"
    )
    graphics.add_argument("output", type=Path)
    graphics.add_argument(
        "action", choices=["inspect", "approve", "validate", "preview", "render"]
    )
    graphics.add_argument("--graphic-id")
    graphics.set_defaults(handler=command_graphics)

    effective_argv = list(argv) if argv is not None else sys.argv[1:]
    args = parser.parse_args(effective_argv)
    args._argv = effective_argv
    # A run started here is invisible to the interface: it reads a folder,
    # and a folder with no report in it is a run that died with the last
    # server. So a cut that was busy cutting showed as "interrupted", under
    # a workspace that never refreshed, for as long as it took to finish.
    # Leaving a pid behind is enough for the reader to tell the two apart.
    where = getattr(args, "output", None)
    if where is None:
        return args.handler(args)
    state = Path(where).expanduser() / "run-state.json"
    _write_run_state(state, "running")

    def _stopped(signum: int, frame: object) -> None:
        # The stop button in the interface reaches a terminal run through this.
        _write_run_state(state, "stopped")
        raise KeyboardInterrupt

    try:
        signal.signal(signal.SIGTERM, _stopped)
    except (ValueError, OSError):
        pass  # not the main thread; the pid check still tells the truth
    try:
        outcome = args.handler(args)
    except KeyboardInterrupt:
        _write_run_state(state, "stopped")
        raise
    except BaseException:
        _write_run_state(state, "failed")
        raise
    _write_run_state(state, "done" if not outcome else "failed")
    return outcome


if __name__ == "__main__":
    sys.exit(main())
