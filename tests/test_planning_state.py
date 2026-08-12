from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
from pydantic import ValidationError

import montagewright.planning_state as planning_state
from montagewright.planning_state import (
    MaterialSpanRecord,
    PlanningState,
    PlanningStateConflictError,
    PlanningStateDelta,
    PlanningStateError,
    PlanningStateLineageError,
    apply_planning_delta,
    canonical_json,
    canonical_sha256,
    write_planning_revision,
)


MATERIAL_DIGEST = hashlib.sha256(b"immutable-material-catalog").hexdigest()


def _span(
    span_id: str,
    *,
    source_id: str | None = None,
    in_seconds: float = 0.0,
    out_seconds: float = 4.0,
    eligibility: str = "eligible",
    disposition: str = "available",
    evidence: tuple[str, ...] = (),
    reason: str | None = None,
) -> MaterialSpanRecord:
    return MaterialSpanRecord(
        source_id=source_id or f"source-{span_id}",
        span_id=span_id,
        in_seconds=in_seconds,
        out_seconds=out_seconds,
        eligibility=eligibility,
        disposition=disposition,
        evidence=evidence,
        reason=reason,
    )


def _state(
    *,
    spans: tuple[MaterialSpanRecord, ...] | None = None,
    selected: tuple[str, ...] = (),
    alternates: tuple[str, ...] = (),
    revision: int = 0,
    parent_sha256: str | None = None,
    material_digest: str = MATERIAL_DIGEST,
) -> PlanningState:
    return PlanningState(
        contract_version="planning-state-v1",
        material_digest=material_digest,
        revision=revision,
        parent_sha256=parent_sha256,
        spans=spans if spans is not None else (_span("s1"), _span("s2")),
        selected_span_ids=selected,
        alternate_span_ids=alternates,
        story_obligations=("story.open", "story.resolve"),
        coverage_obligations=("coverage.subject",),
        music_cue_refs=("cue.intro",),
        grounding_target_refs=("target.primary",),
    )


def _select_delta(base: PlanningState) -> PlanningStateDelta:
    return PlanningStateDelta(
        contract_version="planning-state-delta-v1",
        base_sha256=base.sha256(),
        span_updates=(
            _span(
                "s1",
                disposition="selected",
                evidence=("card:s1", "visual:s1"),
                reason="carries the opening obligation",
            ),
            _span(
                "s2",
                disposition="alternate",
                evidence=("card:s2",),
                reason="fallback for the same story job",
            ),
        ),
        selected_span_ids=("s1",),
        alternate_span_ids=("s2",),
        music_cue_refs=("cue.intro", "cue.resolve"),
    )


def test_canonical_json_and_hash_are_stable_and_round_trip():
    state = _state()

    encoded = state.canonical_json()
    assert encoded == canonical_json(state)
    assert " " not in encoded
    assert "\n" not in encoded
    assert state.sha256() == canonical_sha256(state)
    assert len(state.sha256()) == 64

    restored = PlanningState.model_validate_json(encoded)
    assert restored == state
    assert restored.sha256() == state.sha256()
    assert canonical_json({"b": 2, "a": 1}) == canonical_json({"a": 1, "b": 2})


def test_canonical_json_rejects_non_string_object_keys_before_hashing():
    with pytest.raises(TypeError, match="object key"):
        canonical_json({"nested": [{1: "ambiguous"}]})
    with pytest.raises(TypeError, match="object key"):
        canonical_sha256({1: "would collide with a string key"})


def test_models_are_frozen_and_reject_unknown_fields():
    span = _span("s1")
    with pytest.raises(ValidationError, match="frozen"):
        span.disposition = "selected"

    payload = span.model_dump(mode="json")
    payload["provider_confidence"] = 0.99
    with pytest.raises(ValidationError, match="Extra inputs"):
        MaterialSpanRecord.model_validate(payload)


@pytest.mark.parametrize(
    ("in_seconds", "out_seconds"),
    [(-0.1, 1.0), (1.0, 1.0), (2.0, 1.0), (0.0, float("inf")), (0.0, float("nan"))],
)
def test_span_intervals_must_be_finite_non_empty_half_open_ranges(
    in_seconds, out_seconds
):
    with pytest.raises(ValidationError):
        _span("bad", in_seconds=in_seconds, out_seconds=out_seconds)


def test_span_ids_and_optional_evidence_are_non_empty_and_unique():
    with pytest.raises(ValidationError):
        _span("  ")
    with pytest.raises(ValidationError, match="evidence values must be unique"):
        _span("s1", evidence=("same", "same"))
    with pytest.raises(ValidationError):
        _span("s1", reason="   ")


@pytest.mark.parametrize("disposition", ["available", "selected", "alternate"])
def test_hard_invalid_span_cannot_be_available_or_selectable(disposition):
    with pytest.raises(ValidationError, match="hard_invalid span cannot"):
        _span("broken", eligibility="hard_invalid", disposition=disposition)


def test_state_rejects_duplicate_and_unknown_span_ids():
    with pytest.raises(ValidationError, match="globally unique"):
        _state(spans=(_span("same"), _span("same", source_id="another")))

    with pytest.raises(ValidationError, match="unknown IDs"):
        _state(selected=("missing",))
    with pytest.raises(ValidationError, match="unknown IDs"):
        _state(alternates=("missing",))


def test_state_rejects_duplicate_or_overlapping_selected_and_alternate_ids():
    with pytest.raises(ValidationError, match="must be unique"):
        _state(selected=("s1", "s1"))

    selected = _span("s1", disposition="selected")
    with pytest.raises(ValidationError, match="must be disjoint"):
        _state(
            spans=(selected, _span("s2")),
            selected=("s1",),
            alternates=("s1",),
        )


def test_state_requires_id_lists_and_span_dispositions_to_agree_exactly():
    with pytest.raises(ValidationError, match="selected_span_ids must exactly"):
        _state(selected=("s1",))

    with pytest.raises(ValidationError, match="alternate_span_ids must exactly"):
        _state(spans=(_span("s1", disposition="alternate"), _span("s2")))


@pytest.mark.parametrize(
    "field",
    [
        "story_obligations",
        "coverage_obligations",
        "music_cue_refs",
        "grounding_target_refs",
    ],
)
def test_authority_reference_lists_reject_duplicates(field):
    payload = _state().model_dump(mode="json")
    payload[field] = ["same", "same"]
    with pytest.raises(ValidationError, match="must be unique"):
        PlanningState.model_validate(payload)


def test_revision_parent_and_digest_are_strict():
    with pytest.raises(ValidationError, match="revision 0"):
        _state(parent_sha256="1" * 64)
    with pytest.raises(ValidationError, match="requires parent_sha256"):
        _state(revision=1)
    with pytest.raises(ValidationError):
        _state(material_digest="not-a-sha")
    with pytest.raises(ValidationError):
        _state(material_digest="A" * 64)


def test_apply_delta_uses_base_hash_cas_and_preserves_material_lineage():
    base = _state()
    successor = apply_planning_delta(base, _select_delta(base))

    assert base.revision == 0
    assert base.selected_span_ids == ()
    assert successor.revision == 1
    assert successor.parent_sha256 == base.sha256()
    assert successor.material_digest == base.material_digest
    assert successor.selected_span_ids == ("s1",)
    assert successor.alternate_span_ids == ("s2",)
    assert successor.spans[0].disposition == "selected"
    assert successor.spans[1].disposition == "alternate"
    assert successor.music_cue_refs == ("cue.intro", "cue.resolve")


def test_apply_delta_rejects_a_stale_base_hash_before_mutating_anything():
    base = _state()
    delta = _select_delta(base).model_copy(update={"base_sha256": "f" * 64})

    with pytest.raises(PlanningStateConflictError, match="base_sha256"):
        apply_planning_delta(base, delta)
    assert base.revision == 0
    assert all(span.disposition == "available" for span in base.spans)


@pytest.mark.parametrize("change", ["unknown", "source", "interval"])
def test_apply_delta_cannot_fabricate_or_rebind_material_spans(change):
    base = _state()
    if change == "unknown":
        updated = _span("new")
    elif change == "source":
        updated = _span("s1", source_id="different-source")
    else:
        updated = _span("s1", in_seconds=0.5)
    delta = PlanningStateDelta(
        contract_version="planning-state-delta-v1",
        base_sha256=base.sha256(),
        span_updates=(updated,),
    )

    with pytest.raises(PlanningStateLineageError):
        apply_planning_delta(base, delta)


def test_delta_cannot_reverse_a_hard_invalid_local_decision():
    base = _state(
        spans=(
            _span(
                "s1",
                eligibility="hard_invalid",
                disposition="rejected",
                reason="local decoder proved the interval invalid",
            ),
            _span("s2"),
        )
    )
    delta = PlanningStateDelta(
        contract_version="planning-state-delta-v1",
        base_sha256=base.sha256(),
        span_updates=(_span("s1", eligibility="eligible"),),
    )

    with pytest.raises(PlanningStateLineageError, match="cannot reverse hard_invalid"):
        apply_planning_delta(base, delta)


def test_delta_must_be_non_empty_and_cannot_repeat_span_updates():
    base = _state()
    with pytest.raises(ValidationError, match="at least one change"):
        PlanningStateDelta(
            contract_version="planning-state-delta-v1",
            base_sha256=base.sha256(),
        )
    with pytest.raises(ValidationError, match="duplicate span_id"):
        PlanningStateDelta(
            contract_version="planning-state-delta-v1",
            base_sha256=base.sha256(),
            span_updates=(_span("s1"), _span("s1")),
        )


def test_delta_rejects_a_semantic_noop_but_can_explicitly_clear_collections():
    base = _state()
    noop = PlanningStateDelta(
        contract_version="planning-state-delta-v1",
        base_sha256=base.sha256(),
        story_obligations=base.story_obligations,
    )
    with pytest.raises(PlanningStateError, match="no semantic change"):
        apply_planning_delta(base, noop)

    cleared = PlanningStateDelta(
        contract_version="planning-state-delta-v1",
        base_sha256=base.sha256(),
        story_obligations=(),
        coverage_obligations=(),
        music_cue_refs=(),
        grounding_target_refs=(),
    )
    successor = apply_planning_delta(base, cleared)
    assert successor.story_obligations == ()
    assert successor.coverage_obligations == ()
    assert successor.music_cue_refs == ()
    assert successor.grounding_target_refs == ()


def test_delta_must_update_status_and_selected_lists_coherently():
    base = _state()
    delta = PlanningStateDelta(
        contract_version="planning-state-delta-v1",
        base_sha256=base.sha256(),
        selected_span_ids=("s1",),
    )
    with pytest.raises(ValidationError, match="selected_span_ids must exactly"):
        apply_planning_delta(base, delta)


def test_writer_publishes_all_four_canonical_files_atomically(tmp_path):
    state = _state()
    work = tmp_path / "work"
    destination = write_planning_revision(
        work,
        "coarse-pass",
        state,
        request={"brief_sha256": "b" * 64, "candidate_ids": ["s1", "s2"]},
        response={"accepted": ["s1"]},
        validation={"valid": True, "state_sha256": state.sha256()},
    )

    assert destination == work / "planning" / "coarse-pass" / "rev-0"
    assert sorted(path.name for path in destination.iterdir()) == [
        "request.json",
        "response.json",
        "state.json",
        "validation.json",
    ]
    assert destination.joinpath("state.json").read_text(encoding="utf-8") == (
        state.canonical_json() + "\n"
    )
    restored = PlanningState.model_validate_json(
        destination.joinpath("state.json").read_text(encoding="utf-8")
    )
    assert restored.sha256() == state.sha256()
    assert json.loads(destination.joinpath("validation.json").read_text()) == {
        "state_sha256": state.sha256(),
        "valid": True,
    }


def test_writer_appends_only_a_contiguous_hash_bound_successor(tmp_path):
    work = tmp_path / "work"
    base = _state()
    write_planning_revision(
        work, "selection", base, request={}, response={}, validation={}
    )
    successor = apply_planning_delta(base, _select_delta(base))
    destination = write_planning_revision(
        work, "selection", successor, request={}, response={}, validation={}
    )

    assert destination.name == "rev-1"
    assert PlanningState.model_validate_json(
        destination.joinpath("state.json").read_text(encoding="utf-8")
    ) == successor


def test_writer_cannot_publish_a_forged_successor_that_revives_hard_invalid(
    tmp_path,
):
    work = tmp_path / "work"
    base = _state(
        spans=(
            _span(
                "s1",
                eligibility="hard_invalid",
                disposition="rejected",
                reason="locally proven invalid",
            ),
            _span("s2"),
        )
    )
    write_planning_revision(
        work, "eligibility", base, request={}, response={}, validation={}
    )
    forged = _state(
        spans=(
            _span("s1", eligibility="eligible", disposition="selected"),
            _span("s2"),
        ),
        selected=("s1",),
        revision=1,
        parent_sha256=base.sha256(),
    )

    with pytest.raises(PlanningStateLineageError, match="cannot be restored"):
        write_planning_revision(
            work,
            "eligibility",
            forged,
            request={},
            response={},
            validation={},
        )
    assert not (work / "planning" / "eligibility" / "rev-1").exists()


def test_writer_never_overwrites_an_existing_revision(tmp_path):
    work = tmp_path / "work"
    state = _state()
    destination = write_planning_revision(
        work,
        "visual",
        state,
        request={"first": True},
        response={},
        validation={},
    )
    before = {
        path.name: path.read_bytes() for path in destination.iterdir()
    }

    with pytest.raises(FileExistsError, match="will not be overwritten"):
        write_planning_revision(
            work,
            "visual",
            state,
            request={"second": True},
            response={},
            validation={},
        )
    after = {path.name: path.read_bytes() for path in destination.iterdir()}
    assert after == before
    assert not destination.parent.joinpath(".write.lock").exists()


@pytest.mark.parametrize(
    "failure", ["parent", "digest", "gap", "source", "interval", "catalog"]
)
def test_writer_rejects_broken_parent_digest_or_revision_lineage(tmp_path, failure):
    work = tmp_path / "work"
    base = _state()
    write_planning_revision(
        work, "joint", base, request={}, response={}, validation={}
    )
    successor = apply_planning_delta(base, _select_delta(base))
    payload = successor.model_dump(mode="json")
    if failure == "parent":
        payload["parent_sha256"] = "f" * 64
    elif failure == "digest":
        payload["material_digest"] = "e" * 64
    elif failure == "gap":
        payload["revision"] = 2
    elif failure == "source":
        payload["spans"][0]["source_id"] = "rebound-source"
    elif failure == "interval":
        payload["spans"][0]["out_seconds"] = 3.0
    else:
        payload["spans"].append(
            _span("fabricated").model_dump(mode="json")
        )
    broken = PlanningState.model_validate(payload)

    with pytest.raises(PlanningStateLineageError):
        write_planning_revision(
            work, "joint", broken, request={}, response={}, validation={}
        )
    assert not (work / "planning" / "joint" / f"rev-{broken.revision}").exists()
    assert not (work / "planning" / "joint" / ".write.lock").exists()


@pytest.mark.parametrize("stage", ["", ".", "..", "../escape", "a/b", "/tmp/x"])
def test_writer_rejects_unsafe_stage_paths(tmp_path, stage):
    with pytest.raises(ValueError, match="safe path component"):
        write_planning_revision(
            tmp_path / "work",
            stage,
            _state(),
            request={},
            response={},
            validation={},
        )


@pytest.mark.parametrize(
    "bad_sidecar",
    [{"value": object()}, {"value": float("nan")}],
)
def test_unserialisable_sidecar_has_no_storage_side_effect(tmp_path, bad_sidecar):
    work = tmp_path / "work"
    with pytest.raises((TypeError, ValueError)):
        write_planning_revision(
            work,
            "coarse",
            _state(),
            request=bad_sidecar,
            response={},
            validation={},
        )
    assert not (work / "planning").exists()


def test_writer_failure_does_not_publish_a_partial_revision(
    tmp_path, monkeypatch
):
    work = tmp_path / "work"
    original = planning_state._write_complete_file

    def fail_on_response(path: Path, text: str) -> None:
        if path.name == "response.json":
            raise OSError("simulated disk failure")
        original(path, text)

    monkeypatch.setattr(planning_state, "_write_complete_file", fail_on_response)
    with pytest.raises(OSError, match="simulated disk failure"):
        write_planning_revision(
            work,
            "shortlist",
            _state(),
            request={},
            response={},
            validation={},
        )

    stage = work / "planning" / "shortlist"
    assert not (stage / "rev-0").exists()
    assert not (stage / ".write.lock").exists()
    assert list(stage.glob(".rev-0-*")) == []


def test_writer_fails_closed_when_another_writer_owns_the_stage(tmp_path):
    work = tmp_path / "work"
    stage = work / "planning" / "music"
    stage.mkdir(parents=True)
    lock = stage / ".write.lock"
    lock.write_text("owned", encoding="utf-8")

    with pytest.raises(PlanningStateConflictError, match="another writer"):
        write_planning_revision(
            work,
            "music",
            _state(),
            request={},
            response={},
            validation={},
        )
    assert lock.read_text(encoding="utf-8") == "owned"
    assert not (stage / "rev-0").exists()


def test_writer_exclusive_rename_cannot_replace_a_racing_empty_revision(
    tmp_path, monkeypatch
):
    work = tmp_path / "work"
    competing: Path | None = None
    exclusive_rename = planning_state._rename_no_replace

    def race(source: Path, destination: Path) -> None:
        nonlocal competing
        destination.mkdir()
        competing = destination
        exclusive_rename(source, destination)

    monkeypatch.setattr(planning_state, "_rename_no_replace", race)
    with pytest.raises(FileExistsError):
        write_planning_revision(
            work,
            "racing-stage",
            _state(),
            request={},
            response={},
            validation={},
        )

    assert competing is not None
    assert competing.is_dir()
    assert list(competing.iterdir()) == []
    stage = competing.parent
    assert not (stage / ".write.lock").exists()
    assert list(stage.glob(".rev-0-*")) == []
