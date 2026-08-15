import json
from pathlib import Path
from types import SimpleNamespace

import pytest


def _confirmed(target_id: str):
    from montagewright.reference_grounding import ConfirmedFrame

    digest = "a" * 64
    return ConfirmedFrame(
        target_id=target_id,
        at_seconds=2.0,
        box=(0.1, 0.2, 0.4, 0.7),
        sighting="sighting-a",
        sighting_window=(1.0, 4.0),
        frame_pts=2_000,
        frame_sha256=digest,
        video_asset_id=f"sha256:{digest}",
        frame_time_ms=2_000,
        width=1440,
        height=810,
    )


def test_legacy_source_confirmation_cache_restores_target_lineage(tmp_path):
    from montagewright.reference_grounding import read_source_confirmation_cache

    frame = _confirmed("target.a").model_dump(mode="json")
    frame.pop("target_id")
    cache = tmp_path / "identity.json"
    cache.write_text(json.dumps({
        "target": "target.a",
        "video_sha256": "a" * 64,
        "confirmed": [frame],
    }))

    restored = read_source_confirmation_cache(cache, "target.a")

    assert restored is not None
    assert restored[0].target_id == "target.a"
    assert read_source_confirmation_cache(cache, "target.b") is None


def test_source_confirmation_writer_rejects_cross_target_frame(tmp_path):
    from montagewright.reference_grounding import write_source_confirmation_cache

    with pytest.raises(ValueError, match="frame target disagree"):
        write_source_confirmation_cache(
            tmp_path / "identity.json",
            "target.b",
            "a" * 64,
            (_confirmed("target.a"),),
        )


def test_legacy_empty_confirmation_is_not_a_durable_negative(tmp_path):
    from montagewright.reference_grounding import (
        read_source_confirmation_cache, read_source_confirmation_status,
    )

    cache = tmp_path / "identity.json"
    cache.write_text(json.dumps({
        "target": "target.a", "video_sha256": "a" * 64, "confirmed": [],
    }))

    assert read_source_confirmation_cache(cache, "target.a") is None
    assert read_source_confirmation_status(cache, "target.a") is None


def test_typed_hard_negative_confirmation_can_be_pruned(tmp_path):
    from montagewright.reference_grounding import (
        read_source_confirmation_cache, read_source_confirmation_status,
        write_source_confirmation_cache,
    )

    cache = tmp_path / "identity.json"
    write_source_confirmation_cache(
        cache, "target.a", "a" * 64, (),
        status="hard_negative", reason="two independent frames excluded it",
    )

    assert read_source_confirmation_cache(cache, "target.a") == ()
    assert read_source_confirmation_status(cache, "target.a") == "hard_negative"


def test_target_keyed_confirmation_never_routes_a_box_to_b():
    from montagewright.pipeline import _confirmed_target_frames

    a = _confirmed("target.a")
    b = _confirmed("target.b")
    confirmed = {"C1": {"target.a": (a,), "target.b": (b,)}}

    assert _confirmed_target_frames(confirmed, "C1", "target.a") == (a,)
    assert _confirmed_target_frames(confirmed, "C1", "target.b") == (b,)
    # The transitional source-only shape is filtered by the frame lineage.
    assert _confirmed_target_frames({"C1": (a,)}, "C1", "target.b") == ()


def test_clientless_confirmed_seed_still_runs_local_geometry(monkeypatch, tmp_path):
    from montagewright.executor import Source
    from montagewright.pipeline import Report, _reference_subject_samples

    monkeypatch.setattr(
        "montagewright.pipeline._preflight_sam_checkpoint", lambda path: path
    )
    handed = {}

    def local_geometry(*args, **kwargs):
        handed.update(kwargs)
        return ([{"present": True}], [1.0], ((2.0, (0.1, 0.2, 0.4, 0.7)),))

    monkeypatch.setattr(
        "montagewright.pipeline._geometry_from_confirmed", local_geometry
    )
    report = Report()
    result = _reference_subject_samples(
        Source("C1", tmp_path / "C1.mp4", 5.0, 1920, 1080),
        SimpleNamespace(
            clip_id="k00", approx_in_seconds=1.0, approx_out_seconds=4.0
        ),
        "target.a",
        spec=SimpleNamespace(),
        client=None,
        upload_cache=None,
        report=report,
        work=tmp_path,
        output=None,
        discoveries={},
        checkpoint=tmp_path / "sam.pt",
        confirmed=(_confirmed("target.a"),),
    )

    assert result[0] == [{"present": True}]
    assert handed["validation_mode"] == "single_seed_continuity"


def test_missing_checkpoint_fails_before_exact_frame_work(monkeypatch, tmp_path):
    from montagewright.executor import Source
    from montagewright.pipeline import Report, _reference_subject_samples

    monkeypatch.setattr(
        "montagewright.reference_grounding.inspect_video_lineage",
        lambda path: (_ for _ in ()).throw(
            AssertionError("paid-stage preparation must not start")
        ),
    )
    report = Report()
    with pytest.raises(RuntimeError, match="requires SAM/local geometry"):
        _reference_subject_samples(
            Source("C1", tmp_path / "C1.mp4", 5.0, 1920, 1080),
            SimpleNamespace(
                clip_id="k00", approx_in_seconds=1.0, approx_out_seconds=4.0
            ),
            "target.a",
            spec=SimpleNamespace(),
            client=object(),
            upload_cache=None,
            report=report,
            work=tmp_path,
            output=None,
            discoveries={},
            checkpoint=tmp_path / "missing.pt",
        )

    assert report.reference_grounding["k00"]["status"] == (
        "local_geometry_unavailable"
    )


def test_clientless_final_window_reads_exact_cache_before_refusing(
    monkeypatch, tmp_path
):
    from montagewright import reference_grounding as grounding
    from montagewright.executor import Source
    from montagewright.pipeline import Report, _reference_subject_samples
    from montagewright.reframe import Observation

    monkeypatch.setattr(
        "montagewright.pipeline._preflight_sam_checkpoint", lambda path: path
    )
    digest = "c" * 64
    video = SimpleNamespace(
        asset_id=f"sha256:{digest}", content_sha256=digest, duration_ms=8_000
    )
    monkeypatch.setattr(grounding, "inspect_video_lineage", lambda path: video)

    def materialize(path, requested_time_ms, destination, max_width=None):
        return SimpleNamespace(lineage=SimpleNamespace(
            frame_pts=requested_time_ms,
            frame_time_ms=requested_time_ms,
            video_asset_id=f"sha256:{digest}",
            frame_sha256=f"{requested_time_ms:064x}"[-64:],
            width=1440,
            height=810,
        ))

    monkeypatch.setattr(grounding, "materialize_frame_at_time", materialize)
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
                identity_evidence=("same instance",),
                confidence=0.9,
            ),
        )
        for at in (1_600, 2_500, 3_400)
    )
    batch = SimpleNamespace(
        query_lock_sha256="lock",
        grounding_spec_sha256="spec",
        matched_anchor_count=3,
        sam_seed_evaluations=lambda: evaluations,
    )
    monkeypatch.setattr(
        grounding.ExactFrameBBoxBatchResult,
        "model_validate",
        lambda value: batch,
    )
    monkeypatch.setattr(
        grounding,
        "decide_exact_frame_bboxes",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("clientless cache hit must not ask a model")
        ),
    )
    monkeypatch.setattr(
        "montagewright.pipeline._track_subject",
        lambda *args, **kwargs: (
            [
                Observation(0.6, 0.2, 0.3, 0.2, 0.4),
                Observation(1.0, 0.4, 0.5, 0.2, 0.4),
                Observation(2.4, 0.8, 0.7, 0.2, 0.4),
            ],
            {"tracked": 3},
        ),
    )
    spec = SimpleNamespace(
        definition_sha256=lambda: "5" * 64,
        identity_lock=SimpleNamespace(
            query_id="grounding:target.a",
            definition_sha256=lambda: "1" * 64,
            identity=SimpleNamespace(target=lambda target_id: (_ for _ in ()).throw(
                ValueError(target_id)
            )),
        ),
    )
    key = "d" * 64
    monkeypatch.setattr(
        "montagewright.pipeline._final_exact_cache_key",
        lambda *args, **kwargs: key,
    )
    memory = tmp_path / "memory"
    memory.mkdir()
    (memory / f"exact-v2-{key[:24]}.json").write_text(json.dumps({
        "cache_contract_sha256": key,
        "batch": {},
    }))

    report = Report()
    boxes, _, _ = _reference_subject_samples(
        Source("C1", tmp_path / "C1.mp4", 5.0, 1920, 1080),
        SimpleNamespace(
            clip_id="k00", approx_in_seconds=1.0, approx_out_seconds=4.0
        ),
        "target.a",
        spec=spec,
        client=None,
        upload_cache=None,
        report=report,
        work=tmp_path,
        output=None,
        discoveries={},
        checkpoint=tmp_path / "sam.pt",
        memory=memory,
    )

    assert len(boxes) == 3
    assert report.reference_grounding["k00"]["status"] == "sam_geometry_validated"


def test_final_exact_cache_key_covers_the_complete_request_contract(monkeypatch):
    from montagewright import reference_grounding as grounding
    from montagewright.pipeline import _final_exact_cache_key

    spec = SimpleNamespace(definition_sha256=lambda: "5" * 64)
    prepared = [(SimpleNamespace(lineage=SimpleNamespace(
        frame_pts=10,
        frame_sha256="a" * 64,
    )), "cut_k00")]
    base = _final_exact_cache_key(spec, "target.a", prepared)

    assert _final_exact_cache_key(
        spec, "target.a", prepared, model_id="different-model"
    ) != base
    assert _final_exact_cache_key(
        spec, "target.a", prepared, reference_resolution="medium"
    ) != base
    assert _final_exact_cache_key(
        spec, "target.a", prepared, frame_resolution="medium"
    ) != base
    assert _final_exact_cache_key(
        spec, "target.a", prepared, minimum_matched_anchors=3
    ) != base
    assert _final_exact_cache_key(
        spec,
        "target.a",
        prepared,
        local_validator_version="different-local-validator",
    ) != base
    discovery_a = SimpleNamespace(candidate=lambda candidate_id: SimpleNamespace(
        model_dump=lambda **kwargs: {
            "candidate_id": candidate_id, "start_ms": 0, "end_ms": 1_000,
        }
    ))
    discovery_b = SimpleNamespace(candidate=lambda candidate_id: SimpleNamespace(
        model_dump=lambda **kwargs: {
            "candidate_id": candidate_id, "start_ms": 500, "end_ms": 1_500,
        }
    ))
    assert _final_exact_cache_key(
        spec, "target.a", prepared, discovery=discovery_a
    ) != _final_exact_cache_key(
        spec, "target.a", prepared, discovery=discovery_b
    )

    original_prompt = grounding._read_prompt
    monkeypatch.setattr(grounding, "_read_prompt", lambda: original_prompt() + " changed")
    assert _final_exact_cache_key(spec, "target.a", prepared) != base
    monkeypatch.setattr(grounding, "_read_prompt", original_prompt)

    original_schema = grounding._exact_frame_batch_schema
    monkeypatch.setattr(
        grounding,
        "_exact_frame_batch_schema",
        lambda *args, **kwargs: {**original_schema(*args, **kwargs), "x-test": True},
    )
    assert _final_exact_cache_key(spec, "target.a", prepared) != base


def test_final_exact_cache_rejects_legacy_raw_or_wrong_contract(tmp_path):
    from montagewright.pipeline import _read_final_exact_cache

    class ResultType:
        @staticmethod
        def model_validate(value):
            return value

    cache = tmp_path / "exact.json"
    cache.write_text(json.dumps({"contract_version": "legacy-raw-batch"}))
    assert _read_final_exact_cache(cache, "a" * 64, ResultType) is None

    cache.write_text(json.dumps({
        "cache_contract_sha256": "b" * 64,
        "batch": {"validated": True},
    }))
    assert _read_final_exact_cache(cache, "a" * 64, ResultType) is None

    cache.write_text(json.dumps({
        "cache_contract_sha256": "a" * 64,
        "batch": {"validated": True},
    }))
    assert _read_final_exact_cache(cache, "a" * 64, ResultType) == {
        "validated": True
    }


def test_one_shot_collects_every_entity_fault_in_one_pass(monkeypatch, tmp_path):
    from montagewright.executor import Source
    from montagewright.pipeline import (
        ReferenceShotsUnusable,
        Report,
        follow_subjects,
    )
    from montagewright.schema import Clip, EDL, Look, Reframe

    asked = []

    def absent(source, clip, target_id, **kwargs):
        asked.append(target_id)
        return [], [], ()

    monkeypatch.setattr(
        "montagewright.pipeline._reference_subject_samples", absent
    )
    edl = EDL(project_id="p", clips=[Clip(
        clip_id="k00",
        source_id="C1",
        approx_in_seconds=0.0,
        approx_out_seconds=2.0,
        reframe=Reframe(looks=[
            Look(at="A", entity_id="target.a"),
            Look(at="B", entity_id="target.b"),
        ]),
    )])

    with pytest.raises(ReferenceShotsUnusable) as raised:
        follow_subjects(
            edl,
            {"C1": Source("C1", tmp_path / "C1.mp4", 3.0, 1920, 1080)},
            target_aspect=9 / 16,
            report=Report(),
            checkpoint=tmp_path / "sam.pt",
            grounding_spec=object(),
        )

    assert asked == ["target.a", "target.b"]
    assert [fault.entity_id for fault in raised.value.faults] == [
        "target.a", "target.b",
    ]


def test_reference_report_keeps_each_target_and_projects_latest():
    from montagewright.pipeline import Report, _set_target_grounding

    report = Report()
    _set_target_grounding(report, "k00", "target.a", {"status": "validated-a"})
    _set_target_grounding(report, "k00", "target.b", {"status": "validated-b"})

    record = report.reference_grounding["k00"]
    assert record["status"] == "validated-b"
    assert record["targets"]["target.a"]["status"] == "validated-a"
    assert record["targets"]["target.b"]["status"] == "validated-b"


def test_screen_keeps_b_target_when_a_is_absent(monkeypatch, tmp_path):
    from montagewright import reference_grounding as grounding
    from montagewright.cli import _screen_material_identity
    from montagewright.planner import MaterialItem
    from montagewright.spans import Span

    proxy = tmp_path / "C1.mp4"
    proxy.write_bytes(b"proxy")
    discovery = SimpleNamespace(
        target_summaries=(
            SimpleNamespace(target_id="target.a", verdict="absent", reason="no A"),
            SimpleNamespace(target_id="target.b", verdict="present", reason="B seen"),
        ),
        candidates=(SimpleNamespace(
            target_id="target.b",
            identity_status="matched_target",
            start_ms=1_000,
            end_ms=3_000,
        ),),
    )
    monkeypatch.setattr(
        grounding, "remembered_discovery", lambda *args, **kwargs: (discovery, None)
    )
    spec = SimpleNamespace(identity_lock=SimpleNamespace(
        framing=SimpleNamespace(required_target_ids=("target.a", "target.b")),
        identity=SimpleNamespace(targets=()),
    ))
    item = MaterialItem(
        source_id="C1",
        duration_seconds=5.0,
        summary="A and B table",
        proxy=proxy,
        spans=(Span("C1:s00", "C1", 0.0, 5.0),),
    )

    kept, _, _ = _screen_material_identity(
        [item],
        spec,
        client=object(),
        cache=None,
        ledger=SimpleNamespace(check=lambda: None, spent_usd=0.0),
        library=tmp_path,
    )

    screened = kept[0]
    assert screened.carries_identity is True
    assert screened.identity_absent_targets == ("target.a",)
    assert dict(screened.identity_windows_by_target)["target.b"] == ((1.0, 3.0),)
    assert (screened.spans[0].starts_seconds, screened.spans[0].ends_seconds) == (
        1.0, 3.0,
    )


def test_screen_never_merges_two_sightings_across_the_absent_gap(
    monkeypatch, tmp_path
):
    from montagewright import reference_grounding as grounding
    from montagewright.cli import _screen_material_identity
    from montagewright.planner import MaterialItem
    from montagewright.spans import Span

    proxy = tmp_path / "C1.mp4"
    proxy.write_bytes(b"proxy")
    discovery = SimpleNamespace(
        target_summaries=(SimpleNamespace(
            target_id="target.a", verdict="present", reason="two appearances"
        ),),
        candidates=tuple(
            SimpleNamespace(
                target_id="target.a",
                identity_status="matched_target",
                start_ms=starts,
                end_ms=ends,
            )
            for starts, ends in ((0, 2_000), (8_000, 10_000))
        ),
    )
    monkeypatch.setattr(
        grounding, "remembered_discovery", lambda *args, **kwargs: (discovery, None)
    )
    spec = SimpleNamespace(identity_lock=SimpleNamespace(
        framing=SimpleNamespace(required_target_ids=("target.a",)),
        identity=SimpleNamespace(targets=()),
    ))
    item = MaterialItem(
        source_id="C1",
        duration_seconds=12.0,
        summary="two appearances",
        proxy=proxy,
        spans=(Span("C1:s00", "C1", 0.0, 12.0),),
    )

    kept, _, _ = _screen_material_identity(
        [item],
        spec,
        client=object(),
        cache=None,
        ledger=SimpleNamespace(check=lambda: None, spent_usd=0.0),
        library=tmp_path,
    )

    assert [
        (span.starts_seconds, span.ends_seconds) for span in kept[0].spans
    ] == [(0.0, 2.0), (8.0, 10.0)]
    assert dict(kept[0].identity_windows_by_target)["target.a"] == (
        (0.0, 2.0), (8.0, 10.0),
    )


def test_planner_gates_the_claim_by_target_and_exact_span():
    from montagewright.planner import MaterialItem, _material_can_claim_target
    from montagewright.spans import Span

    inside = Span("C1:s00", "C1", 1.0, 3.0)
    gap = Span("C1:s01", "C1", 4.0, 5.0)
    item = MaterialItem(
        source_id="C1",
        duration_seconds=6.0,
        summary="B only",
        spans=(inside, gap),
        carries_identity=True,
        identity_absent_targets=("target.a",),
        identity_windows_by_target=(
            ("target.a", ()),
            ("target.b", ((1.0, 3.0),)),
        ),
    )

    assert not _material_can_claim_target(item, "target.a", inside)
    assert _material_can_claim_target(item, "target.b", inside)
    assert not _material_can_claim_target(item, "target.b", gap)


def test_final_per_target_tracks_project_as_track_confirmed():
    from montagewright.cli import _project_track_confirmed

    selection = {"shots": [{
        "looks": [
            {"entity_id": "target.a"}, {"entity_id": "target.b"},
        ],
        "identity_status": "source_confirmed",
        "identity_issue": "pending",
    }]}
    report = SimpleNamespace(reference_grounding={"k00": {"targets": {
        "target.a": {"status": "sam_geometry_validated"},
        "target.b": {"status": "sam_geometry_validated"},
    }}})

    _project_track_confirmed(selection, report)

    assert selection["shots"][0]["identity_status"] == "track_confirmed"
    assert selection["shots"][0]["identity_issue"] == ""
