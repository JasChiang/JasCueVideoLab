import pytest

from montagewright.cost import Ledger
from montagewright.reference_grounding import (
    CandidateDiscoveryResult,
    CrossAssetExactFrameItem,
    ReferenceGroundingError,
    confirmed_frame_from_validated_seed,
    decide_cross_asset_exact_frame_bboxes,
    prepare_source_identity_seed,
    validate_exact_frame_payload,
    validate_cross_asset_exact_frame_payload,
)
from tests.test_reference_grounding import (
    _Cache,
    _Client,
    _candidate_payload,
    _exact_decision_payload,
    _exact_frame,
    _video_lineage,
    _write_spec,
)


def _two_assets(tmp_path):
    spec = _write_spec(tmp_path)
    items = []
    videos = []
    frames = []
    for name, digest in (("a", "a" * 64), ("b", "b" * 64)):
        folder = tmp_path / name
        folder.mkdir()
        video = _video_lineage(digest)
        discovery = CandidateDiscoveryResult.model_validate(
            _candidate_payload(spec, video)
        )
        # Deliberately the same candidate id, wall-clock time and source PTS.
        frame = _exact_frame(folder, video, 2_000)
        items.append(CrossAssetExactFrameItem(
            discovery=discovery,
            candidate_id="candidate.001",
            frame=frame,
        ))
        videos.append(video)
        frames.append(frame)
    return spec, tuple(items), tuple(videos), tuple(frames)


def _v2_result(item, item_id, spec, video, frame, *, verdict="matched_target"):
    return {
        "item_id": item_id,
        "decision": _exact_decision_payload(
            spec, video, frame, verdict=verdict,
            candidate_id=item.candidate_id,
        ),
    }


def test_item_id_disambiguates_same_candidate_and_pts_across_assets(tmp_path):
    spec, items, _, _ = _two_assets(tmp_path)

    item_ids = [item.item_id(spec, "target.primary") for item in items]

    assert len(set(item_ids)) == 2
    assert all(item_id.startswith("xf_") and len(item_id) == 67 for item_id in item_ids)


def test_swapped_asset_decisions_are_rejected_per_item(tmp_path):
    spec, items, videos, frames = _two_assets(tmp_path)
    item_ids = [item.item_id(spec, "target.primary") for item in items]
    # Keep routing ids in place but swap the full immutable decision echoes.
    payload = {
        "contract_version": "reference-exact-frame-cross-asset-response-v2",
        "results": [
            _v2_result(items[0], item_ids[0], spec, videos[1], frames[1]),
            _v2_result(items[1], item_ids[1], spec, videos[0], frames[0]),
        ],
    }

    outcomes, protocol = validate_cross_asset_exact_frame_payload(
        payload, spec=spec, target_id="target.primary", items=items
    )

    assert protocol == ()
    assert [outcome.status for outcome in outcomes] == [
        "retry_required", "retry_required"
    ]
    assert all(
        outcome.failures[0].code == "lineage_mismatch"
        and "video_asset_id" in outcome.failures[0].fields
        for outcome in outcomes
    )


def test_missing_item_retries_only_that_singleton(tmp_path):
    spec, items, videos, frames = _two_assets(tmp_path)
    item_ids = [item.item_id(spec, "target.primary") for item in items]
    first_call = {
        "contract_version": "reference-exact-frame-cross-asset-response-v2",
        "results": [
            _v2_result(items[0], item_ids[0], spec, videos[0], frames[0]),
        ],
    }
    singleton_retry = _exact_decision_payload(
        spec, videos[1], frames[1], candidate_id=items[1].candidate_id
    )
    client = _Client([first_call, singleton_retry])
    cache = _Cache()

    result, usage = decide_cross_asset_exact_frame_bboxes(
        spec,
        "target.primary",
        items,
        client=client,
        cache=cache,
    )

    assert [outcome.status for outcome in result.outcomes] == [
        "validated", "validated"
    ]
    assert [outcome.attempts for outcome in result.outcomes] == [1, 2]
    assert len(client.interactions.calls) == 2
    assert usage.input_tokens == 200
    initial_images = [
        part for part in client.interactions.calls[0]["input"]
        if part["type"] == "image"
    ]
    retry_images = [
        part for part in client.interactions.calls[1]["input"]
        if part["type"] == "image"
    ]
    assert len(initial_images) == 3, "one reference plus both exact items"
    assert len(retry_images) == 2, "one reference plus only the failed item"


def test_valid_uncertain_is_not_retried(tmp_path):
    spec, items, videos, frames = _two_assets(tmp_path)
    item_ids = [item.item_id(spec, "target.primary") for item in items]
    response = {
        "contract_version": "reference-exact-frame-cross-asset-response-v2",
        "results": [
            _v2_result(
                item, item_id, spec, video, frame, verdict="uncertain"
            )
            for item, item_id, video, frame in zip(
                items, item_ids, videos, frames, strict=True
            )
        ],
    }
    client = _Client(response)

    result, _ = decide_cross_asset_exact_frame_bboxes(
        spec, "target.primary", items, client=client, cache=_Cache()
    )

    assert len(client.interactions.calls) == 1
    assert all(outcome.status == "validated" for outcome in result.outcomes)
    assert all(
        outcome.evaluation.decision.verdict == "uncertain"
        for outcome in result.outcomes
        if outcome.evaluation is not None
    )


def test_anchors_never_combine_across_assets(tmp_path):
    spec, items, videos, frames = _two_assets(tmp_path)
    item_ids = [item.item_id(spec, "target.primary") for item in items]
    response = {
        "contract_version": "reference-exact-frame-cross-asset-response-v2",
        "results": [
            _v2_result(item, item_id, spec, video, frame)
            for item, item_id, video, frame in zip(
                items, item_ids, videos, frames, strict=True
            )
        ],
    }
    result, _ = decide_cross_asset_exact_frame_bboxes(
        spec, "target.primary", items,
        client=_Client(response), cache=_Cache(),
    )

    assert not hasattr(result, "sam_ready")
    for video in videos:
        single = result.to_single_asset_batch(video.asset_id)
        assert len(single.evaluations) == 1
        assert not single.sam_ready
        with pytest.raises(ReferenceGroundingError, match="at least 2 matched"):
            single.sam_seed_evaluations()


def test_every_asset_preflights_before_upload_or_spend(tmp_path):
    spec, items, _, _ = _two_assets(tmp_path)
    damaged = list(items)
    damaged[1].frame.path.write_bytes(b"changed after lineage")
    cache = _Cache()
    client = _Client({})
    ledger = Ledger(cap_usd=1.0)

    with pytest.raises(ReferenceGroundingError, match="content hash"):
        decide_cross_asset_exact_frame_bboxes(
            spec, "target.primary", damaged,
            client=client, cache=cache, ledger=ledger,
        )

    assert cache.paths == []
    assert client.interactions.calls == []
    assert not ledger.entries and not ledger.reservations


def test_cross_asset_without_client_does_not_iterate_items(tmp_path):
    spec = _write_spec(tmp_path)

    class Poison:
        def __iter__(self):
            raise AssertionError("offline path must not iterate cross-asset items")

    assert decide_cross_asset_exact_frame_bboxes(
        spec, "target.primary", Poison(), client=None
    ) is None


def test_prepare_source_seed_and_build_confirmed_frame(monkeypatch, tmp_path):
    import montagewright.reference_grounding as grounding

    spec = _write_spec(tmp_path)
    video = _video_lineage("c" * 64)
    discovery = CandidateDiscoveryResult.model_validate(
        _candidate_payload(spec, video)
    )
    source = tmp_path / "master.mp4"
    source.write_bytes(b"master")
    exact_dir = tmp_path / "decoded"
    exact_dir.mkdir()
    exact = _exact_frame(exact_dir, video, 4_000)
    monkeypatch.setattr(grounding, "inspect_video_lineage", lambda path: video)
    monkeypatch.setattr(
        grounding, "materialize_frame_at_time",
        lambda *args, **kwargs: exact,
    )

    prepared = prepare_source_identity_seed(
        source, spec, discovery, "target.primary", tmp_path / "seeds"
    )

    assert prepared is not None
    assert prepared.sighting == "candidate.001"
    assert prepared.sighting_window == (1.0, 7.0)
    assert prepared.item.frame.lineage.video_asset_id == video.asset_id
    payload = _exact_decision_payload(
        spec,
        video,
        exact,
        candidate_id="sighting",
    )
    decision = validate_exact_frame_payload(
        payload,
        spec=spec,
        discovery=prepared.item.discovery,
        candidate=prepared.item.candidate(),
        frame=prepared.item.frame.lineage,
    )
    from montagewright.reference_grounding import ExactFrameBBoxEvaluation

    evaluation = ExactFrameBBoxEvaluation(
        lineage=prepared.item.frame.lineage,
        decision=decision,
    )
    confirmed = confirmed_frame_from_validated_seed(prepared, evaluation)

    assert confirmed.sighting == "candidate.001"
    assert confirmed.sighting_window == (1.0, 7.0)
    assert confirmed.frame_pts == exact.lineage.frame_pts
    assert confirmed.video_asset_id == video.asset_id
    assert confirmed.seed_risk_flags == ()


def test_confirmed_frame_helper_rejects_another_asset(monkeypatch, tmp_path):
    import montagewright.reference_grounding as grounding
    from montagewright.reference_grounding import ExactFrameBBoxEvaluation

    spec, items, videos, frames = _two_assets(tmp_path)
    source = tmp_path / "master.mp4"
    source.write_bytes(b"master")
    monkeypatch.setattr(
        grounding, "inspect_video_lineage", lambda path: videos[0]
    )
    monkeypatch.setattr(
        grounding, "materialize_frame_at_time", lambda *args, **kwargs: frames[0]
    )
    prepared = prepare_source_identity_seed(
        source,
        spec,
        items[0].discovery,
        "target.primary",
        tmp_path / "seeds",
    )
    wrong_decision = validate_exact_frame_payload(
        _exact_decision_payload(spec, videos[1], frames[1]),
        spec=spec,
        discovery=items[1].discovery,
        candidate=items[1].candidate(),
        frame=frames[1].lineage,
    )
    wrong = ExactFrameBBoxEvaluation(
        lineage=frames[1].lineage, decision=wrong_decision
    )

    with pytest.raises(ReferenceGroundingError, match="different exact frame"):
        confirmed_frame_from_validated_seed(prepared, wrong)
