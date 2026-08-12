from __future__ import annotations

import json
import subprocess
from types import SimpleNamespace

import pytest

import montagewright.reference_grounding as grounding
from montagewright.cost import Ledger
from montagewright.measure.models import (
    EvidenceAnchor,
    EvidenceApprovalSource,
    EvidenceClaimSource,
    EvidenceFramingObligations,
    EvidenceIdentityContract,
    EvidenceQueryApprovalProvenance,
    EvidenceQueryLock,
    EvidenceQueryProvenance,
    EvidenceTargetIdentity,
    Rational,
)
from montagewright.reference_grounding import (
    CandidateDiscoveryResult,
    ExactFrameLineage,
    ExactFrameMaterial,
    ReferenceGroundingError,
    ReferenceGroundingSpec,
    ReferenceImageSpec,
    VideoAssetLineage,
    decide_exact_frame_bbox,
    decide_exact_frame_bboxes,
    discover_reference_candidates,
    load_grounding_spec,
    materialize_frame_at_pts,
    materialize_frame_at_time,
    reference_prompt_parts,
    validate_candidate_payload,
    validate_exact_frame_payload,
)
from montagewright.uploads import content_hash


def _lock(
    anchor_hash: str,
    negative_anchor_hash: str | None = None,
) -> EvidenceQueryLock:
    return EvidenceQueryLock(
        query_id="query:reference:001",
        revision=1,
        editorial_goal="Keep the approved instance distinct from similar objects.",
        identity=EvidenceIdentityContract(
            targets=(
                EvidenceTargetIdentity(
                    target_id="target.primary",
                    target_description="the approved instance",
                    identity_cues=("distinctive long-edge hinge", "small corner mark"),
                    positive_anchors=(
                        EvidenceAnchor(
                            frame_id="reference.frame.001",
                            crop_sha256=anchor_hash,
                        ),
                    ),
                    stable_exclusions=("flat object without a hinge",),
                    negative_anchors=(
                        (
                            EvidenceAnchor(
                                frame_id="reference.frame.negative.001",
                                crop_sha256=negative_anchor_hash,
                            ),
                        )
                        if negative_anchor_hash is not None
                        else ()
                    ),
                ),
            )
        ),
        framing=EvidenceFramingObligations(
            required_target_ids=("target.primary",),
            framing_intent="Keep the approved instance recognizable.",
        ),
        claim_source=EvidenceClaimSource.HUMAN_REVIEW,
        provenance=EvidenceQueryProvenance(
            created_at="2026-08-12T00:00:00Z",
            created_by="reviewer:001",
        ),
        approval=EvidenceQueryApprovalProvenance(
            approved_at="2026-08-12T00:01:00Z",
            approved_by="reviewer:001",
            approval_source=EvidenceApprovalSource.HUMAN_REVIEW,
        ),
    )


def _write_spec(
    tmp_path,
    *,
    annotated: bool = True,
    include_negative: bool = False,
) -> ReferenceGroundingSpec:
    raw = tmp_path / "raw-reference.jpg"
    visible = tmp_path / ("annotated-reference.jpg" if annotated else raw.name)
    raw.write_bytes(b"approved raw crop")
    if annotated:
        visible.write_bytes(b"approved raw crop plus visible annotation")
    anchor_hash = content_hash(raw)
    negative = tmp_path / "hard-negative-reference.jpg"
    negative_hash = None
    if include_negative:
        negative.write_bytes(b"approved hard negative crop")
        negative_hash = content_hash(negative)
    references = [
        {
            "target_id": "target.primary",
            "polarity": "positive",
            "frame_id": "reference.frame.001",
            "path": visible.name,
            "anchor_crop_sha256": anchor_hash,
            "content_sha256": content_hash(visible),
            "mime_type": "image/jpeg",
            "presentation": "annotated" if annotated else "raw",
        }
    ]
    if include_negative:
        references.append(
            {
                "target_id": "target.primary",
                "polarity": "negative",
                "frame_id": "reference.frame.negative.001",
                "path": negative.name,
                "anchor_crop_sha256": negative_hash,
                "content_sha256": negative_hash,
                "mime_type": "image/jpeg",
                "presentation": "raw",
            }
        )
    payload = {
        "contract_version": "reference-grounding-spec-v1",
        "identity_lock": _lock(anchor_hash, negative_hash).model_dump(mode="json"),
        "reference_images": references,
    }
    spec_path = tmp_path / "grounding.json"
    spec_path.write_text(json.dumps(payload), encoding="utf-8")
    return load_grounding_spec(spec_path)


def _video_lineage(video_hash: str) -> VideoAssetLineage:
    return VideoAssetLineage(
        asset_id=f"sha256:{video_hash}",
        content_sha256=video_hash,
        duration_ms=10_000,
        source_start_pts=100,
        source_time_base=Rational(numerator=1, denominator=1000),
        display_width=1920,
        display_height=1080,
    )


def _candidate_payload(spec, video, *, lock_hash=None, end_ms=7_000):
    return {
        "contract_version": "reference-candidate-discovery-v1",
        "query_id": spec.identity_lock.query_id,
        "query_lock_sha256": lock_hash or spec.identity_lock.definition_sha256(),
        "grounding_spec_sha256": spec.definition_sha256(),
        "video_asset_id": video.asset_id,
        "video_sha256": video.content_sha256,
        "duration_ms": video.duration_ms,
        "candidates": [
            {
                "candidate_id": "candidate.001",
                "target_id": "target.primary",
                "start_ms": 1_000,
                "end_ms": end_ms,
                "recommended_seed_ms": 4_000,
                "identity_status": "matched_target",
                "confidence": 0.91,
                "visible_state": "changed configuration",
                "visibility_state": "full",
                "occlusion_state": "none",
                "frame_entry_ms": None,
                "frame_exit_ms": None,
                "identity_evidence": ["same long-edge hinge"],
                "exclusion_evidence": [],
            }
        ],
        "target_summaries": [
            {
                "target_id": "target.primary",
                "verdict": "present",
                "reason": "The persistent mark and hinge agree.",
            }
        ],
        "warnings": ["coarse video sampling"],
    }


def test_candidate_visibility_facts_are_categorical_and_bounded(tmp_path):
    spec = _write_spec(tmp_path)
    video = _video_lineage("a" * 64)
    payload = _candidate_payload(spec, video)
    candidate = payload["candidates"][0]
    candidate.update({
        "visibility_state": "entering",
        "occlusion_state": "minor",
        "frame_entry_ms": 1_500,
        "frame_exit_ms": 6_500,
    })
    result = validate_candidate_payload(
        payload,
        spec=spec,
        video=video,
        target_ids=("target.primary",),
    )
    assert result.candidates[0].visibility_state == "entering"
    assert result.candidates[0].frame_exit_ms == 6_500

    candidate["frame_exit_ms"] = 9_000
    with pytest.raises(ReferenceGroundingError, match="frame_exit_ms"):
        validate_candidate_payload(
            payload,
            spec=spec,
            video=video,
            target_ids=("target.primary",),
        )


def test_exact_visibility_is_evidence_not_a_blank_matched_box(tmp_path):
    spec = _write_spec(tmp_path)
    video = _video_lineage("b" * 64)
    discovery = CandidateDiscoveryResult.model_validate(
        _candidate_payload(spec, video)
    )
    frame = _exact_frame(tmp_path, video, 4_000)
    payload = _exact_decision_payload(spec, video, frame)
    payload.update({
        "visibility_state": "partial",
        "occlusion_state": "minor",
        "touches_frame_edges": ["right"],
    })
    decision = validate_exact_frame_payload(
        payload,
        spec=spec,
        discovery=discovery,
        candidate=discovery.candidate("candidate.001"),
        frame=frame.lineage,
    )
    assert decision.visibility_state == "partial"
    assert decision.touches_frame_edges == ("right",)

    payload["visibility_state"] = "unknown"
    with pytest.raises(ReferenceGroundingError, match="visibility_state"):
        validate_exact_frame_payload(
            payload,
            spec=spec,
            discovery=discovery,
            candidate=discovery.candidate("candidate.001"),
            frame=frame.lineage,
        )


def _exact_frame(tmp_path, video, time_ms: int) -> ExactFrameMaterial:
    path = tmp_path / f"exact-{time_ms}.jpg"
    path.write_bytes(f"exact decoded frame at {time_ms}".encode())
    return ExactFrameMaterial(
        path=path,
        lineage=ExactFrameLineage(
            video_asset_id=video.asset_id,
            video_sha256=video.content_sha256,
            source_start_pts=video.source_start_pts,
            source_time_base=video.source_time_base,
            requested_time_ms=time_ms,
            frame_time_ms=time_ms,
            frame_pts=video.source_start_pts + time_ms,
            frame_sha256=content_hash(path),
            width=video.display_width,
            height=video.display_height,
        ),
    )


def _exact_decision_payload(
    spec,
    video,
    frame,
    *,
    verdict="matched_target",
    candidate_id="candidate.001",
):
    matched = verdict == "matched_target"
    hard_negative = verdict == "hard_negative"
    return {
        "contract_version": "reference-exact-frame-bbox-v1",
        "query_id": spec.identity_lock.query_id,
        "query_lock_sha256": spec.identity_lock.definition_sha256(),
        "grounding_spec_sha256": spec.definition_sha256(),
        "video_asset_id": video.asset_id,
        "video_sha256": video.content_sha256,
        "target_id": "target.primary",
        "candidate_id": candidate_id,
        "frame_pts": frame.lineage.frame_pts,
        "frame_time_ms": frame.lineage.frame_time_ms,
        "frame_sha256": frame.lineage.frame_sha256,
        "width": frame.lineage.width,
        "height": frame.lineage.height,
        "verdict": verdict,
        "confidence": 0.95 if matched else 0.4,
        "native_box_yxyx_1000": [100, 200, 700, 800] if matched else None,
        "visibility_state": "full" if matched else "unknown",
        "occlusion_state": "none" if matched else "unknown",
        "touches_frame_edges": [],
        "identity_evidence": ["same hinge and corner mark"] if matched else [],
        "exclusion_evidence": ["stable exclusion is visible"]
        if hard_negative
        else [],
        "reason": "The exact frame was judged independently.",
    }


class _Cache:
    def __init__(self):
        self.paths = []

    def uri_for(self, path, client, *, mime_type):
        del client
        self.paths.append((path, mime_type))
        return f"files://{path.name}", False


class _Models:
    def count_tokens(self, **kwargs):
        self.last = kwargs
        return SimpleNamespace(total_tokens=100)


class _Interactions:
    def __init__(self, payload):
        self.payloads = payload if isinstance(payload, list) else [payload]
        self.calls = []

    def create(self, **request):
        self.calls.append(request)
        index = len(self.calls) - 1
        if index >= len(self.payloads):
            raise AssertionError("fake client received an unexpected extra call")
        return SimpleNamespace(
            status="completed",
            output_text=json.dumps(self.payloads[index]),
            usage={
                "total_input_tokens": 100,
                "total_output_tokens": 20,
                "total_thought_tokens": 5,
                "total_cached_tokens": 0,
            },
        )


class _Client:
    def __init__(self, payload):
        self.models = _Models()
        self.interactions = _Interactions(payload)


def test_loader_keeps_approved_anchor_and_annotated_bytes_distinct(tmp_path):
    spec = _write_spec(tmp_path, annotated=True)
    reference = spec.reference_images[0]
    assert reference.presentation == "annotated"
    assert reference.anchor_crop_sha256 != reference.content_sha256
    assert spec.resolve_reference_path(reference).is_absolute()
    assert "source_path" not in spec.canonical_definition_json()
    assert len(spec.definition_sha256()) == 64


def test_loader_rejects_changed_reference_bytes(tmp_path):
    spec = _write_spec(tmp_path)
    path = spec.resolve_reference_path(spec.reference_images[0])
    path.write_bytes(b"changed after approval")
    with pytest.raises(ReferenceGroundingError, match="content hash mismatch"):
        load_grounding_spec(tmp_path / "grounding.json")


def test_raw_reference_cannot_claim_different_bytes(tmp_path):
    raw = tmp_path / "reference.jpg"
    raw.write_bytes(b"raw")
    with pytest.raises(ValueError, match="raw reference bytes"):
        ReferenceImageSpec(
            target_id="target.primary",
            polarity="positive",
            frame_id="reference.frame.001",
            path=raw.name,
            anchor_crop_sha256="a" * 64,
            content_sha256="b" * 64,
            mime_type="image/jpeg",
            presentation="raw",
        )


def test_candidate_validation_is_offline_and_rejects_lineage_mismatch(tmp_path):
    spec = _write_spec(tmp_path)
    video = _video_lineage("c" * 64)
    valid = validate_candidate_payload(
        _candidate_payload(spec, video),
        spec=spec,
        video=video,
        target_ids=("target.primary",),
    )
    assert valid.candidates[0].recommended_seed_ms == 4_000

    with pytest.raises(ReferenceGroundingError, match="query lock hash mismatch"):
        validate_candidate_payload(
            _candidate_payload(spec, video, lock_hash="d" * 64),
            spec=spec,
            video=video,
            target_ids=("target.primary",),
        )
    with pytest.raises(ReferenceGroundingError, match="exceeds video duration"):
        validate_candidate_payload(
            _candidate_payload(spec, video, end_ms=12_000),
            spec=spec,
            video=video,
            target_ids=("target.primary",),
        )


def test_missing_client_never_probes_uploads_or_reserves_budget(tmp_path):
    spec = _write_spec(tmp_path)
    cache = _Cache()
    ledger = Ledger(cap_usd=1.0)
    assert discover_reference_candidates(
        tmp_path / "does-not-exist.mp4",
        spec,
        client=None,
        cache=cache,
        ledger=ledger,
    ) is None
    assert cache.paths == []
    assert not ledger.entries
    assert not ledger.reservations


def test_candidate_call_uses_mixed_media_adapter_cache_and_budget(
    tmp_path, monkeypatch
):
    spec = _write_spec(tmp_path)
    video_path = tmp_path / "candidate.mov"
    video_path.write_bytes(b"offline fake video")
    video = _video_lineage(content_hash(video_path))
    monkeypatch.setattr(
        "montagewright.reference_grounding.inspect_video_lineage",
        lambda path: video,
    )
    client = _Client(_candidate_payload(spec, video))
    cache = _Cache()
    ledger = Ledger(cap_usd=1.0)

    result, usage = discover_reference_candidates(
        video_path,
        spec,
        client=client,
        cache=cache,
        ledger=ledger,
    )

    assert result.video_asset_id == video.asset_id
    assert usage.thought_tokens == 5
    request = client.interactions.calls[0]
    assert [part["type"] for part in request["input"]].count("image") == 1
    assert [part["type"] for part in request["input"]].count("video") == 1
    assert next(
        part["mime_type"] for part in request["input"] if part["type"] == "video"
    ) == "video/quicktime"
    assert request["input"][-1]["type"] == "text"
    assert [path.name for path, _ in cache.paths] == [
        "annotated-reference.jpg",
        "candidate.mov",
    ]
    assert ledger.entries[0]["stage"] == "reference_candidate_discovery"


def test_exact_frame_lineage_is_derived_from_pts():
    with pytest.raises(ValueError, match="does not match source PTS"):
        ExactFrameLineage(
            video_asset_id="sha256:" + "e" * 64,
            video_sha256="e" * 64,
            source_start_pts=100,
            source_time_base=Rational(numerator=1, denominator=1000),
            requested_time_ms=1_000,
            frame_time_ms=999,
            frame_pts=1_100,
            frame_sha256="f" * 64,
            width=1920,
            height=1080,
        )


def test_exact_bbox_decision_preserves_lineage_and_converts_native_order(tmp_path):
    spec = _write_spec(tmp_path)
    video = _video_lineage("e" * 64)
    discovery = CandidateDiscoveryResult.model_validate(
        _candidate_payload(spec, video)
    )
    frame_path = tmp_path / "exact.jpg"
    frame_path.write_bytes(b"exact decoded frame")
    frame = ExactFrameMaterial(
        path=frame_path,
        lineage=ExactFrameLineage(
            video_asset_id=video.asset_id,
            video_sha256=video.content_sha256,
            source_start_pts=100,
            source_time_base=Rational(numerator=1, denominator=1000),
            requested_time_ms=4_000,
            frame_time_ms=4_000,
            frame_pts=4_100,
            frame_sha256=content_hash(frame_path),
            width=1920,
            height=1080,
        ),
    )
    payload = {
        "contract_version": "reference-exact-frame-bbox-v1",
        "query_id": spec.identity_lock.query_id,
        "query_lock_sha256": spec.identity_lock.definition_sha256(),
        "grounding_spec_sha256": spec.definition_sha256(),
        "video_asset_id": video.asset_id,
        "video_sha256": video.content_sha256,
        "target_id": "target.primary",
        "candidate_id": "candidate.001",
        "frame_pts": 4_100,
        "frame_time_ms": 4_000,
        "frame_sha256": content_hash(frame_path),
        "width": 1920,
        "height": 1080,
        "verdict": "matched_target",
        "confidence": 0.95,
        "native_box_yxyx_1000": [100, 200, 700, 800],
        "visibility_state": "full",
        "occlusion_state": "none",
        "touches_frame_edges": [],
        "identity_evidence": ["same hinge and corner mark"],
        "exclusion_evidence": [],
        "reason": "The approved stable cues are both directly visible.",
    }
    client = _Client(payload)
    cache = _Cache()

    decision, _ = decide_exact_frame_bbox(
        spec,
        discovery,
        "candidate.001",
        frame,
        client=client,
        cache=cache,
        ledger=Ledger(cap_usd=1.0),
    )

    assert decision.tracking_box_xyxy_1000 == (200, 100, 800, 700)
    assert [path.name for path, _ in cache.paths] == [
        "annotated-reference.jpg",
        "exact.jpg",
    ]


def test_exact_bbox_rejects_changed_frame_before_any_upload(tmp_path):
    spec = _write_spec(tmp_path)
    video = _video_lineage("e" * 64)
    discovery = CandidateDiscoveryResult.model_validate(
        _candidate_payload(spec, video)
    )
    frame_path = tmp_path / "exact.jpg"
    frame_path.write_bytes(b"original")
    lineage = ExactFrameLineage(
        video_asset_id=video.asset_id,
        video_sha256=video.content_sha256,
        source_start_pts=100,
        source_time_base=Rational(numerator=1, denominator=1000),
        requested_time_ms=4_000,
        frame_time_ms=4_000,
        frame_pts=4_100,
        frame_sha256=content_hash(frame_path),
        width=1920,
        height=1080,
    )
    frame_path.write_bytes(b"changed")
    cache = _Cache()
    with pytest.raises(ReferenceGroundingError, match="content hash"):
        decide_exact_frame_bbox(
            spec,
            discovery,
            "candidate.001",
            ExactFrameMaterial(path=frame_path, lineage=lineage),
            client=_Client({}),
            cache=cache,
        )
    assert cache.paths == []


def test_offline_exact_validator_rejects_a_fabricated_candidate(tmp_path):
    spec = _write_spec(tmp_path)
    video = _video_lineage("c" * 64)
    discovery = CandidateDiscoveryResult.model_validate(
        _candidate_payload(spec, video)
    )
    frame = _exact_frame(tmp_path, video, 4_000)
    fabricated = discovery.candidates[0].model_copy(
        update={"start_ms": 3_500}
    )

    with pytest.raises(
        ReferenceGroundingError,
        match="differs from candidate discovery",
    ):
        validate_exact_frame_payload(
            _exact_decision_payload(spec, video, frame),
            spec=spec,
            discovery=discovery,
            candidate=fabricated,
            frame=frame.lineage,
        )


def test_changed_video_upload_cannot_poison_the_content_cache(tmp_path):
    path = tmp_path / "mutable.mp4"
    path.write_bytes(b"before")
    expected = content_hash(path)

    class MutatingCache:
        def __init__(self):
            self.entries = {}
            self.saved = 0

        def uri_for(self, source, _client, *, mime_type):
            self.entries[expected] = {
                "uri": "files://wrong-bytes",
                "mime_type": mime_type,
            }
            source.write_bytes(b"after")
            return "files://wrong-bytes", False

        def save(self):
            self.saved += 1

    cache = MutatingCache()
    with pytest.raises(ReferenceGroundingError, match="changed during upload"):
        grounding._media_uri(
            path,
            client=object(),
            cache=cache,
            mime_type="video/mp4",
            expected_sha256=expected,
        )

    assert expected not in cache.entries
    assert cache.saved == 1


def test_semantic_time_materializes_a_real_fractional_rate_pts(tmp_path):
    video = tmp_path / "fractional-rate.mp4"
    subprocess.run(
        [
            "ffmpeg", "-hide_banner", "-loglevel", "error",
            "-f", "lavfi", "-i",
            "testsrc2=s=320x180:r=30000/1001:d=1",
            "-c:v", "libx264", "-pix_fmt", "yuv420p", str(video),
        ],
        check=True,
    )

    semantic = materialize_frame_at_time(
        video, 600, tmp_path / "semantic.jpg"
    )
    exact = materialize_frame_at_pts(
        video, semantic.lineage.frame_pts, tmp_path / "exact.jpg"
    )

    assert semantic.lineage.frame_pts == 18_018
    assert semantic.lineage.frame_pts != 18_000
    assert exact.lineage.frame_pts == semantic.lineage.frame_pts
    assert exact.lineage.frame_sha256 == semantic.lineage.frame_sha256


def test_simple_builder_creates_content_addressed_approved_lock(tmp_path):
    from montagewright.reference_grounding import build_reference_grounding_spec

    positive = tmp_path / "fold.jpg"
    negative = tmp_path / "tablet.png"
    positive.write_bytes(b"fold identity bytes")
    negative.write_bytes(b"hard negative bytes")

    spec = build_reference_grounding_spec(
        tmp_path / "grounding.json",
        target_id="device.fold",
        target_description="the exact foldable device selected by the user",
        identity_cues=("distinct hinge", "distinct hinge", "camera layout"),
        stable_exclusions=("ordinary tablet",),
        positive_images=(positive,),
        negative_images=(negative,),
    )

    target = spec.identity_lock.identity.target("device.fold")
    assert target.identity_cues == ("distinct hinge", "camera layout")
    assert target.stable_exclusions == ("ordinary tablet",)
    assert len(target.positive_anchors) == len(target.negative_anchors) == 1
    assert all(
        spec.resolve_reference_path(reference).is_file()
        for reference in spec.reference_images
    )
    assert spec.identity_lock.contract_version == "grounding-query-lock-v1"


def test_reference_prompt_parts_without_client_is_text_only_and_never_reads_media(
    tmp_path,
):
    spec = _write_spec(tmp_path, include_negative=True)
    positive_path = spec.resolve_reference_path(spec.reference_images[0])
    positive_path.write_bytes(b"changed after the spec was loaded")
    cache = _Cache()

    parts = reference_prompt_parts(spec, client=None, cache=cache)

    assert [part["type"] for part in parts] == ["text"]
    assert "REFERENCE_MEDIA_ATTACHED=false" in parts[0]["text"]
    assert '"polarity":"positive"' in parts[0]["text"]
    assert '"polarity":"negative"' in parts[0]["text"]
    assert cache.paths == []


def test_exact_batch_without_client_does_not_iterate_or_spend(tmp_path):
    spec = _write_spec(tmp_path)
    video = _video_lineage("e" * 64)
    discovery = CandidateDiscoveryResult.model_validate(
        _candidate_payload(spec, video)
    )
    cache = _Cache()
    ledger = Ledger(cap_usd=1.0)

    class PoisonFrames:
        def __iter__(self):
            raise AssertionError("client=None must not iterate frames")

    assert decide_exact_frame_bboxes(
        spec,
        discovery,
        "target.primary",
        PoisonFrames(),
        client=None,
        cache=cache,
        ledger=ledger,
    ) is None
    assert cache.paths == []
    assert not ledger.entries
    assert not ledger.reservations


def test_exact_batch_preflights_every_frame_before_upload_or_budget(tmp_path):
    spec = _write_spec(tmp_path)
    video = _video_lineage("e" * 64)
    discovery = CandidateDiscoveryResult.model_validate(
        _candidate_payload(spec, video)
    )
    frames = [_exact_frame(tmp_path, video, 2_000), _exact_frame(tmp_path, video, 3_000)]
    frames[1].path.write_bytes(b"changed after lineage was captured")
    cache = _Cache()
    ledger = Ledger(cap_usd=1.0)
    client = _Client({})

    with pytest.raises(ReferenceGroundingError, match="content hash"):
        decide_exact_frame_bboxes(
            spec,
            discovery,
            "target.primary",
            frames,
            client=client,
            cache=cache,
            ledger=ledger,
        )

    assert cache.paths == []
    assert client.interactions.calls == []
    assert not ledger.entries
    assert not ledger.reservations


def test_exact_batch_chunks_calls_reuses_both_reference_polarities_and_orders_output(
    tmp_path,
):
    spec = _write_spec(tmp_path, include_negative=True)
    video = _video_lineage("e" * 64)
    discovery = CandidateDiscoveryResult.model_validate(
        _candidate_payload(spec, video)
    )
    frames = [
        _exact_frame(tmp_path, video, time_ms)
        for time_ms in (2_000, 3_000, 4_000, 5_000, 6_000)
    ]
    responses = []
    for start in range(0, len(frames), 2):
        chunk = frames[start : start + 2]
        responses.append(
            {
                "contract_version": (
                    "reference-exact-frame-bbox-batch-response-v1"
                ),
                # Provider order is deliberately unstable; local output must not be.
                "decisions": [
                    _exact_decision_payload(spec, video, frame)
                    for frame in reversed(chunk)
                ],
            }
        )
    client = _Client(responses)
    cache = _Cache()
    ledger = Ledger(cap_usd=1.0)

    result, usage = decide_exact_frame_bboxes(
        spec,
        discovery,
        "target.primary",
        frames,
        client=client,
        cache=cache,
        ledger=ledger,
        max_frames_per_call=2,
    )

    assert [item.lineage.frame_pts for item in result.evaluations] == [
        frame.lineage.frame_pts for frame in frames
    ]
    assert result.sam_ready
    assert len(result.sam_seed_evaluations()) == 5
    assert usage.input_tokens == 300
    assert usage.output_tokens == 60
    assert usage.thought_tokens == 15
    assert len(client.interactions.calls) == 3
    assert [path.name for path, _ in cache.paths] == [
        "annotated-reference.jpg",
        "hard-negative-reference.jpg",
        *[frame.path.name for frame in frames],
    ]
    for call, expected_frame_count in zip(
        client.interactions.calls, (2, 2, 1), strict=True
    ):
        types = [part["type"] for part in call["input"]]
        assert types.count("image") == 2 + expected_frame_count
        reference_labels = [
            part["text"]
            for part in call["input"]
            if part["type"] == "text" and part["text"].startswith("REFERENCE ")
        ]
        assert any("polarity=positive" in label for label in reference_labels)
        assert any("polarity=negative" in label for label in reference_labels)
    assert [entry["stage"] for entry in ledger.entries] == [
        "reference_exact_frame_bbox_batch",
        "reference_exact_frame_bbox_batch",
        "reference_exact_frame_bbox_batch",
    ]


def test_exact_batch_requires_two_distinct_matched_decisions_for_sam(tmp_path):
    spec = _write_spec(tmp_path)
    video = _video_lineage("e" * 64)
    discovery = CandidateDiscoveryResult.model_validate(
        _candidate_payload(spec, video)
    )
    frames = [_exact_frame(tmp_path, video, 2_000), _exact_frame(tmp_path, video, 3_000)]
    response = {
        "contract_version": "reference-exact-frame-bbox-batch-response-v1",
        "decisions": [
            _exact_decision_payload(spec, video, frames[0]),
            _exact_decision_payload(
                spec,
                video,
                frames[1],
                verdict="uncertain",
            ),
        ],
    }

    result, _ = decide_exact_frame_bboxes(
        spec,
        discovery,
        "target.primary",
        frames,
        client=_Client(response),
        cache=_Cache(),
    )

    assert result.matched_anchor_count == 1
    assert not result.sam_ready
    with pytest.raises(ReferenceGroundingError, match="at least 2 matched"):
        result.sam_seed_evaluations()
