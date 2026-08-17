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


def test_identity_box_ratio_is_a_non_blocking_spec_disagreement() -> None:
    spec = SimpleNamespace(identity_lock=SimpleNamespace(
        identity=SimpleNamespace(targets=(SimpleNamespace(
            target_id="target.primary",
            identity_cues=(
                "target short over long is 0.77",
                "front view short side divided by long side is 0.77",
                "lookalike is near 0.93",
            ),
            stable_exclusions=("lookalike short over long is 0.93",),
        ),)),
    ))
    correct = SimpleNamespace(box=(0.0, 0.0, 0.78, 1.0))
    wrong = SimpleNamespace(box=(0.0, 0.0, 0.96, 1.0))

    assert grounding.declared_identity_box_ratios(
        spec, "target.primary"
    ) == (0.77,)
    assert grounding.identity_box_ratio_disagreement(
        spec, "target.primary", [correct]
    ) is None
    warning = grounding.identity_box_ratio_disagreement(
        spec, "target.primary", [wrong]
    )
    assert warning is not None
    assert "advisory" in warning


def test_identity_box_ratio_reads_one_ratio_per_filmed_state() -> None:
    """A real target states each state's ratio once, in its own cue.

    The shipped Fold8 spec names 0.76 unfolded and 0.63 folded exactly once
    each.  An earlier reader kept a ratio only when two cues repeated it, so
    it collected the folded number alone and reported every correctly
    identified unfolded frame as a disagreement -- scoring the true target
    identically to the lookalike this check exists to catch.
    """

    spec = SimpleNamespace(identity_lock=SimpleNamespace(
        identity=SimpleNamespace(targets=(SimpleNamespace(
            target_id="target.primary",
            identity_cues=(
                "unfolded, front on: short side about 0.76 of the long side",
                "folded, front on: short side about 0.63 of the long side",
            ),
            stable_exclusions=("the lookalike is nearer 0.93",),
        ),)),
    ))

    assert grounding.declared_identity_box_ratios(
        spec, "target.primary"
    ) == (0.63, 0.76)

    def ratios(*values: float) -> list[SimpleNamespace]:
        return [SimpleNamespace(box=(0.0, 0.0, one, 1.0)) for one in values]

    # Both filmed states agree with the target they actually match.
    assert grounding.identity_box_ratio_disagreement(
        spec, "target.primary", ratios(0.78, 0.75, 0.78)
    ) is None
    assert grounding.identity_box_ratio_disagreement(
        spec, "target.primary", ratios(0.63, 0.63)
    ) is None
    # The excluded lookalike still is not one of them.
    assert grounding.identity_box_ratio_disagreement(
        spec, "target.primary", ratios(0.96, 0.92, 0.90)
    ) is not None


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


def test_identity_draft_reads_the_pictures_and_stays_a_draft(tmp_path):
    """Nobody types "5.5-inch 10:16 cover display" into an empty box.

    That cue is what made the Fold8 lock work, and it is a specification
    somebody looked up. Asking every user to be that person means the ones
    who are not leave the field blank and grounding silently gets worse. The
    draft is proposed from the pictures and remains editable text: it is not
    a lock, and nothing downstream may read it.
    """

    from montagewright.reference_grounding import draft_identity_from_references

    one = tmp_path / "front.jpg"
    two = tmp_path / "back.png"
    one.write_bytes(b"front bytes")
    two.write_bytes(b"back bytes")
    client = _Client({
        "contract_version": "reference-identity-draft-v1",
        "target_description": "參考圖中這一隻黑白花貓，站姿與趴姿都算同一隻。",
        "identity_cues": ["左耳尖有一撮白毛", "右前腳白襪只到腕部"],
        "stable_exclusions": ["另一隻全黑的貓：沒有白襪"],
        "caveat": "",
    })
    cache = _Cache()

    drafted = draft_identity_from_references(
        [one, two], client=client, cache=cache, ledger=Ledger(cap_usd=1.0)
    )

    assert drafted is not None
    draft, usage = drafted
    assert draft.identity_cues[0] == "左耳尖有一撮白毛"
    assert usage.input_tokens == 100
    assert [path.name for path, _ in cache.paths] == ["front.jpg", "back.png"], (
        "both pictures are read, in the order they were given"
    )
    sent = client.interactions.calls[0]["input"]
    assert sum(1 for part in sent if part["type"] == "image") == 2
    assert not any(part["type"] == "video" for part in sent), (
        "drafting looks at references only; it never pays to watch the rushes"
    )


def test_identity_draft_without_a_client_spends_nothing(tmp_path):
    """Assembling a spec offline must not acquire a client or upload bytes."""

    from montagewright.reference_grounding import draft_identity_from_references

    picture = tmp_path / "front.jpg"
    picture.write_bytes(b"front bytes")
    assert draft_identity_from_references([picture], client=None) is None


def test_identity_screening_is_remembered_where_the_cards_are(tmp_path):
    """Whether a source contains the locked identity is a material fact.

    The same shape as a clip card: true of those pixels and that lock, not
    of this cut. Keeping the answer in the run directory made a second cut
    of the same rushes pay for all of it again -- and on the afternoon this
    was written, a run that stopped early paid for it twice before dinner.
    """

    from montagewright.planner import Usage
    from montagewright.reference_grounding import (
        CandidateDiscoveryResult, remembered_discovery, sha256_file,
    )

    spec = _write_spec(tmp_path)
    video = tmp_path / "take.mp4"
    video.write_bytes(b"video bytes")
    library = tmp_path / "library"
    digest = sha256_file(video)
    answer = CandidateDiscoveryResult.model_validate({
        "contract_version": "reference-candidate-discovery-v1",
        "query_id": spec.identity_lock.query_id,
        "query_lock_sha256": spec.identity_lock.definition_sha256(),
        "grounding_spec_sha256": spec.definition_sha256(),
        "video_asset_id": f"sha256:{digest}",
        "video_sha256": digest,
        "duration_ms": 15015,
        "candidates": [],
        "target_summaries": [{
            "target_id": "device.fold",
            "verdict": "absent",
            "reason": "a three-camera device, not this one",
        }],
        "warnings": [],
    })
    asked = {"times": 0}

    def instead(*_args, **_kwargs):
        asked["times"] += 1
        return answer, Usage(input_tokens=10, output_tokens=1, thought_tokens=0)

    # A different lock over the same bytes is a different question, so it is
    # asked again and remembered under its own name.
    second = tmp_path / "second"
    second.mkdir()
    other = _write_spec(second, include_negative=True)
    original = grounding.discover_reference_candidates
    grounding.discover_reference_candidates = instead
    try:
        first = remembered_discovery(
            video, spec, client=object(), library=library,
        )
        again = remembered_discovery(
            video, spec, client=object(), library=library,
        )
        after_the_same_question_twice = asked["times"]
        remembered_discovery(video, other, client=object(), library=library)
    finally:
        grounding.discover_reference_candidates = original

    assert first is not None and again is not None
    assert after_the_same_question_twice == 1, "the second run asks nobody"
    assert again[1] is None, "a remembered answer has no usage to charge"
    assert again[0].target_summaries[0].verdict == "absent"
    assert len(list((library / "reference-grounding").glob("*.json"))) == 2, (
        "one file per (bytes, lock) pair"
    )

    assert asked["times"] == 2
    assert len(list((library / "reference-grounding").glob("*.json"))) == 2


def test_a_lookalike_reports_where_it_is_without_becoming_a_seed(tmp_path):
    """Two devices in one frame is a composition problem, not a dead shot.

    A 9:16 crop out of 16:9 keeps about a third of the width, so the other
    model can often simply be left outside the frame -- but nothing could
    even ask while its position was never reported. The rule that only a
    matched target carries `native_box_yxyx_1000` is what stops a tracker
    seeding on the wrong instance, so avoidance gets its own field instead
    of relaxing that.
    """

    from montagewright.reference_grounding import (
        ExactFrameBBoxDecision, ExcludedInstance,
    )

    spec = _write_spec(tmp_path)
    decision = ExactFrameBBoxDecision(
        query_id=spec.identity_lock.query_id,
        query_lock_sha256=spec.identity_lock.definition_sha256(),
        grounding_spec_sha256=spec.definition_sha256(),
        video_asset_id=f"sha256:{'a' * 64}",
        video_sha256="a" * 64,
        target_id="device.fold",
        candidate_id="cand_001",
        frame_pts=18018,
        frame_time_ms=600,
        frame_sha256="b" * 64,
        width=1440,
        height=810,
        verdict="matched_target",
        confidence=0.95,
        native_box_yxyx_1000=(100, 200, 700, 800),
        visibility_state="full",
        occlusion_state="none",
        identity_evidence=("same hinge and corner mark",),
        excluded_instances=(
            ExcludedInstance(
                native_box_yxyx_1000=(120, 780, 640, 980),
                reason="the three-camera model, right of frame",
            ),
        ),
        reason="judged independently on this exact frame",
    )

    assert decision.tracking_box_xyxy_1000 == (200, 100, 800, 700), (
        "the seed still comes from the target's own box"
    )
    assert len(decision.excluded_instances) == 1
    assert decision.excluded_instances[0].reason.startswith("the three-camera")

    with pytest.raises(Exception):
        ExactFrameBBoxDecision.model_validate(dict(
            decision.model_dump(mode="json"),
            excluded_instances=[{
                "native_box_yxyx_1000": [900, 100, 100, 200],
                "reason": "bottom edge above top edge",
            }],
        ))


def test_a_subject_may_leave_exactly_when_its_interval_ends(tmp_path):
    """Leaving is a boundary, not a sample.

    `recommended_seed_ms` is half-open because it has to name a frame
    somebody can decode. `frame_exit_ms` answers a different question: a
    subject still on screen when the interval ends exits at exactly
    `end_ms`. Borrowing the seed's rule rejected that answer and took a
    whole seventy-four-source screen down on the fourth one.
    """

    from pydantic import ValidationError

    from montagewright.reference_grounding import CandidateInterval

    def interval(**changes):
        return CandidateInterval.model_validate({
            "candidate_id": "cand_001",
            "target_id": "device.fold",
            "start_ms": 0,
            "end_ms": 15015,
            "recommended_seed_ms": 7000,
            "identity_status": "matched_target",
            "confidence": 0.9,
            "visible_state": "unfolded, held",
            "visibility_state": "full",
            "occlusion_state": "none",
            "identity_evidence": ["the hinge and the corner mark"],
            **changes,
        })

    assert interval(frame_exit_ms=15015).frame_exit_ms == 15015
    assert interval(frame_entry_ms=0).frame_entry_ms == 0

    with pytest.raises(ValidationError):
        interval(frame_exit_ms=15016)
    with pytest.raises(ValidationError):
        interval(frame_exit_ms=0)
    with pytest.raises(ValidationError):
        interval(frame_entry_ms=15015)
    with pytest.raises(ValidationError):
        interval(recommended_seed_ms=15015)


def test_identity_is_sampled_where_the_screen_says_it_is_clearest(tmp_path):
    """Per sighting, not per source and not per cut.

    The screen already says where the target is and which moment shows it
    most clearly, and every one of those answers was thrown away: the
    frames were taken from the seconds an edit happened to want, so a
    sixteen-second take of a handset seen edge-on beside a coin was judged
    on three views of an edge. Each sighting needs its own moments, because
    a tracker cannot carry a box across a gap where the subject left.
    """

    from montagewright.reference_grounding import (
        CandidateDiscoveryResult, sampling_times_for,
    )

    def screened(candidates):
        return CandidateDiscoveryResult.model_validate({
            "contract_version": "reference-candidate-discovery-v1",
            "query_id": "grounding:device.fold",
            "query_lock_sha256": "a" * 64,
            "grounding_spec_sha256": "b" * 64,
            "video_asset_id": "sha256:" + "c" * 64,
            "video_sha256": "c" * 64,
            "duration_ms": 31_000,
            "candidates": candidates,
            "target_summaries": [{
                "target_id": "device.fold", "verdict": "present",
                "reason": "seen",
            }],
            "warnings": [],
        })

    def sighting(start, end, seed, status="matched_target"):
        return {
            "candidate_id": f"c{start}", "target_id": "device.fold",
            "start_ms": start, "end_ms": end, "recommended_seed_ms": seed,
            "identity_status": status, "confidence": 0.9,
            "visible_state": "folded", "visibility_state": "full",
            "occlusion_state": "none",
            "identity_evidence": ["the hinge"] if status == "matched_target" else [],
            "exclusion_evidence": [] if status == "matched_target" else ["another model"],
        }

    short = [at for at, _ in sampling_times_for(
        screened([sighting(0, 11_000, 5_000)]), "device.fold"
    )]
    assert 5_000 in short, "the moment the screen picked is always looked at"
    assert len(short) == 3, "首中尾 for an ordinary take"

    long_take = sampling_times_for(
        screened([sighting(0, 31_000, 15_000)]), "device.fold"
    )
    assert len(long_take) == 5, "a long take gets more, not the same three"

    # Five sightings overflow the per-call limit. Round robin, so the later
    # ones still get a moment -- taking the earliest eight would leave them
    # with none, and a seed cannot cross the gap to reach them.
    many = sampling_times_for(
        screened([
            sighting(n * 6_000, n * 6_000 + 5_000, n * 6_000 + 2_000)
            for n in range(5)
        ]),
        "device.fold",
    )
    assert len({name for _, name in many}) == 5, (
        "every appearance gets at least one moment"
    )

    twice = [at for at, _ in sampling_times_for(
        screened([sighting(0, 8_000, 4_000), sighting(14_000, 22_000, 18_000)]),
        "device.fold",
    )]
    assert any(t < 8_000 for t in twice) and any(t > 14_000 for t in twice), (
        "each appearance is proved on its own; a seed from one is no use in "
        "the other"
    )

    # A summary of "present" needs a matched candidate, so the refusal case
    # is written as the screen would really write it.
    lookalike = screened([sighting(0, 9_000, 4_000)])
    refused = sampling_times_for(
        lookalike.model_copy(update={"candidates": tuple(
            one.model_copy(update={
                "identity_status": "hard_negative",
                "identity_evidence": (),
                "exclusion_evidence": ("another model",),
            })
            for one in lookalike.candidates
        )}),
        "device.fold",
    )
    assert refused == [], "nothing is sampled where the screen saw a lookalike"


def test_a_confirmation_only_speaks_for_the_appearance_it_belongs_to():
    """A box proved before the subject left says nothing after it returns.

    The tracker cannot cross that gap either, so reaching across one would
    seed a cut from whatever the mask drifted onto. The rule was written in
    a comment and enforced by nothing: the filter was pure arithmetic on
    seconds, and the sighting a confirmation came from had already been
    flattened away before the filter could see it.
    """

    from montagewright.pipeline import CONFIRMED_REACH_SECONDS, _reaches
    from montagewright.reference_grounding import ConfirmedFrame

    def proved(at, window):
        return ConfirmedFrame(
            at_seconds=at, box=(0.2, 0.2, 0.4, 0.6), sighting="c1",
            sighting_window=window, frame_pts=int(at * 1000),
            frame_sha256="a" * 64,
        )

    # Proved at 2.5s during an appearance that runs 0-3s; the cut is at
    # 4.5-6.5s, after the subject left. Close in time, wrong appearance.
    assert not _reaches(proved(2.5, (0.0, 3.0)), 4.5, 6.5)

    # Same instant, but the appearance covers the cut: this is the case the
    # reach exists for -- a close-up that opens a take.
    assert _reaches(proved(2.5, (0.0, 12.0)), 4.5, 6.5)

    # Still bounded in time even inside one long appearance.
    assert not _reaches(
        proved(2.0, (0.0, 60.0)), 2.0 + CONFIRMED_REACH_SECONDS + 1.0, 30.0
    )


def test_a_source_without_the_identity_is_context_not_rubbish():
    """The venue is not the product, and a launch film needs both.

    The screen used to delete a source the locked identity was absent from,
    which threw away the entrance, the main visual and the people at the
    stand -- every establishing shot the brief asked for in the same breath
    as it said those shots need not contain the product.
    """

    from montagewright.planner import MaterialItem, context_only_disagreements

    venue = MaterialItem(
        source_id="C8370", duration_seconds=5.0,
        summary="Galaxy Unpacked main visual", carries_identity=False,
    )
    product = MaterialItem(
        source_id="C8340", duration_seconds=5.0, summary="rear camera module",
    )
    targets = {"device.galaxy_z_fold8"}

    # Context, used as context: nothing to say.
    assert context_only_disagreements(
        [{"source_id": "C8370", "looks": [{"entity_id": "none"}]}],
        [venue, product], targets,
    ) == []

    # Context, claiming the product: refused before geometry goes looking
    # for a subject that was never in the frame.
    refused = context_only_disagreements(
        [{"source_id": "C8370",
          "looks": [{"entity_id": "device.galaxy_z_fold8"}]}],
        [venue, product], targets,
    )
    assert len(refused) == 1
    assert "C8370" in refused[0]

    # A source that carries it may of course name it.
    assert context_only_disagreements(
        [{"source_id": "C8340",
          "looks": [{"entity_id": "device.galaxy_z_fold8"}]}],
        [venue, product], targets,
    ) == []


def test_selection_promotes_identity_capable_alternate_over_absent_primary():
    """An absent primary is not allowed to survive as repair advice."""

    from montagewright.candidate_commitments import (
        CandidateCommitments, CandidateOption,
    )
    from montagewright.planner import (
        MaterialItem, _commitments_without_context_claims,
        _context_claiming_source_ids,
    )
    from montagewright.spans import Span

    absent = MaterialItem(
        source_id="C8388", duration_seconds=3.0, summary="weather screen",
        carries_identity=False,
        spans=(Span("C8388:s00", "C8388", 0.0, 3.0),),
    )
    present = MaterialItem(
        source_id="C8387", duration_seconds=3.0, summary="folded handset",
        spans=(Span("C8387:s00", "C8387", 0.0, 3.0),),
    )
    common = {
        "commitment_id": "comm_17", "purpose": "show the cover screen",
        "required": True, "picture_role": "illustrative_broll",
        "min_supported_seconds": 1.0,
        "presentation_intent": "centered_hold",
        "motion_preference": "hold",
        "target_id": "device.galaxy_z_fold8", "why": "visible handset",
    }
    commitments = CandidateCommitments(
        contract_version="candidate-commitment-v1",
        material_digest="a" * 64, direction_sha256="b" * 64,
        target_aspect="9:16", target_seconds=3.0,
        options=(
            CandidateOption(**common, span_id="C8388:s00", tier="primary"),
            CandidateOption(**common, span_id="C8387:s00", tier="alternate"),
        ),
    )

    eligible = _commitments_without_context_claims(
        commitments, [absent, present], {"device.galaxy_z_fold8"}
    )

    assert [(one.span_id, one.tier) for one in eligible.options] == [
        ("C8387:s00", "primary")
    ]
    assert [(one.span_id, one.tier) for one in commitments.options] == [
        ("C8388:s00", "primary"), ("C8387:s00", "alternate")
    ]
    assert _context_claiming_source_ids(
        [{
            "span_id": "C8388:s00",
            "looks": [{"entity_id": "device.galaxy_z_fold8"}],
        }],
        [absent, present], {"device.galaxy_z_fold8"},
    ) == {"C8388"}


def test_selection_fails_locally_when_absent_required_commitment_is_exhausted():
    from montagewright.candidate_commitments import (
        CandidateCommitments, CandidateOption,
    )
    from montagewright.planner import (
        MaterialItem, PlannerError, _commitments_without_context_claims,
    )
    from montagewright.spans import Span

    absent = MaterialItem(
        source_id="C8388", duration_seconds=3.0, summary="wrong handset",
        carries_identity=False,
        spans=(Span("C8388:s00", "C8388", 0.0, 3.0),),
    )
    commitments = CandidateCommitments(
        contract_version="candidate-commitment-v1",
        material_digest="a" * 64, direction_sha256="b" * 64,
        target_aspect="9:16", target_seconds=3.0,
        options=(CandidateOption(
            commitment_id="comm_17", purpose="show the cover screen",
            required=True, picture_role="illustrative_broll",
            span_id="C8388:s00", tier="primary",
            min_supported_seconds=1.0,
            presentation_intent="centered_hold", motion_preference="hold",
            target_id="device.galaxy_z_fold8", why="claimed product",
        ),),
    )

    with pytest.raises(PlannerError, match="no identity-capable primary"):
        _commitments_without_context_claims(
            commitments, [absent], {"device.galaxy_z_fold8"}
        )


def test_selection_distinguishes_source_evidence_from_final_track_proof():
    from montagewright.cli import _annotate_selection_identity_evidence

    selection = {"shots": [{
        "source_id": "C1",
        "looks": [{"entity_id": "device.fold"}],
    }, {
        "source_id": "C2",
        "looks": [{"entity_id": "device.fold"}],
    }, {
        "source_id": "C3",
        "looks": [{"entity_id": "none"}],
    }]}

    _annotate_selection_identity_evidence(
        selection, {"C1": {"device.fold": (object(),)}}
    )

    assert [shot["identity_status"] for shot in selection["shots"]] == [
        "source_confirmed", "unverified", "not_applicable",
    ]
    assert selection["shots"][0]["identity_target_id"] == "device.fold"
    assert "最終片段" in selection["shots"][0]["identity_issue"]
    assert len(selection["plan_disagreements"]) == 1
    assert "C2 claims device.fold" in selection["plan_disagreements"][0]


def test_exact_frame_output_budget_scales_for_multi_frame_answers():
    from montagewright.reference_grounding import exact_frame_output_budget

    assert exact_frame_output_budget(1) == 4096
    assert exact_frame_output_budget(5) == 7424
    assert exact_frame_output_budget(8) == 11264
