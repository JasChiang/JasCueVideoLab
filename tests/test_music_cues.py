from __future__ import annotations

import math
import struct
import wave
from pathlib import Path

import pytest

from jascue_video_lab.media import sha256_file
from jascue_video_lab.music import (
    MusicAnalysisParameters,
    MusicCueCandidate,
    MusicEnergyPoint,
    MusicMapProposal,
    MusicSectionCandidate,
    analyze_music,
    review_music_map,
)
from jascue_video_lab.music_cues import (
    CuePlanProposal,
    MusicSectionInterpretation,
    SemanticCuePairing,
    SemanticMusicPairingProposal,
    VisualSyncMap,
    derive_visual_sync_map,
    plan_music_cues,
    review_cue_plan,
)
from jascue_video_lab.models import ModelProvenance
from jascue_video_lab.models import (
    DenseFrame,
    DenseFrameCatalog,
    TrimFrameEvidence,
    TrimHumanReview,
    TrimIntentDecision,
    TrimIntentProposal,
    TrimPhaseSelection,
)
from jascue_video_lab.storage import read_json, write_json


def _write_click_track(path: Path, *, bpm: float = 120.0, duration_seconds: int = 12) -> None:
    sample_rate = 48_000
    total = sample_rate * duration_seconds
    beat_period = round(sample_rate * 60 / bpm)
    samples = [0.0] * total
    for beat_index, start in enumerate(range(0, total, beat_period)):
        amplitude = 0.95 if beat_index % 4 == 0 else 0.65
        for offset in range(min(round(sample_rate * 0.035), total - start)):
            envelope = math.exp(-offset / (sample_rate * 0.008))
            samples[start + offset] += (
                amplitude
                * envelope
                * math.sin(2 * math.pi * (160 if beat_index % 4 == 0 else 240) * offset / sample_rate)
            )
    with wave.open(str(path), "wb") as output:
        output.setnchannels(1)
        output.setsampwidth(2)
        output.setframerate(sample_rate)
        output.writeframes(
            b"".join(
                struct.pack("<h", max(-32768, min(32767, round(value * 32767))))
                for value in samples
            )
        )


def _proposal() -> MusicMapProposal:
    parameters = MusicAnalysisParameters()
    duration_samples = 48_000 * 4
    return MusicMapProposal(
        music_id=f"sha256:{'a' * 64}",
        source_sha256="a" * 64,
        master_sample_rate=48_000,
        duration_samples=duration_samples,
        duration_ms=4_000,
        analysis_parameters=parameters,
        estimated_bpm=120.0,
        tempo_confidence=0.8,
        meter_suggestion=4,
        first_beat_sample=0,
        cues=[
            MusicCueCandidate(
                cue_id=f"mc-{index + 1:05d}",
                kind="beat_candidate",
                sample_index=sample,
                time_ms=round(sample * 1000 / 48_000),
                strength=0.8,
                confidence=0.8,
            )
            for index, sample in enumerate(range(0, duration_samples, 24_000))
        ],
        sections=[
            MusicSectionCandidate(
                section_id="section-001",
                start_sample=0,
                end_sample=duration_samples,
                label="section_001",
                boundary_source="whole_track",
                confidence=0.5,
            )
        ],
        energy_curve=[
            MusicEnergyPoint(
                sample_index=0,
                time_ms=0,
                energy=0.5,
                onset_strength=0.8,
            )
        ],
        uncertainties=["human review required"],
        generated_at="2026-07-23T00:00:00+00:00",
    )


def _render_manifest(path: Path) -> None:
    write_json(
        path,
        {
            "project_id": "generic-edit",
            "horizontal": {
                "status": "rendered",
                "chapters": [
                    {"feature_id": "chapter-a", "duration_ms": 1_000},
                    {"feature_id": "chapter-b", "duration_ms": 1_200},
                    {"feature_id": "chapter-c", "duration_ms": 800},
                ],
            },
            "vertical": {
                "status": "rendered",
                "chapters": [
                    {"feature_id": "chapter-a", "duration_ms": 1_000},
                    {"feature_id": "chapter-b", "duration_ms": 1_200},
                    {"feature_id": "chapter-c", "duration_ms": 800},
                ],
            },
        },
    )


def test_local_music_analysis_recovers_click_track_tempo(tmp_path: Path) -> None:
    music = tmp_path / "click.wav"
    _write_click_track(music)
    proposal = analyze_music(music)
    assert proposal.requires_human_review is True
    assert proposal.estimated_bpm is not None
    assert proposal.estimated_bpm == pytest.approx(120.0, abs=3.0)
    assert proposal.duration_ms == pytest.approx(12_000, abs=2)
    assert any(cue.kind == "beat_candidate" for cue in proposal.cues)
    assert proposal.sections[0].start_sample == 0
    assert proposal.sections[-1].end_sample == proposal.duration_samples


def test_music_map_requires_explicit_review_before_lock(tmp_path: Path) -> None:
    proposal = _proposal()
    proposal_path = tmp_path / "music-map.proposal.json"
    write_json(proposal_path, proposal)
    review, lock = review_music_map(
        proposal,
        proposal_path=proposal_path,
        reviewer="editor",
        decision="rejected",
        notes="wrong half-time interpretation",
    )
    assert review.decision == "rejected"
    assert lock is None

    approved_review, approved = review_music_map(
        proposal,
        proposal_path=proposal_path,
        reviewer="editor",
        decision="approved",
        bpm=120.0,
        first_downbeat_sample=0,
        meter=4,
    )
    assert approved_review.decision == "approved"
    assert approved is not None
    assert approved.review.reviewer == "editor"
    assert any(cue.kind == "downbeat" for cue in approved.cues)
    assert len({cue.sample_index for cue in approved.cues}) == len(approved.cues)


def test_visual_sync_map_zero_flex_is_read_only(tmp_path: Path) -> None:
    manifest = tmp_path / "render-manifest.json"
    _render_manifest(manifest)
    visual = derive_visual_sync_map(manifest, aspect_ratio="9:16")
    assert visual.flexibility_authorization == "read_only_boundaries"
    assert all(point.flex_before_ms == 0 for point in visual.points)
    assert [point.project_time_ms for point in visual.points] == [0, 1000, 2200, 3000]


def test_visual_sync_map_projects_human_approved_trim_phases(
    tmp_path: Path,
) -> None:
    source_asset_id = f"sha256:{'b' * 64}"
    frames = [
        DenseFrame(
            frame_id=f"DF{index:06d}",
            event_id="event-a",
            requested_time_ms=time_ms,
            frame_time_ms=time_ms,
            frame_pts=index,
            frame_hash=f"{index}" * 64,
            width=640,
            height=360,
            image_path=f"/tmp/df-{index}.png",
            transport_image_path=f"/tmp/df-{index}.jpg",
            transport_image_hash=f"{index + 3}" * 64,
        )
        for index, time_ms in [(1, 0), (2, 500), (3, 1000)]
    ]
    catalog = DenseFrameCatalog(
        source_asset_id=source_asset_id,
        event_id="event-a",
        sampling_fps=2,
        source_start_ms=0,
        source_end_ms=1500,
        frames=frames,
        contact_sheet_paths=["/tmp/sheet.jpg"],
        contact_sheet_hashes=["e" * 64],
        generated_at="2026-07-23T00:00:00+00:00",
    )
    catalog_path = tmp_path / "dense-catalog.json"
    write_json(catalog_path, catalog)
    model_provenance = ModelProvenance(
        model_id="gemini-test",
        api="gemini_interactions",
        sdk="google-genai",
        sdk_version="test",
        run_id="test",
        generated_at="2026-07-23T00:00:00+00:00",
    )
    trim_proposal = TrimIntentProposal(
        source_asset_id=source_asset_id,
        event_id="event-a",
        usable=True,
        selections=[
            TrimPhaseSelection(phase="action_start", frame_id="DF000001"),
            TrimPhaseSelection(phase="result_start", frame_id="DF000002"),
            TrimPhaseSelection(phase="recommended_in", frame_id="DF000001"),
            TrimPhaseSelection(phase="recommended_out", frame_id="DF000003"),
        ],
        tail_intent="none",
        observed_phase_evidence="A generic action begins and reaches a visible result.",
        hold_evidence="",
        trim_reason="Preserve setup and result.",
        quality_risks=[],
        uncertainties=[],
        requires_human_review=True,
        confidence=0.8,
        model_provenance=model_provenance,
    )
    proposal_path = tmp_path / "trim-proposal.json"
    write_json(proposal_path, trim_proposal)
    evidence = [
        TrimFrameEvidence(
            frame_id=frame.frame_id,
            requested_time_ms=frame.requested_time_ms,
            frame_time_ms=frame.frame_time_ms,
            frame_pts=frame.frame_pts,
            frame_hash=frame.frame_hash,
        )
        for frame in frames
    ]
    decision = TrimIntentDecision(
        source_asset_id=source_asset_id,
        event_id="event-a",
        shot_id="shot-001",
        usable=True,
        first_included_frame=evidence[0],
        last_included_frame=evidence[1],
        exclusive_out_frame=evidence[2],
        hold_start_frame=None,
        hold_end_frame=None,
        source_in_ms=0,
        source_out_ms=1000,
        source_in_pts=1,
        source_out_pts=3,
        handle_in_ms=0,
        handle_out_ms=1500,
        tail_intent="none",
        approval_status="approved",
        requires_human_review=False,
        human_review=TrimHumanReview(
            reviewer="editor",
            reviewed_at="2026-07-23T00:00:00+00:00",
            decision="approved",
        ),
        proposal_path=str(proposal_path),
        catalog_path=str(catalog_path),
    )
    decision_path = tmp_path / "trim-decision.reviewed.json"
    write_json(decision_path, decision)
    manifest = tmp_path / "render-manifest.json"
    write_json(
        manifest,
        {
            "project_id": "trim-phases",
            "vertical": {
                "status": "rendered",
                "chapters": [
                    {
                        "feature_id": "chapter-a",
                        "duration_ms": 1000,
                        "source_in_ms": 0,
                        "source_out_ms": 1000,
                        "trim_decision_path": str(decision_path),
                        "semantic_intent": "A generic observed action.",
                    }
                ],
            },
        },
    )
    visual = derive_visual_sync_map(
        manifest, aspect_ratio="9:16", default_flex_ms=50
    )
    phases = {point.phase: point for point in visual.points}
    assert phases["action_start"].project_time_ms == 0
    assert phases["result_start"].project_time_ms == 500
    assert phases["result_start"].evidence_refs[0].startswith("trim-decision:")


def test_global_cue_plan_only_uses_authorized_windows(tmp_path: Path) -> None:
    proposal = _proposal()
    proposal_path = tmp_path / "music-map.proposal.json"
    write_json(proposal_path, proposal)
    _, lock = review_music_map(
        proposal,
        proposal_path=proposal_path,
        reviewer="editor",
        decision="approved",
        bpm=120.0,
        first_downbeat_sample=0,
        meter=4,
    )
    assert lock is not None
    lock_path = tmp_path / "music-map.lock.json"
    write_json(lock_path, lock)

    manifest = tmp_path / "render-manifest.json"
    _render_manifest(manifest)
    visual = derive_visual_sync_map(
        manifest, aspect_ratio="9:16", default_flex_ms=300
    )
    visual_path = tmp_path / "visual-sync-map.json"
    write_json(visual_path, visual)
    plan = plan_music_cues(
        lock,
        visual,
        music_lock_path=lock_path,
        visual_sync_map_path=visual_path,
        preset="balanced",
    )
    assert plan.changes_applied is False
    assert plan.requires_human_review is True
    alignment = next(
        row for row in plan.alignments if row.original_project_time_ms == 2200
    )
    assert alignment.status == "aligned"
    assert alignment.proposed_project_time_ms in {2000, 2500}
    assert abs(alignment.delta_ms or 0) <= 300


def test_cue_plan_lock_is_hash_bound_and_human_approved(tmp_path: Path) -> None:
    proposal = _proposal()
    proposal_path = tmp_path / "music-map.proposal.json"
    write_json(proposal_path, proposal)
    _, music_lock = review_music_map(
        proposal,
        proposal_path=proposal_path,
        reviewer="editor",
        decision="approved",
        bpm=120.0,
        first_downbeat_sample=0,
        meter=4,
    )
    assert music_lock is not None
    music_lock_path = tmp_path / "music-map.lock.json"
    write_json(music_lock_path, music_lock)
    manifest = tmp_path / "render-manifest.json"
    _render_manifest(manifest)
    visual = derive_visual_sync_map(
        manifest, aspect_ratio="16:9", default_flex_ms=300
    )
    visual_path = tmp_path / "visual-sync-map.json"
    write_json(visual_path, visual)
    plan = plan_music_cues(
        music_lock,
        visual,
        music_lock_path=music_lock_path,
        visual_sync_map_path=visual_path,
    )
    plan_path = tmp_path / "cue-plan.proposal.json"
    write_json(plan_path, plan)
    review, cue_lock = review_cue_plan(
        CuePlanProposal.model_validate(read_json(plan_path)),
        cue_plan_path=plan_path,
        reviewer="editor",
        decision="approved",
        notes="timing windows reviewed",
    )
    assert review.decision == "approved"
    assert cue_lock is not None
    assert cue_lock.plan.changes_applied is False
    assert cue_lock.definition_sha256


def test_semantic_pairing_guides_ranking_but_not_timing_window(
    tmp_path: Path,
) -> None:
    proposal = _proposal()
    proposal_path = tmp_path / "music-map.proposal.json"
    write_json(proposal_path, proposal)
    _, music_lock = review_music_map(
        proposal,
        proposal_path=proposal_path,
        reviewer="editor",
        decision="approved",
        bpm=120.0,
        first_downbeat_sample=0,
        meter=4,
    )
    assert music_lock is not None
    music_lock_path = tmp_path / "music-map.lock.json"
    write_json(music_lock_path, music_lock)
    manifest = tmp_path / "render-manifest.json"
    _render_manifest(manifest)
    visual = derive_visual_sync_map(
        manifest, aspect_ratio="9:16", default_flex_ms=300
    )
    visual_path = tmp_path / "visual-sync-map.json"
    write_json(visual_path, visual)
    target_visual = next(
        point for point in visual.points if point.project_time_ms == 2200
    )
    preferred_cue = next(
        cue
        for cue in music_lock.cues
        if cue.kind == "beat" and cue.time_ms == 2500
    )
    semantic = SemanticMusicPairingProposal(
        music_id=music_lock.music_id,
        music_definition_sha256=music_lock.definition_sha256,
        visual_sync_map_sha256=sha256_file(visual_path),
        global_strategy="Use the later beat for a more relaxed transition.",
        section_interpretations=[
            MusicSectionInterpretation(
                section_id="section-001",
                role="neutral",
                energy_level="medium",
                motion_character="steady",
                emotional_character=("neutral",),
                recommended_visual_roles=("continuity",),
                audible_evidence="Steady pulse.",
                confidence=0.8,
            )
        ],
        pairings=[
            SemanticCuePairing(
                visual_event_id=target_visual.visual_event_id,
                preferred_cue_ids=(preferred_cue.cue_id,),
                sync_mode="soft",
                rhythmic_intent="subtle_accent",
                rationale="The following visual idea benefits from a delayed entrance.",
                confidence=0.8,
            )
        ],
        uncertainties=[],
        model_provenance=ModelProvenance(
            model_id="gemini-test",
            api="gemini_interactions",
            sdk="google-genai",
            sdk_version="test",
            run_id="test-run",
            generated_at="2026-07-23T00:00:00+00:00",
        ),
    )
    semantic_path = tmp_path / "semantic-music-pairing.proposal.json"
    write_json(semantic_path, semantic)
    plan = plan_music_cues(
        music_lock,
        visual,
        music_lock_path=music_lock_path,
        visual_sync_map_path=visual_path,
        semantic_pairing=semantic,
        semantic_pairing_path=semantic_path,
    )
    alignment = next(
        row
        for row in plan.alignments
        if row.visual_event_id == target_visual.visual_event_id
    )
    assert plan.semantic_pairing_used is True
    assert alignment.proposed_project_time_ms == 2500
    assert alignment.within_authorized_window is True
