from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

import scripts.build_vertical_crop_audit as audit_script
from jascue_video_lab.media import sha256_file


def test_legacy_adapter_preserves_clip_identity_without_fabricating_asset_id() -> None:
    plan = {
        "project_id": "project",
        "chapters": [
            {
                "feature_id": "scene_01",
                "vertical_event_id": "event_01",
                "vertical_frame_id": "RF000001",
                "observed_visual_evidence": "visible evidence",
                "selection_reason": "reason",
                "quality_risks": [],
                "vertical_strategy": "tracked_crop",
                "vertical_target_description": "visible subject",
            }
        ],
    }
    brief = {
        "chapters": [
            {
                "feature_id": "scene_01",
                "title": "Scene",
                "vertical_crop_mode": "strict",
                "vertical_regions": [],
            }
        ]
    }
    vertical = {"scene_01": {"source_clip_id": "clip_0007"}}

    adapted = audit_script._adapt_legacy_plan(plan, brief, vertical)
    candidate = adapted["shots"][0]["candidates"][0]

    assert candidate["source_asset_id"] is None
    assert candidate["source_clip_id"] == "clip_0007"


def test_alternative_candidate_tolerates_clip_only_identity() -> None:
    candidate = {
        "candidate_id": "candidate_01",
        "source_clip_id": "clip_0007",
        "frame_id": "RF000001",
        "observed_visual_evidence": "visible evidence",
        "selection_reason": "reason",
        "quality_risks": [],
    }

    extracted = audit_script._alternative_candidate(candidate)

    assert extracted["source_asset_id"] is None
    assert extracted["source_clip_id"] == "clip_0007"
    assert extracted["event_id"] is None


def test_saved_crop_keyframes_are_authoritative_even_with_one_matching_track(
    monkeypatch,
    tmp_path: Path,
) -> None:
    def fail_if_recomputed(*_args, **_kwargs):
        raise AssertionError("saved render geometry must not be recomputed")

    monkeypatch.setattr(audit_script, "_usable_track_centers", fail_if_recomputed)
    rendered = {
        "crop_keyframes": [
            {"time_ms": 0, "crop_x_pixels": 120},
            {"time_ms": 1000, "crop_x_pixels": 140},
        ],
        "crop_coordinate_space": "orientation_corrected_source_pixels",
        "geometry_feasible": True,
    }
    track_file = tmp_path / "segmentation-track.json"
    unrelated_track_file = tmp_path / "unrelated" / "segmentation-track.json"

    result = audit_script._track_audit(
        rendered=rendered,
        track_files=[track_file, unrelated_track_file],
        matching_tracks=[(track_file, object())],
    )

    assert result is not None
    assert result["source"] == "render_manifest_authoritative_geometry"
    assert result["authoritative_for_rendered_output"] is True
    assert result["crop_keyframes"] == rendered["crop_keyframes"]
    assert result["track_paths"] == [str(track_file.resolve())]


def test_recomputed_geometry_is_explicitly_comparison_only(
    monkeypatch,
    tmp_path: Path,
) -> None:
    fake_track = SimpleNamespace(
        target_description="visible subject",
        analysis_fps=2.0,
        analysis_start_ms=0,
        analysis_end_ms=1000,
        seed_source_width=1920,
        seed_source_height=1080,
        state_counts={"tracked": 2},
    )
    monkeypatch.setattr(
        audit_script,
        "_usable_track_centers",
        lambda _track: ([0, 1000], [0.4, 0.5], [[0.2, 0.3, 0.6, 0.7]] * 2),
    )
    monkeypatch.setattr(
        audit_script,
        "_vertical_crop_geometry",
        lambda *_args, **_kwargs: (
            [0.2, 0.3],
            {"crop_keyframes": [{"time_ms": 0, "crop_x_pixels": 100}]},
        ),
    )
    track_file = tmp_path / "segmentation-track.json"

    result = audit_script._track_audit(
        rendered={},
        track_files=[track_file],
        matching_tracks=[(track_file, fake_track)],
    )

    assert result is not None
    assert result["source"] == "current_algorithm_comparison_not_rendered_geometry"
    assert result["authoritative_for_rendered_output"] is False
    assert "may differ from the rendered video" in result["comparison_warning"]


def test_resolver_only_returns_single_track_selected_by_manifest_fingerprint(
    monkeypatch,
    tmp_path: Path,
) -> None:
    geometry = tmp_path / "geometry" / "scene_01" / "vertical"
    selected = geometry / "selected" / "segmentation-track.json"
    stale = geometry / "stale" / "segmentation-track.json"
    selected.parent.mkdir(parents=True)
    stale.parent.mkdir(parents=True)
    selected.write_text('{"fingerprint":"selected"}', encoding="utf-8")
    stale.write_text('{"fingerprint":"stale"}', encoding="utf-8")
    monkeypatch.setattr(
        audit_script,
        "SegmentationTrack",
        SimpleNamespace(model_validate=lambda value: value),
    )
    monkeypatch.setattr(
        audit_script,
        "_track_geometry_fingerprint",
        lambda track: "a" * 64 if track["fingerprint"] == "selected" else "b" * 64,
    )

    tracks, lineage = audit_script._resolve_manifest_tracks(
        render_manifest_dir=tmp_path,
        feature_id="scene_01",
        rendered={"track_geometry_fingerprint": "a" * 64},
    )

    assert [path for path, _track in tracks] == [selected]
    assert lineage is not None
    assert lineage["match_kind"] == "single_track_geometry_fingerprint"
    assert lineage["track_artifacts"] == [
        {
            "path": str(selected.resolve()),
            "sha256": sha256_file(selected),
            "geometry_fingerprint": "a" * 64,
        }
    ]


def test_resolver_fails_closed_when_single_track_lineage_is_ambiguous(
    monkeypatch,
    tmp_path: Path,
) -> None:
    geometry = tmp_path / "geometry" / "scene_01" / "vertical"
    for name in ("first", "second"):
        path = geometry / name / "segmentation-track.json"
        path.parent.mkdir(parents=True)
        path.write_text('{"fingerprint":"same"}', encoding="utf-8")
    monkeypatch.setattr(
        audit_script,
        "SegmentationTrack",
        SimpleNamespace(model_validate=lambda value: value),
    )
    monkeypatch.setattr(
        audit_script,
        "_track_geometry_fingerprint",
        lambda _track: "a" * 64,
    )

    with pytest.raises(ValueError, match="ambiguous track lineage"):
        audit_script._resolve_manifest_tracks(
            render_manifest_dir=tmp_path,
            feature_id="scene_01",
            rendered={"track_geometry_fingerprint": "a" * 64},
        )


def test_resolver_reproduces_single_required_region_composite(
    monkeypatch,
    tmp_path: Path,
) -> None:
    regions = [
        {
            "region_id": "title_line",
            "target_description": "the complete visible title line",
            "kind": "text_region",
            "role": "required",
        }
    ]
    region_key = audit_script.hashlib.sha256(
        audit_script.json.dumps(
            regions,
            ensure_ascii=False,
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()[:12]
    track_path = (
        tmp_path
        / "geometry"
        / "scene_01"
        / "vertical"
        / f"regions-{region_key}"
        / "regions"
        / "title_line"
        / "sam21"
        / "segmentation-track.json"
    )
    track_path.parent.mkdir(parents=True)
    track_path.write_text('{"fingerprint":"region"}', encoding="utf-8")
    fingerprint = "a" * 64
    expected = audit_script._shared_geometry_fingerprint(regions, [fingerprint])
    monkeypatch.setattr(
        audit_script,
        "SegmentationTrack",
        SimpleNamespace(model_validate=lambda value: value),
    )
    monkeypatch.setattr(
        audit_script,
        "_track_geometry_fingerprint",
        lambda _track: fingerprint,
    )

    tracks, lineage = audit_script._resolve_manifest_tracks(
        render_manifest_dir=tmp_path,
        feature_id="scene_01",
        rendered={
            "track_geometry_fingerprint": expected,
            "vertical_regions": regions,
        },
    )

    assert [path for path, _track in tracks] == [track_path]
    assert lineage is not None
    assert lineage["match_kind"] == (
        "single_required_region_composite_geometry_fingerprint"
    )


def test_resolver_requires_shared_session_track_hashes_and_composite_fingerprint(
    monkeypatch,
    tmp_path: Path,
) -> None:
    regions = [
        {
            "region_id": "subject_a",
            "target_description": "first visible subject",
            "kind": "subject",
            "role": "required",
        },
        {
            "region_id": "subject_b",
            "target_description": "second visible subject",
            "kind": "subject",
            "role": "required",
        },
    ]
    region_key = audit_script.hashlib.sha256(
        audit_script.json.dumps(
            regions,
            ensure_ascii=False,
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()[:12]
    geometry = tmp_path / "geometry" / "scene_01" / "vertical"
    session = (
        geometry
        / f"regions-{region_key}"
        / "shared-sam21"
        / "session-01"
    )
    track_paths = [
        session / "targets" / target_id / "segmentation-track.json"
        for target_id in ("subject_a", "subject_b")
    ]
    for index, path in enumerate(track_paths):
        path.parent.mkdir(parents=True)
        path.write_text(f'{{"index":{index}}}', encoding="utf-8")
    fingerprints = ["a" * 64, "b" * 64]
    session_manifest = {
        "targets": [
            {
                "target_id": target_id,
                "track_path": str(path.relative_to(session)),
                "track_sha256": sha256_file(path),
            }
            for target_id, path in zip(
                ("subject_a", "subject_b"), track_paths, strict=True
            )
        ]
    }
    audit_script.write_json(session / "shared-session.json", session_manifest)
    monkeypatch.setattr(
        audit_script,
        "SegmentationTrack",
        SimpleNamespace(model_validate=lambda value: value),
    )
    monkeypatch.setattr(
        audit_script,
        "_track_geometry_fingerprint",
        lambda track: fingerprints[track["index"]],
    )
    expected = audit_script._shared_geometry_fingerprint(regions, fingerprints)

    tracks, lineage = audit_script._resolve_manifest_tracks(
        render_manifest_dir=tmp_path,
        feature_id="scene_01",
        rendered={
            "track_geometry_fingerprint": expected,
            "vertical_regions": regions,
        },
    )

    assert [path for path, _track in tracks] == track_paths
    assert lineage is not None
    assert lineage["match_kind"] == (
        "shared_session_composite_geometry_fingerprint"
    )

    track_paths[0].write_text('{"index":0,"tampered":true}', encoding="utf-8")
    with pytest.raises(ValueError, match="no hash/fingerprint-verified track lineage"):
        audit_script._resolve_manifest_tracks(
            render_manifest_dir=tmp_path,
            feature_id="scene_01",
            rendered={
                "track_geometry_fingerprint": expected,
                "vertical_regions": regions,
            },
        )


def test_manifest_video_requires_exact_hash_dimensions_and_duration(
    monkeypatch,
    tmp_path: Path,
) -> None:
    video = tmp_path / "vertical.mp4"
    video.write_bytes(b"video")
    media = SimpleNamespace(
        path=str(video.resolve()),
        sha256="a" * 64,
        duration_ms=1234,
        video=SimpleNamespace(coded_width=1080, coded_height=1920),
    )
    monkeypatch.setattr(audit_script, "probe_video", lambda _path: media)
    manifest = {
        "vertical": {
            "media": {
                "sha256": "a" * 64,
                "width": 1080,
                "height": 1920,
                "duration_seconds": 1.234,
            }
        }
    }

    verified = audit_script._manifest_video(video, manifest)

    assert verified["validation"] == "exact_hash_dimensions_duration_match"
    assert verified["path"] == str(video.resolve())

    manifest["vertical"]["media"]["sha256"] = "b" * 64
    with pytest.raises(ValueError, match="does not match.*sha256"):
        audit_script._manifest_video(video, manifest)


def test_manifest_video_fails_closed_without_media_provenance(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="vertical.media provenance"):
        audit_script._manifest_video(tmp_path / "anything.mp4", {"vertical": {}})
