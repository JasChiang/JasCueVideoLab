from __future__ import annotations

import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

import jascue_video_lab.sam_tracking as sam_tracking
from jascue_video_lab.media import probe_video
from jascue_video_lab.models import (
    SegmentationSample,
    SemanticIdentityStatus,
    TrackingState,
)
from jascue_video_lab.sam_tracking import (
    _extract_analysis_frames,
    _normalize_shot_manifest,
    _seed_shot,
    _timeline_ms_from_pts,
    approximate_connected_components,
    binary_mask_geometry,
    classify_tracking_state,
    normalized_box_to_xyxy,
    normalized_polygon_to_mask,
    pad_normalized_box,
    require_bbox_track_request_match,
    track_bbox_sam21,
)
from jascue_video_lab.shots import ShotBoundary, ShotManifest, ShotSegment


def test_normalized_box_to_xyxy_is_x_first() -> None:
    assert normalized_box_to_xyxy([100, 200, 600, 800], 2000, 1000) == [
        200.0,
        200.0,
        1200.0,
        800.0,
    ]


def test_padding_preserves_semantic_box_order_and_clamps() -> None:
    assert pad_normalized_box([100, 200, 600, 800], 0.1) == [50, 140, 650, 860]
    assert pad_normalized_box([0, 0, 1000, 1000], 0.5) == [0, 0, 1000, 1000]


def test_normalized_polygon_rasterizes_in_x_y_order() -> None:
    mask = normalized_polygon_to_mask(
        [(100, 200), (600, 200), (600, 800), (100, 800)], 200, 100
    )
    geometry = binary_mask_geometry(mask)
    assert geometry["box_2d"] == pytest.approx([100, 200, 600, 800], abs=6)
    assert geometry["area_ratio"] == pytest.approx(0.3, abs=0.02)


def test_binary_mask_geometry_and_components() -> None:
    np = pytest.importorskip("numpy")
    mask = np.zeros((100, 200), dtype=bool)
    mask[20:60, 40:140] = True
    geometry = binary_mask_geometry(mask)
    assert geometry["area_pixels"] == 4000
    assert geometry["area_ratio"] == pytest.approx(0.2)
    assert geometry["box_2d"] == [200, 200, 700, 600]
    assert geometry["center_2d"] == [450.0, 400.0]
    assert approximate_connected_components(mask) == 1


def test_component_count_ignores_tiny_mask_speckle() -> None:
    np = pytest.importorskip("numpy")
    mask = np.zeros((100, 200), dtype=bool)
    mask[10:90, 20:180] = True
    mask[1, 1] = True
    assert approximate_connected_components(mask) == 1


def test_binary_mask_empty_has_no_guessed_geometry() -> None:
    np = pytest.importorskip("numpy")
    geometry = binary_mask_geometry(np.zeros((8, 8), dtype=bool))
    assert geometry == {
        "area_pixels": 0,
        "area_ratio": 0.0,
        "box_2d": None,
        "center_2d": None,
    }


def test_drift_gate_rejects_cut_even_with_valid_mask() -> None:
    state, reasons = classify_tracking_state(
        area_ratio=0.1,
        connected_components=1,
        mean_positive_probability=0.99,
        previous_area_ratios=[0.1, 0.1],
        center_2d=[500, 500],
        previous_center_2d=[500, 500],
        shot_boundary=True,
    )
    assert state == TrackingState.DRIFT_SUSPECTED
    assert "shot_boundary_requires_new_seed" in reasons


def test_shot_manifest_normalizes_nonzero_pts_and_selects_seed_shot() -> None:
    raw = ShotManifest(
        video_path="/tmp/nonzero.mp4",
        duration_ms=2000,
        detector="test detector",
        threshold=4,
        generated_at="2026-01-01T00:00:00Z",
        boundaries=[
            ShotBoundary(
                boundary_id="boundary-0001",
                frame_pts=58_000,
                frame_time_ms=5800,
                score=12.5,
            )
        ],
        # This represents the detector's pre-normalization fallback. The helper
        # must rebuild shots from exact PTS rather than trust these intervals.
        shots=[
            ShotSegment(
                shot_id="shot-0001",
                start_time_ms=0,
                end_time_ms=2000,
                start_frame_pts=None,
                boundary_source="video_start",
                boundary_score=None,
            )
        ],
    )
    normalized = _normalize_shot_manifest(
        raw,
        duration_ms=2000,
        source_start_pts=50_000,
        time_base_numerator=1,
        time_base_denominator=10_000,
    )

    assert normalized.boundaries[0].frame_pts == 58_000
    assert normalized.boundaries[0].frame_time_ms == 800
    assert [(shot.start_time_ms, shot.end_time_ms) for shot in normalized.shots] == [
        (0, 800),
        (800, 2000),
    ]
    assert _seed_shot(normalized.shots, 799).start_time_ms == 0
    selected = _seed_shot(normalized.shots, 800)
    assert (selected.start_time_ms, selected.end_time_ms) == (800, 2000)
    assert selected.start_frame_pts == 58_000


def test_pts_to_local_timeline_uses_rational_time_base() -> None:
    assert (
        _timeline_ms_from_pts(
            61_440 + 4096,
            source_start_pts=61_440,
            time_base_numerator=1,
            time_base_denominator=12_288,
        )
        == 333
    )


def test_analysis_frame_extraction_is_shot_local_and_preserves_source_pts(
    tmp_path: Path,
) -> None:
    source = tmp_path / "nonzero-pts.mp4"
    subprocess.run(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-f",
            "lavfi",
            "-i",
            "testsrc=size=320x180:rate=4:duration=2",
            "-vf",
            "setpts=PTS+5/TB",
            "-c:v",
            "mpeg4",
            "-q:v",
            "3",
            "-pix_fmt",
            "yuv420p",
            str(source),
        ],
        check=True,
    )
    media = probe_video(source)
    assert media.video.start_pts is not None
    frames, _, _ = _extract_analysis_frames(
        source,
        tmp_path / "frames",
        2,
        320,
        start_time_ms=400,
        end_time_ms=1600,
        source_start_pts=media.video.start_pts,
        time_base_numerator=media.video.time_base.numerator,
        time_base_denominator=media.video.time_base.denominator,
    )

    assert [frame.timeline_time_ms for frame in frames] == [500, 1000, 1500]
    assert all(400 <= frame.timeline_time_ms < 1600 for frame in frames)
    assert all(frame.source_pts > media.video.start_pts for frame in frames)
    assert len(list((tmp_path / "frames").glob("*.jpg"))) == 3


def test_track_initializes_predictor_with_only_the_seed_shot(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    np = pytest.importorskip("numpy")
    source = tmp_path / "source.mp4"
    checkpoint = tmp_path / "checkpoint.pt"
    source.write_bytes(b"test video placeholder")
    checkpoint.write_bytes(b"test checkpoint placeholder")
    output_dir = tmp_path / "track"
    raw_manifest = ShotManifest(
        video_path=str(source),
        duration_ms=3000,
        detector="test detector",
        threshold=4,
        generated_at="2026-01-01T00:00:00Z",
        boundaries=[
            ShotBoundary(
                boundary_id="boundary-0001",
                frame_pts=1000,
                frame_time_ms=1000,
                score=15,
            ),
            ShotBoundary(
                boundary_id="boundary-0002",
                frame_pts=2000,
                frame_time_ms=2000,
                score=18,
            ),
        ],
        shots=[
            ShotSegment(
                shot_id="shot-0001",
                start_time_ms=0,
                end_time_ms=3000,
                start_frame_pts=None,
                boundary_source="video_start",
                boundary_score=None,
            )
        ],
    )
    media = SimpleNamespace(
        duration_ms=3000,
        asset_id="sha256:" + "a" * 64,
        video=SimpleNamespace(
            start_pts=0,
            time_base=SimpleNamespace(numerator=1, denominator=1000),
        ),
    )
    call_order: list[str] = []

    def fake_extract(
        video_path: Path,
        frames_dir: Path,
        analysis_fps: float,
        max_side: int,
        **timing: int,
    ) -> tuple[list[sam_tracking._AnalysisFrame], int, int]:
        del video_path, analysis_fps, max_side
        call_order.append("extract")
        assert timing["start_time_ms"] == 1200
        assert timing["end_time_ms"] == 1800
        frames_dir.mkdir(parents=True)
        records: list[sam_tracking._AnalysisFrame] = []
        for index, (pts, time_ms) in enumerate(((1300, 1300), (1600, 1600))):
            path = frames_dir / f"{index:06d}.jpg"
            path.write_bytes(b"frame")
            records.append(
                sam_tracking._AnalysisFrame(
                    path=path,
                    source_pts=pts,
                    timeline_time_ms=time_ms,
                )
            )
        return records, 20, 12

    class FakeTensor:
        def __init__(self) -> None:
            self.array = np.ones((12, 20), dtype=np.float32)

        def __getitem__(self, _: object) -> "FakeTensor":
            return self

        def detach(self) -> "FakeTensor":
            return self

        def float(self) -> "FakeTensor":
            return self

        def cpu(self) -> "FakeTensor":
            return self

        def numpy(self) -> object:
            return self.array

    class FakePredictor:
        def init_state(self, *, video_path: str, **_: object) -> object:
            call_order.append("init")
            assert call_order == ["detect", "extract", "init"]
            assert [path.name for path in sorted(Path(video_path).glob("*.jpg"))] == [
                "000000.jpg",
                "000001.jpg",
            ]
            return object()

        def add_new_points_or_box(self, **_: object) -> tuple[None, None, FakeTensor]:
            return None, None, FakeTensor()

        def propagate_in_video(
            self, _: object, *, start_frame_idx: int, reverse: bool
        ) -> object:
            del start_frame_idx
            indices = (1, 0) if reverse else (0, 1)
            for index in indices:
                yield index, None, FakeTensor()

    def fake_detect(*_: object, **__: object) -> ShotManifest:
        call_order.append("detect")
        return raw_manifest

    fake_torch = SimpleNamespace(__version__="test")
    monkeypatch.setattr(sam_tracking, "probe_video", lambda _: media)
    monkeypatch.setattr(sam_tracking, "detect_shots_ffmpeg", fake_detect)
    monkeypatch.setattr(sam_tracking, "_extract_analysis_frames", fake_extract)
    monkeypatch.setattr(
        sam_tracking,
        "_require_segmentation_dependencies",
        lambda: (np, fake_torch, lambda *_, **__: FakePredictor()),
    )
    monkeypatch.setattr(sam_tracking, "_save_mask", lambda *_, **__: "b" * 64)
    monkeypatch.setattr(sam_tracking, "_render_overlay", lambda *_, **__: None)
    monkeypatch.setattr(sam_tracking, "_render_video", lambda *_, **__: None)

    track = track_bbox_sam21(
        video_path=source,
        checkpoint_path=checkpoint,
        seed_time_ms=1500,
        seed_box_2d=[100, 100, 900, 900],
        target_description="the selected object instance",
        output_dir=output_dir,
        seed_source="test bbox",
        device="cpu",
        allowed_start_ms=1200,
        allowed_end_ms=1800,
    )

    assert track.seed_sample_index == 1
    assert track.analysis_start_ms == 1200
    assert track.analysis_end_ms == 1800
    assert track.method == "bbox_seed_sam2_video_mask_propagation"
    assert track.source_start_pts == 0
    assert track.source_time_base is not None
    assert (track.source_time_base.numerator, track.source_time_base.denominator) == (1, 1000)
    assert [sample.source_pts for sample in track.samples] == [1300, 1600]
    assert [sample.analysis_sample_time_ms for sample in track.samples] == [1300, 1600]
    assert all(sample.timing_basis == "decoded_source_pts" for sample in track.samples)
    assert all(
        sample.tracking_state != TrackingState.DRIFT_SUSPECTED
        for sample in track.samples
    )
    request = {
        "video_path": source,
        "asset_id": media.asset_id,
        "target_description": "the selected object instance",
        "seed_time_ms": 1500,
        "seed_box_2d": [100, 100, 900, 900],
        "seed_box_padding_ratio": 0.0,
        "analysis_fps": 2.0,
        "analysis_start_ms": 1200,
        "analysis_end_ms": 1800,
        "checkpoint_sha256": track.model_provenance.checkpoint_sha256,
    }
    require_bbox_track_request_match(track, **request)
    stale = track.model_copy(update={"target_description": "a different instance"})
    with pytest.raises(ValueError, match="does not match bbox seed request"):
        require_bbox_track_request_match(stale, **request)


def test_primary_sam_path_rejects_polygon_seed_before_loading_sam(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="polygon seeds are disabled"):
        track_bbox_sam21(
            video_path=tmp_path / "missing.mp4",
            checkpoint_path=tmp_path / "missing.pt",
            seed_time_ms=0,
            seed_box_2d=[100, 100, 900, 900],
            target_description="the selected object instance",
            output_dir=tmp_path / "out",
            seed_source="test",
            seed_mask_polygon_xy=[(100, 100), (900, 100), (900, 900)],
        )


def test_sam_rejects_seed_from_another_asset_before_loading_dependencies(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "source.mp4"
    subprocess.run(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-f",
            "lavfi",
            "-i",
            "color=c=blue:s=160x90:r=5:d=1",
            "-c:v",
            "mpeg4",
            str(source),
        ],
        check=True,
    )
    checkpoint = tmp_path / "checkpoint.pt"
    checkpoint.write_bytes(b"checkpoint")

    def forbidden_dependencies() -> object:
        raise AssertionError("asset mismatch must fail before loading SAM")

    monkeypatch.setattr(
        sam_tracking,
        "_require_segmentation_dependencies",
        forbidden_dependencies,
    )
    with pytest.raises(ValueError, match="asset_id does not match"):
        track_bbox_sam21(
            video_path=source,
            checkpoint_path=checkpoint,
            seed_time_ms=0,
            seed_box_2d=[100, 100, 900, 900],
            target_description="the selected subject",
            output_dir=tmp_path / "out",
            seed_source="proposal from another source",
            asset_id="sha256:" + "0" * 64,
        )
    assert not (tmp_path / "out").exists()


def test_drift_gate_distinguishes_low_confidence_from_lost() -> None:
    low, _ = classify_tracking_state(
        area_ratio=0.1,
        connected_components=8,
        mean_positive_probability=0.55,
        previous_area_ratios=[0.1],
        center_2d=[500, 500],
        previous_center_2d=[500, 500],
        shot_boundary=False,
    )
    lost, reasons = classify_tracking_state(
        area_ratio=0,
        connected_components=0,
        mean_positive_probability=None,
        previous_area_ratios=[0.1],
        center_2d=None,
        previous_center_2d=[500, 500],
        shot_boundary=False,
    )
    assert low == TrackingState.LOW_CONFIDENCE
    assert lost == TrackingState.LOST
    assert reasons == ["mask_empty"]


def test_sample_contract_forbids_geometry_when_lost() -> None:
    common = {
        "sample_index": 0,
        "analysis_sample_time_ms": 0,
        "source_pts": None,
        "timing_basis": "uniform_ffmpeg_analysis_sample",
        "mask_path": None,
        "mask_sha256": None,
        "mask_area_pixels": 0,
        "mask_area_ratio": 0,
        "connected_components": 0,
        "derived_tracking_box": [1, 1, 2, 2],
        "center_2d": [1.5, 1.5],
        "mean_positive_probability": None,
        "scene_cut_score": None,
        "shot_boundary": False,
        "tracking_state": "lost",
        "state_reasons": ["mask_empty"],
        "semantic_identity_status": SemanticIdentityStatus.REVALIDATION_REQUIRED,
    }
    with pytest.raises(ValueError, match="empty masks cannot contain geometry"):
        SegmentationSample.model_validate(common)
