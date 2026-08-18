"""Offline regressions distilled from the two canonical one-shot runs."""

from __future__ import annotations


def test_source_motion_never_expands_a_sequential_digital_route() -> None:
    from montagewright.reframe import camera_route_policy, declared_look_centres
    from montagewright.schema import Look, Reframe

    reframe = Reframe(
        camera_move="hold",
        editorial_intent="use_source_motion",
        looks=[Look(
            at="the complete row",
            presentation_intent="sequential_read",
            must_be_whole=False,
        )],
    )

    assert not camera_route_policy(reframe).expand_sequential_read
    assert declared_look_centres(
        reframe, centre_x=0.5, subject_width=0.9, crop_width=0.3164,
    ) == [0.5]


def test_tracking_jitter_is_not_labelled_multi_stop() -> None:
    from montagewright.pipeline import _digital_motion_of
    from montagewright.reframe import CropBox, CropPath, Keyframe

    def crop(cx: float) -> CropBox:
        return CropBox(x=cx - 0.15, y=0.0, width=0.3, height=1.0)

    path = CropPath([
        Keyframe(0.0, crop(0.5000)),
        Keyframe(0.5, crop(0.5005)),
        Keyframe(1.0, crop(0.5000)),
        Keyframe(1.5, crop(0.5150)),
    ])

    assert "multi_stop" not in _digital_motion_of(path)


def test_real_rebound_is_still_labelled_multi_stop() -> None:
    from montagewright.pipeline import _digital_motion_of
    from montagewright.reframe import CropBox, CropPath, Keyframe

    def crop(cx: float) -> CropBox:
        return CropBox(x=cx - 0.15, y=0.0, width=0.3, height=1.0)

    path = CropPath([
        Keyframe(0.0, crop(0.35)),
        Keyframe(0.8, crop(0.72)),
        Keyframe(1.6, crop(0.50)),
    ])

    assert "multi_stop" in _digital_motion_of(path)


def test_static_hold_audit_reads_editorial_intent_not_lossy_move_label() -> None:
    from montagewright.pipeline import _audit_static_holds
    from montagewright.schema import Clip, EDL, Look, Reframe

    moving = Clip(
        clip_id="k07", source_id="C1",
        approx_in_seconds=0.0, approx_out_seconds=4.0,
        reframe=Reframe(
            camera_move="hold", editorial_intent="reveal",
            source_motion_role="locked",
            looks=[Look(at="answer", presentation_intent="reveal_endpoint")],
        ),
    )

    _, notes = _audit_static_holds(EDL(project_id="p", clips=[moving]), 2.0)
    assert notes == []


def test_shot_key_is_stable_when_neighbour_positions_change() -> None:
    from montagewright.camera import shot_key

    shot = {
        "commitment_id": "c7", "span_id": "C1:s03",
        "start_offset_seconds": "0:01.2", "camera_intent": "push_in",
        "looks": [{"at": "the lens", "framing": "fill"}],
    }
    moved_in_list = {"clip_id": "k12", **shot}
    locally_normalized = {
        **shot,
        "camera_intent": "hold",
        "delivered_camera_intent": "hold",
        "looks": [{"at": "the lens", "framing": "thirds"}],
    }

    assert shot_key(shot) == shot_key(moved_in_list)
    assert shot_key(shot) == shot_key(locally_normalized)


def test_camera_compiler_only_blocks_semantic_delivery_failures() -> None:
    from montagewright.camera import compile_camera
    from montagewright.reframe import CropBox, CropPath, Keyframe
    from montagewright.schema import Look, Reframe

    endpoint_only = compile_camera(
        Reframe(
            camera_move="push_in", editorial_intent="push_in",
            intent="tighten on the product",
        ),
        path=CropPath([
            Keyframe(0.0, CropBox(x=0.0, y=0.0, width=0.8, height=1.0)),
            Keyframe(0.97, CropBox(x=0.1, y=0.0, width=0.6, height=1.0)),
        ]),
        duration_seconds=1.0,
        stable_key="shot",
        attempt_id="attempt-1",
    )
    # An endpoint that arrives a little short is advisory, not blocking.
    assert not any(f.severity == "blocking_shot" for f in endpoint_only)
    assert endpoint_only[0].severity == "advisory"
    assert endpoint_only[0].attempt_id == "attempt-1"

    static_compare = compile_camera(
        Reframe(
            camera_move="pan",
            editorial_intent="compare",
            looks=[Look(at="one"), Look(at="two")],
        ),
        path=CropPath([
            Keyframe(0.0, CropBox(x=0.0, y=0.0, width=0.8, height=1.0)),
        ]),
        duration_seconds=1.0,
        stable_key="shot",
    )
    # A compare that produced no travel did not deliver its move.
    assert any(f.severity == "blocking_shot" for f in static_compare)
    assert static_compare[0].severity == "blocking_shot"


def test_tracked_push_compares_against_its_tracked_endpoint() -> None:
    from montagewright.reframe import build_look_path

    degradations = []
    build_look_path(
        [(0.35, 0.4, 0.5, 0.5), (0.35, 0.5, 0.5, 0.35)],
        source_aspect=16 / 9,
        target_aspect=9 / 16,
        duration_seconds=4.0,
        energy="dynamic",
        tracks=[
            [(0.0, 0.4, 0.5), (4.0, 0.6, 0.5)],
            [(0.0, 0.4, 0.5), (4.0, 0.6, 0.5)],
        ],
        track_during_stops=True,
        degradations=degradations,
        clip_id="k05",
    )

    assert not any(
        step.ladder_other == "camera_endpoint_not_reached_before_cut"
        for step in degradations
    )


def test_hurried_route_renders_the_speed_limited_path_it_reports() -> None:
    from montagewright.reframe import build_look_path

    degradations = []
    path = build_look_path(
        [(0.35, 0.2, 0.5, 0.5), (0.35, 0.8, 0.5, 0.35)],
        source_aspect=16 / 9,
        target_aspect=9 / 16,
        duration_seconds=1.0,
        energy="calm",
        degradations=degradations,
        clip_id="k05",
    )

    final_x = path.keyframes[-1].crop.x + path.keyframes[-1].crop.width / 2
    assert final_x < 0.8
    assert any(
        step.ladder_other == "camera_endpoint_not_reached_before_cut"
        for step in degradations
    )
