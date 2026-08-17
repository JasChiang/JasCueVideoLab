import pytest

from montagewright.candidate_commitments import (
    CandidateCommitments,
    CandidateOption,
    bind_selection_content_contracts,
    resolve_candidate_commitments,
)
from montagewright.cli import _delivery_selection
from montagewright.motion import MotionInterval
from montagewright.planner import (
    MaterialItem,
    action_contract_disagreements,
    frame_disagreements,
)
from montagewright.reframe import (
    CropPath,
    Keyframe,
    Observation,
    build_crop_path,
    build_look_path,
    build_tilt_path,
    build_zoom_path,
    ffmpeg_crop_expression,
    camera_delivery_faults,
)
from montagewright.executor import CropBox
from montagewright.schema import DegradationStep, Look, Reframe
from montagewright.spans import Span
from montagewright.webapp import _manual_replacement_plan


def _commitments() -> CandidateCommitments:
    return CandidateCommitments(
        contract_version="candidate-commitment-v5-visual-relationships",
        material_digest="0" * 64,
        direction_sha256="1" * 64,
        target_aspect="9:16",
        target_seconds=10.0,
        options=(CandidateOption(
            commitment_id="c1",
            purpose="show the phone UI",
            required=True,
            picture_role="illustrative_broll",
            span_id="C1:s00",
            tier="primary",
            min_supported_seconds=2.0,
            presentation_intent="reveal_endpoint",
            content_policy="continuous_process",
            content_action_id="none",
            required_visuals=("v01", "v03"),
            visual_relationship="ordered",
            motion_preference="native_first",
            target_id="none",
            why="read the phone, not the prop beside it",
            feasible_treatments=("use_source_motion", "hold"),
        ),),
    )


def test_stable_visual_ids_replace_a_contradictory_free_text_target():
    item = MaterialItem(
        source_id="C1", duration_seconds=8.0, summary="phone beside vase",
        subject_geometry=(
            ("the Pixel phone", None, 0.25, 0.5, 0.2, 0.7),
            ("the green vase", None, 0.75, 0.5, 0.2, 0.7),
            ("the phone UI", None, 0.25, 0.5, 0.18, 0.6),
        ),
    )
    shot = {
        "source_id": "C1", "span_id": "C1:s00", "commitment_id": "c1",
        "looks": [{
            "at": "the green vase", "includes": ["v01", "v03"],
            "entity_id": "none", "seconds": 2.0, "framing": "centre",
            "must_be_whole": False, "presentation_intent": "reveal_endpoint",
        }],
    }

    bind_selection_content_contracts([shot], _commitments(), [item])

    assert shot["looks"][0]["at"] == "the Pixel phone + the phone UI"


def test_wide_required_visual_is_promoted_to_a_sequential_read():
    item = MaterialItem(
        source_id="C1", duration_seconds=6.0, summary="wide event wordmark",
        spans=(Span("C1:s00", "C1", 0.0, 6.0, "wordmark", "locked"),),
        subject_geometry=(("the event wordmark", None, 0.5, 0.5, 0.78, 0.2),),
        crop_width=0.3164, pan_room=0.68,
    )
    direction = {"direction": "read the complete event wordmark", "candidate_options": [{
        "commitment_id": "c1", "purpose": "establish the event",
        "required": True, "picture_role": "establishing",
        "span_id": "C1:s00", "tier": "primary",
        "min_supported_seconds": "0:03", "presentation_intent": "centered_hold",
        "content_policy": "static_display", "content_action_id": "none",
        "required_visuals": ["v01"], "required_evidence": [],
        "visual_relationship": "single", "motion_preference": "virtual_allowed",
        "target_id": "none", "recommended_treatment": "hold",
        "suggested_move": "hold", "camera_route": "hold the event sign",
        "motion_reason": "the vertical crop is narrower than the wordmark",
        "fallback_treatment": "hold", "why": "make the whole mark readable",
    }]}

    resolved = resolve_candidate_commitments(
        direction, [item], material_digest="a" * 64, aspect="9:16",
        target_seconds=20.0, grounding_target_ids=(), grounding_sha256=None,
    )

    assert resolved.options[0].presentation_intent == "sequential_read"
    assert resolved.options[0].direction_treatment == "hold"
    assert resolved.options[0].preferred_treatment == "reveal"
    assert "promoted the content" in resolved.options[0].feasibility_reason


def test_small_native_drift_cannot_claim_to_read_a_wide_visual():
    item = MaterialItem(
        source_id="C1", duration_seconds=6.0, summary="wide event wordmark",
        spans=(Span("C1:s00", "C1", 0.0, 6.0, "wordmark", "authored"),),
        subject_geometry=(("the event wordmark", None, 0.5, 0.5, 0.78, 0.2),),
        crop_width=0.3164,
        motion=(MotionInterval(
            event_id="m00", starts_seconds=0.0, ends_seconds=3.0,
            state="moving", peak_vw_s=0.04, travel_vw=0.10, settles=True,
        ),),
    )
    shot = {
        "source_id": "C1", "span_id": "C1:s00", "start_seconds": 0.0,
        "seconds_needed": 3.0, "camera_intent": "use_source_motion",
        "source_motion_role": "authored", "frame": "settles",
        "looks": [{
            "at": "the event wordmark", "includes": ["v01"],
            "entity_id": "none", "seconds": 1.0, "framing": "centre",
            "must_be_whole": False, "presentation_intent": "sequential_read",
        }],
    }

    faults = frame_disagreements([shot], [item])

    assert any("selected source window supplies 0.10" in fault for fault in faults)
    assert any("choose reveal or multi_stop" in fault for fault in faults)


def test_intentional_cut_must_overlap_the_action_it_names():
    item = MaterialItem(
        source_id="C1", duration_seconds=8.0, summary="gesture",
        action_ids=("a01",), action_windows=(("a01", 2.0, 4.0),),
        spans=(Span("C1:s00", "C1", 0.0, 8.0),),
    )
    shot = {
        "source_id": "C1", "span_id": "C1:s00",
        "picture_role": "primary_action", "content_policy": "continuous_process",
        "action_id": "a01", "action_treatment": "intentional_cut",
        "start_seconds": 0.0, "seconds_needed": 1.0,
        "why": "cut during the gesture",
    }

    faults = action_contract_disagreements([shot], [item])

    assert any("ends before that action begins" in fault for fault in faults)


def test_designed_multi_stop_pan_eases_even_with_more_than_four_keys():
    path = build_look_path(
        [
            (0.5, 0.2, 0.5, 0.3164),
            (0.5, 0.5, 0.5, 0.3164),
            (0.5, 0.8, 0.5, 0.3164),
        ],
        source_aspect=16 / 9, target_aspect=9 / 16,
        duration_seconds=6.0, energy="active",
        tracks=[[(0.0, 0.2, 0.5), (6.0, 0.3, 0.5)]] * 3,
        track_during_stops=False,
    )

    assert len(path.keyframes) > 4
    _, _, x, _ = ffmpeg_crop_expression(path, 3840, 2160)
    assert "(3-2*" in x
    assert path.keyframes[-1].crop.x > path.keyframes[0].crop.x


def test_unreachable_pan_still_reaches_endpoint_but_requires_replan():
    degradations = []
    path = build_look_path(
        [(0.35, 0.2, 0.5, 0.3164), (0.35, 0.8, 0.5, 0.3164)],
        source_aspect=16 / 9, target_aspect=9 / 16,
        duration_seconds=0.6, energy="calm", clip_id="k03",
        degradations=degradations,
    )

    assert not path.is_static
    assert path.keyframes[-1].crop.x > path.keyframes[0].crop.x
    assert any(step.adjudication == "replan" for step in degradations)


def test_replan_degradation_cannot_be_reported_ready():
    degradation = DegradationStep(
        clip_id="k00", ladder="other", ladder_other="camera_failed",
        trigger="camera missed its endpoint", adjudication="replan",
        adjudication_reason="choose another treatment",
    )

    delivered, status = _delivery_selection(
        {"shots": [{"identity_status": "not_applicable"}]}, {}, [degradation]
    )

    assert status == "needs_review"
    assert delivered["shots"][0]["delivery_status"] == "needs_review"
    assert "another treatment" in delivered["shots"][0]["delivery_issue"]


def test_unreviewed_degradation_cannot_be_reported_ready() -> None:
    degradation = DegradationStep(
        clip_id="k00",
        ladder="slower_follow",
        trigger="the compiled follow exceeded its camera budget",
    )

    delivered, status = _delivery_selection(
        {"shots": [{"identity_status": "not_applicable"}]}, {}, [degradation]
    )

    assert status == "needs_review"
    assert delivered["shots"][0]["delivery_status"] == "needs_review"
    assert "尚未逐顆驗收" in delivered["shots"][0]["delivery_issue"]


def test_locally_accepted_degradation_can_still_be_ready() -> None:
    degradation = DegradationStep(
        clip_id="k00",
        ladder="other",
        ladder_other="redundant_readback_suppressed",
        trigger="a redundant return was removed",
        adjudication="accept",
        adjudication_reason="the useful endpoint is preserved",
    )

    delivered, status = _delivery_selection(
        {"shots": [{"identity_status": "not_applicable"}]}, {}, [degradation]
    )

    assert status == "ready"
    assert "delivery_status" not in delivered["shots"][0]


def test_manual_replacement_preserves_a_supported_push() -> None:
    plan = _manual_replacement_plan(
        {"camera_intent": "push_in", "looks": [{"framing": "thirds"}]},
        {"summary": "phone", "subjects": [{"label": "the phone", "moves": False}],
         "segments": [{"motion_role": "locked"}]},
        source_id="C2", span_id="C2:s00", start=0.0, end=5.0,
        duration=3.0, confirms_identity=False,
    )

    assert plan["camera_intent"] == "push_in"
    assert plan["looks"][0]["at"] == "the phone"
    assert plan["delivery_status"] == "needs_review"


def test_manual_replacement_uses_a_compatible_camera_fallback() -> None:
    plan = _manual_replacement_plan(
        {"camera_intent": "use_source_motion", "fallback_treatment": "push_in"},
        {"summary": "phone", "subjects": [{"label": "the phone", "moves": False}],
         "segments": [{"motion_role": "locked"}]},
        source_id="C2", span_id="C2:s00", start=0.0, end=5.0,
        duration=3.0, confirms_identity=True,
    )

    assert plan["camera_intent"] == "push_in"
    assert "不適用此素材" in plan["delivery_issue"]


def test_sequential_read_with_complete_holds_cannot_rebound() -> None:
    degradations = []
    path = build_look_path(
        [
            (0.5, 0.20, 0.5, 0.3164),
            (0.5, 0.52, 0.5, 0.3164),
            (0.5, 0.82, 0.5, 0.3164),
            (0.5, 0.61, 0.5, 0.3164),
        ],
        source_aspect=16 / 9,
        target_aspect=9 / 16,
        duration_seconds=5.0,
        energy="active",
        clip_id="k05",
        degradations=degradations,
        continuous_read=False,
        monotonic_route=True,
    )

    centres = [one.crop.x + one.crop.width / 2 for one in path.keyframes]
    assert all(later + 1e-6 >= earlier for earlier, later in zip(centres, centres[1:]))
    assert centres[-1] > 0.75
    assert any(
        step.ladder_other == "redundant_readback_suppressed"
        and step.adjudication == "accept"
        for step in degradations
    )


def test_sequential_read_cannot_silently_absorb_a_push_endpoint() -> None:
    with pytest.raises(ValueError, match="separate treatments"):
        build_look_path(
            [
                (0.4, 0.20, 0.5, 0.3164),
                (0.4, 0.50, 0.5, 0.3164),
                (0.4, 0.80, 0.5, 0.3164),
                (0.8, 0.49, 0.5, 0.2848),
            ],
            source_aspect=16 / 9, target_aspect=9 / 16,
            duration_seconds=3.0, energy="active", clip_id="k05",
            monotonic_route=True,
        )


def test_camera_delivery_audit_rejects_a_push_that_became_a_hold() -> None:
    reframe = Reframe(
        looks=[Look(at="LED", framing="fill")],
        camera_move="push_in",
        editorial_intent="push_in",
    )
    crop = CropBox(0.3, 0.0, 0.3, 1.0)
    path = CropPath([Keyframe(0.0, crop), Keyframe(2.5, crop)])

    assert camera_delivery_faults(
        reframe, path, duration_seconds=2.5
    ) == ("push_in did not finish tighter than it started",)


def test_camera_delivery_audit_allows_native_motion_without_a_digital_path() -> None:
    reframe = Reframe(
        looks=[Look(at="product")],
        camera_move="use_source_motion",
        editorial_intent="use_source_motion",
        source_motion_role="authored",
    )

    assert not camera_delivery_faults(reframe, None, duration_seconds=3.0)


def test_moving_follow_reports_the_visibility_of_the_compiled_path() -> None:
    degradations = []
    observations = [
        Observation(0.0, 0.20, 0.5, 0.70, 0.8),
        Observation(1.0, 0.45, 0.5, 0.70, 0.8),
        Observation(2.0, 0.70, 0.5, 0.70, 0.8),
    ]

    path = build_crop_path(
        observations,
        source_aspect=16 / 9,
        target_aspect=9 / 16,
        energy="active",
        clip_id="k02",
        min_visible=0.9,
        degradations=degradations,
    )

    assert not path.is_static
    assert any(
        step.ladder_other == "subject_larger_than_crop"
        for step in degradations
    )


def test_4k_vertical_delivery_uses_real_resolution_room_for_tilt() -> None:
    observations = [
        Observation(0.0, 0.5, 0.30, 0.15, 0.15),
        Observation(2.0, 0.5, 0.70, 0.15, 0.15),
    ]
    degradations = []

    path = build_tilt_path(
        observations,
        source_aspect=16 / 9,
        target_aspect=9 / 16,
        source_width=3840,
        source_height=2160,
        output_width=1080,
        output_height=1920,
        energy="active",
        clip_id="k04",
        degradations=degradations,
    )

    assert not path.is_static
    assert path.keyframes[0].crop.height == 1920 / 2160
    assert path.keyframes[-1].crop.y > path.keyframes[0].crop.y
    assert not any(step.ladder == "static_on_subject" for step in degradations)


def test_tracked_push_rests_its_scale_and_ignores_subvisible_jitter() -> None:
    degradations = []
    path = build_zoom_path(
        source_aspect=16 / 9,
        target_aspect=9 / 16,
        duration_seconds=3.0,
        direction="push_in",
        centre_x=0.5,
        centre_y=0.5,
        track=[
            (0.0, 0.40, 0.50),
            (0.3, 0.405, 0.50),
            (1.5, 0.55, 0.50),
            (2.7, 0.70, 0.50),
            (3.0, 0.70, 0.50),
        ],
        energy="active",
        clip_id="k05",
        degradations=degradations,
    )

    assert path.keyframes[0].crop.width == path.keyframes[1].crop.width
    assert path.keyframes[-2].crop.width == path.keyframes[-1].crop.width
    assert path.keyframes[1].crop.x == path.keyframes[0].crop.x
    assert any(
        step.ladder_other == "zoom_followed_subject"
        and step.adjudication == "accept"
        for step in degradations
    )
