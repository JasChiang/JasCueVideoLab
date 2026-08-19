from __future__ import annotations

import json


def _edl():
    from montagewright.schema import Clip, EDL

    return EDL(
        project_id="resume",
        clips=[
            Clip(
                clip_id="k00", source_id="A",
                approx_in_seconds=0.0, approx_out_seconds=3.0,
                audio_role="discard",
            ),
            Clip(
                clip_id="k01", source_id="B",
                approx_in_seconds=0.0, approx_out_seconds=3.0,
                audio_role="discard",
            ),
        ],
    )


def _grid():
    from montagewright.grounding import BeatGrid, Cue

    return BeatGrid(
        bpm=60.0,
        meter=4,
        duration_seconds=12.0,
        cues=(
            Cue("beat-4", 4.0, "beat", 1.0),
            Cue("beat-8", 8.0, "beat", 1.0),
        ),
    )


def test_preferred_rhythm_drops_optional_snaps_before_exceeding_target():
    from montagewright.grounding import ground_timeline
    from montagewright.planner import _fit_preferred_rhythm_to_target

    edl = _edl()
    assert ground_timeline(edl, _grid()).duration_seconds == 8.0

    fitted = _fit_preferred_rhythm_to_target(
        edl, _grid(), target_seconds=3.0, duration_mode="preferred",
    )

    # At 60 BPM in 4/4, preferred delivery may use one 4s bar of slack.
    assert ground_timeline(fitted, _grid()).duration_seconds <= 7.0
    assert any(not clip.music_sync.cut_on_beat for clip in fitted.clips)


def test_preferred_rhythm_keeps_a_valid_edit_within_one_bar():
    from montagewright.planner import _fit_preferred_rhythm_to_target

    fitted = _fit_preferred_rhythm_to_target(
        _edl(), _grid(), target_seconds=6.0, duration_mode="preferred",
    )

    # The executable edit is 8s: two seconds over the preference but still
    # inside one 4/4 bar, so there is no reason to knock either cut off-grid.
    assert all(clip.music_sync.cut_on_beat for clip in fitted.clips)


def test_grounding_protects_a_camera_move_floor_off_grid():
    from montagewright.grounding import BeatGrid, Cue, ground_timeline
    from montagewright.schema import Clip, EDL, Look, Reframe

    grid = BeatGrid(
        bpm=120.0,
        meter=4,
        duration_seconds=12.0,
        cues=(
            Cue("b2", 2.0, "beat", 1.0),
            Cue("b35", 3.5, "beat", 1.0),
            Cue("b7", 7.0, "beat", 1.0),
        ),
    )
    moving = Clip(
        clip_id="k00",
        source_id="A",
        approx_in_seconds=0.0,
        approx_out_seconds=3.4,
        audio_role="discard",
        reframe=Reframe(
            camera_move="pan",
            looks=[Look(at="left", seconds=1.0), Look(at="right", seconds=1.0)],
            look_boxes=[(0.25, 0.5, 0.2), (0.75, 0.5, 0.2)],
        ),
    )
    edl = EDL(project_id="camera-floor", clips=[moving])
    after = ground_timeline(edl, grid)

    assert after.clips[0].move_too_short is None
    assert after.clips[0].duration_seconds == 5.0
    assert after.clips[0].landed_on is None
    assert "planned pan needs 5.00s" in (after.clips[0].note or "")


def test_no_music_uses_the_same_camera_completion_floor():
    from montagewright.grounding import ground_timeline
    from montagewright.schema import Clip, EDL, Look, Reframe

    clip = Clip(
        clip_id="k00", source_id="A",
        approx_in_seconds=0.0, approx_out_seconds=1.0,
        reframe=Reframe(
            camera_move="pan",
            looks=[Look(at="left", seconds=1.0), Look(at="right", seconds=1.0)],
            look_boxes=[(0.25, 0.5, 0.2), (0.75, 0.5, 0.2)],
        ),
    )

    grounded = ground_timeline(EDL(project_id="silent", clips=[clip]), None)

    assert grounded.clips[0].duration_seconds == 5.0
    assert grounded.clips[0].move_too_short is None


def test_verified_rhythm_is_reused_without_a_second_provider_call(tmp_path):
    from montagewright.planner import decide_rhythm

    payload = {
        "music_from_seconds": "0:00",
        "decisions": [
            {
                "clip_id": clip_id,
                "cut_on_beat": True,
                "hold_seconds": "0:03",
                "rhythm_reason": "畫面完成後切換",
            }
            for clip_id in ("k00", "k01")
        ],
    }

    class Reply:
        status = "completed"
        usage = {}
        output_text = json.dumps(payload, ensure_ascii=False)

    class Client:
        def __init__(self):
            self.calls = 0

        @property
        def interactions(self):
            return self

        def create(self, **_request):
            self.calls += 1
            return Reply()

    first_client = Client()
    first, _ = decide_rhythm(
        _edl(), _grid(), intent="test", target_seconds=3.0,
        duration_mode="preferred", client=first_client,
        artifact_dir=tmp_path,
    )
    # The first executable 8s result is fed back to Gemini. The second answer
    # receives the bounded local fallback, then is the one persisted.
    assert first_client.calls == 2
    assert (tmp_path / "rhythm.json").exists()

    class MustNotBeCalled:
        @property
        def interactions(self):  # pragma: no cover - cache must return first
            raise AssertionError("provider was called despite a valid artifact")

    remembered, usage = decide_rhythm(
        _edl(), _grid(), intent="test", target_seconds=3.0,
        duration_mode="preferred", client=MustNotBeCalled(),
        artifact_dir=tmp_path,
    )
    assert remembered.model_dump(mode="json") == first.model_dump(mode="json")
    assert usage.input_tokens == usage.output_tokens == usage.thought_tokens == 0


def test_exact_rhythm_is_not_locally_desynchronised():
    from montagewright.planner import _fit_preferred_rhythm_to_target

    fitted = _fit_preferred_rhythm_to_target(
        _edl(), _grid(), target_seconds=6.0, duration_mode="exact",
    )
    assert all(clip.music_sync.cut_on_beat for clip in fitted.clips)
