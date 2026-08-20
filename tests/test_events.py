"""The per-clip event table: collect measured moments, resolve names, snap.

Offline only. Proves the table gathers what the detectors already produced,
turns a stable id back into a source clock, and -- the point that matters --
delegates snapping to the opinionated policies rather than flattening them to
nearest-distance.
"""

from __future__ import annotations

from types import SimpleNamespace

from montagewright.events import Event, EventTable, tables_by_clip


def _manifest():
    # Two scene cuts on the clip's own clock; frame_time_ms is clip-local.
    boundaries = [
        SimpleNamespace(boundary_id="boundary-0001", frame_pts=90000,
                        frame_time_ms=1000, score=7.5),
        SimpleNamespace(boundary_id="boundary-0002", frame_pts=271000,
                        frame_time_ms=3017, score=9.2),
    ]
    return SimpleNamespace(boundaries=boundaries)


def _words():
    return [
        SimpleNamespace(starts_seconds=2.90, ends_seconds=3.14, text="price"),
        SimpleNamespace(starts_seconds=3.20, ends_seconds=3.55, text="today"),
    ]


def test_collects_cuts_with_clip_local_seconds_and_pts():
    table = EventTable("C3").add_cuts(_manifest())
    cuts = table.of_kind("cut")
    assert [round(c.source_seconds, 3) for c in cuts] == [1.0, 3.017]
    # PTS is preserved so a consumer need not re-derive the frame from seconds.
    assert cuts[1].source_pts == 271000
    assert isinstance(cuts[0], Event)


def test_resolve_turns_a_named_moment_back_into_a_source_clock():
    table = EventTable("C3").add_cuts(_manifest()).add_words(_words())
    # The id scheme is stable and per-kind indexed.
    assert table.resolve("cut:C3:001") == 3.017
    assert round(table.resolve("word_start:C3:000"), 2) == 2.90
    # A name this clip does not carry resolves to None (caller falls back).
    assert table.resolve("cut:C3:999") is None


def test_a_silent_clip_still_has_visual_anchors():
    # No words, no music: the floor (scene cuts) is still there. No clip empties.
    table = EventTable("Broll").add_cuts(_manifest())
    assert table.seconds_of("word_start") == []
    assert table.seconds_of("cut") == [1.0, 3.017]


def test_in_point_snaps_to_nearest_cut_within_tolerance():
    table = EventTable("C3").add_cuts(_manifest())
    # 3.1 pulls to the real cut at 3.017; far-away stays put.
    assert table.snap(3.1, purpose="in_point") == 3.017
    assert table.snap(8.0, purpose="in_point", within=0.6) == 8.0


def test_out_point_lands_after_the_pause_not_before():
    # snap_end must never move the cut earlier and eat the last word.
    table = EventTable("C3").add_silence_edges([3.60, 4.20])
    # Asking for 3.5: the safe edge after it is 3.60, within tolerance.
    assert table.snap(3.5, purpose="out_point") == 3.60
    # Asking after every edge: nothing to move onto, keep the second.
    assert table.snap(5.0, purpose="out_point") == 5.0


def test_music_cut_delegates_to_the_grid_priority_not_distance():
    # A fake grid whose nearest_cue prefers the downbeat half a beat FURTHER
    # than an ordinary beat -- the table must return the grid's choice, proving
    # it defers to policy instead of taking min-distance itself.
    class _Grid:
        def nearest_cue(self, seconds):
            # ordinary beat at 3.0 (closer), downbeat at 3.5 (policy winner)
            return SimpleNamespace(time_seconds=3.5)

    table = EventTable("C3")
    assert table.snap(3.05, purpose="music_cut", grid=_Grid()) == 3.5
    # No grid: nothing to snap to, keep the second.
    assert table.snap(3.05, purpose="music_cut") == 3.05


def test_tables_indexed_by_clip():
    a, b = EventTable("C1"), EventTable("C2")
    index = tables_by_clip([a, b])
    assert set(index) == {"C1", "C2"} and index["C2"] is b


def test_builder_assembles_from_real_artifact_shapes():
    # Prove the builder reads the shapes a run actually writes -- the scdet
    # manifest's ShotBoundary fields, and the transcript card's utterances[]/
    # silences[] as the real words_of / detector_silences readers parse them.
    from montagewright.events import build_event_table
    from montagewright.measure.shots import ShotBoundary

    manifest = SimpleNamespace(boundaries=[
        ShotBoundary(boundary_id="boundary-0001", frame_pts=90000,
                     frame_time_ms=1000, score=7.5),
        ShotBoundary(boundary_id="boundary-0002", frame_pts=271000,
                     frame_time_ms=3017, score=9.2),
    ])
    payload = {
        "utterances": [
            {"words": [
                {"starts_seconds": 2.90, "ends_seconds": 3.14,
                 "text": "price", "confidence": 0.9},
                {"starts_seconds": 3.20, "ends_seconds": 3.55, "text": "today"},
            ]},
        ],
        # The SpeechDetector's measured pause; its far edge is the safe out-point.
        "silences": [{"starts_seconds": 3.60, "ends_seconds": 4.20}],
    }

    table = build_event_table("C3", shots=manifest, transcript_payload=payload)
    assert table.seconds_of("cut") == [1.0, 3.017]
    assert [round(s, 2) for s in table.seconds_of("word_start")] == [2.90, 3.20]
    # The silence edge came from the VAD, not from re-deriving word gaps.
    assert table.seconds_of("silence_edge") == [4.20]
    # And it resolves as a named moment for an event_ref contract.
    assert table.resolve("cut:C3:001") == 3.017


def test_builder_tolerates_a_clip_with_no_transcript():
    from montagewright.events import build_event_table
    from montagewright.measure.shots import ShotBoundary

    manifest = SimpleNamespace(boundaries=[
        ShotBoundary(boundary_id="boundary-0001", frame_pts=0,
                     frame_time_ms=500, score=6.0),
    ])
    table = build_event_table("Broll", shots=manifest, transcript_payload=None)
    assert table.seconds_of("cut") == [0.5]
    assert table.seconds_of("word_start", "silence_edge") == []
