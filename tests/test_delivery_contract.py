from montagewright.delivery_contract import (
    editorial_obligation_faults, graphic_obligation_faults, music_policy_faults,
)
from montagewright.graphics import CopyFact, GraphicCue, GraphicsPlan
from montagewright.job import MusicPolicy, TimeRange, TimelineObligation, TimelineWindow


SHOTS = [
    {"seconds_needed": "0:03", "looks": [{"entity_id": "flip"}]},
    {"seconds_needed": "0:02", "looks": [
        {"entity_id": "fold", "co_visible_entity_ids": ["ultra"]},
    ]},
    {"seconds_needed": "0:04", "looks": [{"entity_id": "flip"}]},
]


def test_picture_obligations_measure_the_actual_timeline_windows():
    obligations = (
        TimelineObligation(
            obligation_id="flip-min", kind="minimum_presence", refs=("flip",),
            minimum_seconds=7,
        ),
        TimelineObligation(
            obligation_id="reveal", kind="forbidden_presence", refs=("ultra",),
            window=TimelineWindow(end_seconds=3),
        ),
        TimelineObligation(
            obligation_id="pair", kind="required_cooccurrence",
            refs=("fold", "ultra"),
        ),
    )

    assert editorial_obligation_faults(SHOTS, obligations) == []


def test_obligations_fail_on_presence_not_on_explanatory_prose():
    obligations = (
        TimelineObligation(
            obligation_id="ultra-too-early", kind="forbidden_presence",
            refs=("ultra",), window=TimelineWindow(end_seconds=6),
        ),
        TimelineObligation(
            obligation_id="three-together", kind="required_cooccurrence",
            refs=("flip", "fold", "ultra"),
        ),
    )

    faults = editorial_obligation_faults(SHOTS, obligations)
    assert any("ultra-too-early" in one for one in faults)
    assert any("three-together" in one for one in faults)


def test_sequential_pan_does_not_masquerade_as_simultaneous_group_shot():
    shots = [{"seconds_needed": 3, "looks": [
        {"entity_id": "flip"}, {"entity_id": "fold"},
    ]}]
    rule = TimelineObligation(
        obligation_id="together", kind="required_cooccurrence",
        refs=("flip", "fold"),
    )
    assert editorial_obligation_faults(shots, (rule,))


def test_music_must_stay_inside_the_licensed_source_window():
    policy = MusicPolicy(allowed_ranges=(TimeRange(
        start_seconds=40, end_seconds=70,
    ),))

    assert music_policy_faults({"music_spans": [(42, 60)]}, policy) == []
    assert music_policy_faults({"music_spans": [(35, 60)]}, policy)


def test_graphic_read_time_requires_an_approved_cue_in_the_window():
    obligation = TimelineObligation(
        obligation_id="cta-read", kind="minimum_read", track="graphic",
        refs=("cta",), minimum_seconds=3,
        window=TimelineWindow(final_seconds=5),
    )
    fact = CopyFact(
        fact_id="cta", exact_text="立即預購", source_kind="user",
        approved=True, approved_by="human_review",
    )
    plan = GraphicsPlan(facts=[fact], cues=[GraphicCue(
        graphic_id="end-card", kind="end_card", primary_fact_id="cta",
        at_seconds=7, duration_seconds=3, status="approved",
    )])

    assert graphic_obligation_faults(plan, (obligation,), total=10) == []
    draft = plan.model_copy(update={
        "cues": [plan.cues[0].model_copy(update={"status": "draft"})],
    })
    assert graphic_obligation_faults(draft, (obligation,), total=10)
