from types import SimpleNamespace

from montagewright.planner import Usage


def _sample(at_ms, box, *, state="tracked", **updates):
    values = {
        "analysis_sample_time_ms": at_ms,
        "tracking_state": state,
        "semantic_identity_status": "not_revalidated",
        "derived_tracking_box": box,
        "shot_boundary": False,
        "connected_components": 1,
        "mean_positive_probability": 0.9,
    }
    values.update(updates)
    return SimpleNamespace(**values)


def _one_seed_track(samples):
    return SimpleNamespace(samples=samples, analysis_fps=4.0)


def _observe(samples):
    from montagewright.reframe import observations_from_sam

    return observations_from_sam(
        _one_seed_track(samples),
        clip_start_seconds=0.0,
        semantic_anchors=((0.0, (0.1, 0.2, 0.4, 0.7)),),
        require_identity_validation=True,
    )


def test_clean_single_seed_track_is_accepted_by_continuity():
    observations, states = _observe([
        _sample(0, [100, 200, 400, 700]),
        _sample(250, [110, 200, 410, 700]),
        _sample(500, [120, 200, 420, 700]),
        _sample(750, [130, 200, 430, 700]),
    ])

    assert len(observations) == 4
    assert states["_identity_by_continuity"] == 1
    assert not any(name.startswith("_continuity_risk:") for name in states)


def test_single_seed_does_not_cross_a_missing_or_lost_gap():
    for gap in (
        _sample(250, None, state="lost"),
        _sample(250, None, state="tracked"),
    ):
        observations, states = _observe([
            _sample(0, [100, 200, 400, 700]),
            gap,
            _sample(500, [600, 200, 900, 700]),
            _sample(750, [610, 200, 910, 700]),
        ])

        assert observations == []
        assert states["identity_unverified"] == 4
        assert states["_continuity_risk:missing_mask"] == 1


def test_single_seed_rejects_boxed_low_confidence_and_shot_boundary():
    for risky in (
        _sample(250, [110, 200, 410, 700], state="low_confidence"),
        _sample(250, [110, 200, 410, 700], shot_boundary=True),
    ):
        observations, states = _observe([
            _sample(0, [100, 200, 400, 700]),
            risky,
            _sample(500, [120, 200, 420, 700]),
        ])

        assert observations == []
        assert states["identity_unverified"] == 3
        assert any(name.startswith("_continuity_risk:") for name in states)


def test_single_seed_eligibility_requires_exact_lineage_short_span_and_no_risk():
    from montagewright.pipeline import _single_seed_eligibility

    clean = SimpleNamespace(
        at_seconds=2.0,
        seed_risk_flags=(),
        video_asset_id="sha256:" + "a" * 64,
        frame_time_ms=2000,
        width=1440,
        height=810,
    )
    assert _single_seed_eligibility(clean, 1.0, 4.0) == (True, ())

    old_cache = SimpleNamespace(at_seconds=2.0, seed_risk_flags=())
    assert _single_seed_eligibility(old_cache, 1.0, 4.0) == (
        False, ("exact_lineage_unavailable",)
    )

    occluded = SimpleNamespace(**{
        **clean.__dict__, "seed_risk_flags": ("occlusion_major",),
    })
    assert _single_seed_eligibility(occluded, 1.0, 4.0) == (
        False, ("occlusion_major",)
    )

    eligible, risks = _single_seed_eligibility(clean, 0.0, 8.0)
    assert not eligible and "unchecked_span_too_long" in risks


def test_old_confirmed_frame_cache_remains_loadable_but_is_not_single_seed():
    from montagewright.reference_grounding import ConfirmedFrame

    old = ConfirmedFrame.model_validate({
        "at_seconds": 2.0,
        "box": [0.1, 0.2, 0.4, 0.7],
        "sighting": "c1",
        "sighting_window": [0.0, 5.0],
        "frame_pts": 2000,
        "frame_sha256": "b" * 64,
    })

    assert old.video_asset_id is None
    assert old.seed_risk_flags == ()


def test_exact_seed_risks_include_lookalikes_occlusion_and_edges():
    from montagewright.reference_grounding import exact_seed_risk_flags

    decision = SimpleNamespace(
        excluded_instances=(object(),),
        visibility_state="occluded",
        occlusion_state="major",
        touches_frame_edges=("right",),
    )
    assert exact_seed_risk_flags(decision) == (
        "excluded_instance_in_seed_frame",
        "visibility_occluded",
        "occlusion_major",
        "target_touches_frame_edge",
    )


def _source_confirmation_fixture(monkeypatch, tmp_path, *, risky_seed=False):
    from montagewright import reference_grounding as grounding

    video = tmp_path / "take.mp4"
    video.write_bytes(b"immutable-video")
    digest = grounding.sha256_file(video)
    candidate = SimpleNamespace(
        candidate_id="c1",
        target_id="target.one",
        start_ms=0,
        end_ms=4000,
        recommended_seed_ms=2000,
        identity_status="matched_target",
    )
    discovery = SimpleNamespace(candidates=(candidate,))
    spec = SimpleNamespace(
        definition_sha256=lambda: "d" * 64,
        identity_lock=SimpleNamespace(
            query_id="grounding:target.one",
            definition_sha256=lambda: "e" * 64,
        ),
    )
    lineage_video = SimpleNamespace(
        asset_id=f"sha256:{digest}",
        content_sha256=digest,
        duration_ms=4000,
    )
    monkeypatch.setattr(grounding, "inspect_video_lineage", lambda path: lineage_video)

    materials = []

    def materialize(path, at, destination, max_width=None):
        lineage = SimpleNamespace(
            video_asset_id=f"sha256:{digest}",
            video_sha256=digest,
            frame_pts=at,
            frame_time_ms=at,
            frame_sha256=f"{at:064x}"[-64:],
            width=1440,
            height=810,
        )
        material = SimpleNamespace(lineage=lineage)
        materials.append(material)
        return material

    monkeypatch.setattr(grounding, "materialize_frame_at_time", materialize)
    calls = {"single": 0, "batch": 0}

    def decision_for(material, *, risky):
        return SimpleNamespace(
            verdict="matched_target",
            tracking_box_xyxy_1000=(100, 200, 500, 800),
            excluded_instances=((object(),) if risky else ()),
            visibility_state=("occluded" if risky else "full"),
            occlusion_state=("major" if risky else "none"),
            touches_frame_edges=(),
        )

    def decide_one(*args, **kwargs):
        calls["single"] += 1
        return decision_for(materials[0], risky=risky_seed), Usage(1, 1, 0)

    def decide_batch(*args, **kwargs):
        calls["batch"] += 1
        evaluations = tuple(
            SimpleNamespace(
                lineage=material.lineage,
                decision=decision_for(material, risky=False),
            )
            for material in materials[:2]
        )
        return SimpleNamespace(
            evaluations=evaluations,
            sam_seed_evaluations=lambda: evaluations,
        ), Usage(2, 1, 0)

    monkeypatch.setattr(grounding, "decide_exact_frame_bbox", decide_one)
    monkeypatch.setattr(grounding, "decide_exact_frame_bboxes", decide_batch)
    found = grounding.confirm_source_identity(
        video, spec, discovery, "target.one",
        client=object(), frames_dir=tmp_path / "frames",
    )
    return found, calls


def test_clean_source_confirmation_pays_for_one_exact_frame(monkeypatch, tmp_path):
    found, calls = _source_confirmation_fixture(monkeypatch, tmp_path)

    assert len(found) == 1
    assert calls == {"single": 1, "batch": 0}
    assert found[0].video_asset_id is not None
    assert found[0].seed_risk_flags == ()


def test_risky_single_seed_falls_back_to_multi_anchor(monkeypatch, tmp_path):
    found, calls = _source_confirmation_fixture(
        monkeypatch, tmp_path, risky_seed=True
    )

    assert len(found) == 2
    assert calls == {"single": 1, "batch": 1}
