from pathlib import Path
from types import SimpleNamespace


def _spec():
    return SimpleNamespace(identity_lock=SimpleNamespace(
        framing=SimpleNamespace(required_target_ids=("target.one",)),
        identity=SimpleNamespace(targets=()),
    ))


def test_commitment_pool_sources_are_prepared_then_judged_in_one_cross_call(
    monkeypatch, tmp_path: Path,
):
    from montagewright import reference_grounding as grounding
    from montagewright.cli import _confirm_material_identity

    items = [
        SimpleNamespace(source_id=f"C{index}", carries_identity=True, duration_seconds=4.0)
        for index in range(7)
    ]
    masters = {}
    for item in items:
        source = tmp_path / f"{item.source_id}.mp4"
        source.write_bytes(item.source_id.encode())
        masters[item.source_id] = source

    monkeypatch.setattr(
        grounding, "source_confirmation_cache_path",
        lambda source, spec, target, library: tmp_path / f"{Path(source).stem}.json",
    )
    monkeypatch.setattr(
        grounding, "read_source_confirmation_cache", lambda path, target: None
    )
    prepared = []

    def prepare(source, spec, discovery, target, frames_dir, at_ms=()):
        seed = SimpleNamespace(item=SimpleNamespace(source=Path(source).stem))
        prepared.append(seed)
        return seed

    monkeypatch.setattr(grounding, "prepare_source_identity_seed", prepare)
    calls = []

    def decide(spec, target, requests, **kwargs):
        calls.append(tuple(requests))
        outcomes = tuple(
            SimpleNamespace(
                    evaluation=SimpleNamespace(
                        lineage=SimpleNamespace(video_sha256="a" * 64),
                        decision=SimpleNamespace(verdict="matched_target"),
                )
            )
            for _ in requests
        )
        return SimpleNamespace(outcomes=outcomes), SimpleNamespace()

    monkeypatch.setattr(grounding, "decide_cross_asset_exact_frame_bboxes", decide)
    monkeypatch.setattr(
        grounding, "confirmed_frame_from_validated_seed",
        lambda seed, evaluation: SimpleNamespace(seed_risk_flags=()),
    )
    written = []
    monkeypatch.setattr(
        grounding, "write_source_confirmation_cache",
        lambda path, target, digest, found: written.append(path),
    )
    singleton = []
    monkeypatch.setattr(
        grounding, "confirm_source_identity",
        lambda *args, **kwargs: singleton.append(args) or (),
    )
    ledger = SimpleNamespace(spent_usd=0.0, check=lambda: None)

    found = _confirm_material_identity(
        items, {item.source_id: object() for item in items}, _spec(),
        masters=masters, client=object(), cache=object(), ledger=ledger,
        library=tmp_path / "library", work=tmp_path / "work",
    )

    assert len(calls) == 1 and len(calls[0]) == 7
    assert len(found) == 7 and len(written) == 7
    assert singleton == []


def test_risky_cross_source_seed_alone_uses_the_multi_anchor_fallback(
    monkeypatch, tmp_path: Path,
):
    from montagewright import reference_grounding as grounding
    from montagewright.cli import _confirm_material_identity

    item = SimpleNamespace(source_id="C1", carries_identity=True, duration_seconds=4.0)
    source = tmp_path / "C1.mp4"
    source.write_bytes(b"video")
    monkeypatch.setattr(
        grounding, "source_confirmation_cache_path",
        lambda *args: tmp_path / "identity.json",
    )
    monkeypatch.setattr(
        grounding, "read_source_confirmation_cache", lambda *args: None
    )
    seed = SimpleNamespace(item=object())
    monkeypatch.setattr(
        grounding, "prepare_source_identity_seed", lambda *args, **kwargs: seed
    )
    evaluation = SimpleNamespace(
        lineage=SimpleNamespace(video_sha256="a" * 64),
        decision=SimpleNamespace(verdict="matched_target"),
    )
    monkeypatch.setattr(
        grounding, "decide_cross_asset_exact_frame_bboxes",
        lambda *args, **kwargs: (
            SimpleNamespace(outcomes=(SimpleNamespace(evaluation=evaluation),)),
            SimpleNamespace(),
        ),
    )
    monkeypatch.setattr(
        grounding, "confirmed_frame_from_validated_seed",
        lambda *args: SimpleNamespace(seed_risk_flags=("occlusion_major",)),
    )
    fallback = SimpleNamespace(seed_risk_flags=())
    calls = []
    monkeypatch.setattr(
        grounding, "confirm_source_identity",
        lambda *args, **kwargs: calls.append(args) or (fallback,),
    )
    monkeypatch.setattr(
        grounding, "write_source_confirmation_cache", lambda *args: None
    )

    found = _confirm_material_identity(
        [item], {"C1": object()}, _spec(), masters={"C1": source},
        client=object(), cache=object(),
        ledger=SimpleNamespace(spent_usd=0.0, check=lambda: None),
        library=tmp_path / "library", work=tmp_path / "work",
    )

    assert found["C1"] == (fallback,)
    assert len(calls) == 1


def test_valid_semantic_negative_is_cached_without_retrying_for_a_match(
    monkeypatch, tmp_path: Path,
):
    from montagewright import reference_grounding as grounding
    from montagewright.cli import _confirm_material_identity

    item = SimpleNamespace(source_id="C1", carries_identity=True, duration_seconds=4.0)
    source = tmp_path / "C1.mp4"
    source.write_bytes(b"video")
    monkeypatch.setattr(
        grounding, "source_confirmation_cache_path", lambda *args: tmp_path / "id.json"
    )
    monkeypatch.setattr(grounding, "read_source_confirmation_cache", lambda *args: None)
    monkeypatch.setattr(
        grounding, "prepare_source_identity_seed",
        lambda *args, **kwargs: SimpleNamespace(item=object()),
    )
    evaluation = SimpleNamespace(
        lineage=SimpleNamespace(video_sha256="a" * 64),
        decision=SimpleNamespace(verdict="hard_negative"),
    )
    monkeypatch.setattr(
        grounding, "decide_cross_asset_exact_frame_bboxes",
        lambda *args, **kwargs: (
            SimpleNamespace(outcomes=(SimpleNamespace(evaluation=evaluation),)),
            SimpleNamespace(),
        ),
    )
    monkeypatch.setattr(
        grounding, "confirmed_frame_from_validated_seed",
        lambda *args: (_ for _ in ()).throw(ValueError("not matched")),
    )
    written = []
    monkeypatch.setattr(
        grounding, "write_source_confirmation_cache",
        lambda path, target, digest, found: written.append(tuple(found)),
    )
    retried = []
    monkeypatch.setattr(
        grounding, "confirm_source_identity",
        lambda *args, **kwargs: retried.append(True) or (),
    )

    found = _confirm_material_identity(
        [item], {"C1": object()}, _spec(), masters={"C1": source},
        client=object(), cache=object(),
        ledger=SimpleNamespace(spent_usd=0.0, check=lambda: None),
        library=tmp_path / "library", work=tmp_path / "work",
    )
    assert found == {}
    assert written == [()]
    assert retried == []


def test_eager_identity_confirmation_is_after_commitments_not_after_screen():
    import inspect
    from montagewright import cli

    source = inspect.getsource(cli.command_render)
    screen = source.index("_screen_material_identity(")
    commitments = source.index("commitments = bind_commitments(direction)")
    exact = source.index("confirmed_identities.update(_confirm_material_identity(")
    selection = source.index("selection, usage_selection = select_shots(")
    assert screen < commitments < exact < selection
