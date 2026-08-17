import json

from montagewright.planning_artifacts import (
    asked, decide, decided, latest_decision,
)


def test_paid_decision_cache_is_keyed_and_atomically_round_trips(tmp_path):
    key = asked("material", "brief", "schema")
    value = {"shots": [{"span_id": "C1:s00"}]}
    assert decided(tmp_path, "selection", key) is None
    assert decide(tmp_path, "selection", key, value) == value
    assert decided(tmp_path, "selection", key) == value
    assert decided(tmp_path, "selection", asked("different")) is None
    assert not list(tmp_path.glob(".*.tmp"))


def test_paid_decision_cache_rejects_a_value_changed_after_publish(tmp_path):
    key = asked("material", "brief", "schema")
    decide(tmp_path, "selection", key, {"shots": [{"span_id": "C1:s00"}]})
    path = tmp_path / "selection.json"
    saved = json.loads(path.read_text(encoding="utf-8"))
    saved["value"]["shots"][0]["span_id"] = "tampered:s00"
    path.write_text(json.dumps(saved), encoding="utf-8")

    assert decided(tmp_path, "selection", key) is None


def test_legacy_paid_decision_without_checksum_remains_readable(tmp_path):
    """Adding integrity metadata must not itself cause a paid rerun."""

    key = asked("old")
    value = {"shots": [{"span_id": "C1:s00"}]}
    (tmp_path / "selection.json").write_text(
        json.dumps({"key": key, "value": value}), encoding="utf-8"
    )

    assert decided(tmp_path, "selection", key) == value


def test_latest_decision_is_only_a_checksum_valid_migration_input(tmp_path):
    decide(tmp_path, "selection-attempt", "old-key", {"shots": [1]})

    assert decided(tmp_path, "selection-attempt", "new-key") is None
    assert latest_decision(tmp_path, "selection-attempt") == {"shots": [1]}

    path = tmp_path / "selection-attempt.json"
    saved = json.loads(path.read_text(encoding="utf-8"))
    saved["value"]["shots"] = [2]
    path.write_text(json.dumps(saved), encoding="utf-8")
    assert latest_decision(tmp_path, "selection-attempt") is None
