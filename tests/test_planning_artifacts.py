from montagewright.planning_artifacts import asked, decide, decided


def test_paid_decision_cache_is_keyed_and_atomically_round_trips(tmp_path):
    key = asked("material", "brief", "schema")
    value = {"shots": [{"span_id": "C1:s00"}]}
    assert decided(tmp_path, "selection", key) is None
    assert decide(tmp_path, "selection", key, value) == value
    assert decided(tmp_path, "selection", key) == value
    assert decided(tmp_path, "selection", asked("different")) is None
    assert not list(tmp_path.glob(".*.tmp"))
