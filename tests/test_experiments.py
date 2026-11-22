import pytest

from src.assets import experiments as ex


def _save(st, name="a", sharpe=1.0, **req):
    return st.save("futures_spread", name, {"path": "/api/futures/spread", "query": {"a": "CL", "b": "NG", **req}}, {"summary": {"sharpe": sharpe, "trades": 10}}, {"kind": "synthetic"})


def test_save_get_list_delete_round_trip(tmp_path):
    st = ex.Store(tmp_path)
    r = _save(st)
    assert st.get(r["id"])["metrics"] == {"summary.sharpe": 1.0, "summary.trades": 10.0} and st.list()[0]["id"] == r["id"]
    st.delete(r["id"])
    assert st.list() == []
    with pytest.raises(KeyError):
        st.get(r["id"])


def test_same_request_same_result_reproduces_and_changed_result_is_flagged(tmp_path):
    st = ex.Store(tmp_path)
    a, b, c = _save(st, "a"), _save(st, "b"), _save(st, "c", sharpe=0.7)
    assert a["config_hash"] == b["config_hash"] and a["result_hash"] == b["result_hash"]
    assert any("reproduced" in w for w in st.compare([a["id"], b["id"]])["warnings"])
    assert any("data or the code" in w for w in st.compare([a["id"], c["id"]])["warnings"])


def test_compare_lists_the_settings_that_differ_and_warns_on_mixed_provenance(tmp_path):
    st = ex.Store(tmp_path)
    a = _save(st, "a", lookback=100)
    b = st.save("futures_spread", "b", {"path": "/api/futures/spread", "query": {"a": "CL", "b": "NG", "lookback": 200}}, {"summary": {"sharpe": 2.0}}, {"kind": "observed_delayed"})
    out = st.compare([a["id"], b["id"]])
    assert [d["setting"] for d in out["setting_differences"]] == ["param.query.lookback"] and any("differ in kind" in w for w in out["warnings"])


def test_bad_input_and_path_traversal_are_refused(tmp_path):
    st = ex.Store(tmp_path)
    with pytest.raises(ValueError):
        st.save("nope", "x", {"path": "/api/x"}, {"a": 1})
    with pytest.raises(ValueError):
        st.save("other", "x", {"path": "http://evil"}, {"a": 1})
    with pytest.raises(ValueError):
        st.save("other", "x", {"path": "/api/x"}, {"a": "text"})
    with pytest.raises(ValueError):
        st.get("../../etc/passwd")
    with pytest.raises(ValueError):
        st.compare(["only-one"])
