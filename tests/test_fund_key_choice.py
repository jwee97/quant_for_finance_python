"""Choosing which key makes the requests: a button's worth of choice, kept across restarts, never an automatic switch, and refused while the key in use has spent its allowance (OpenFilings' usage policy
forbids rotating accounts or keys to get past a quota). A key is named by its slot (``key 2``) when it is chosen and kept as the salted hash that identifies it, never as the key."""

from __future__ import annotations

import pytest

from src.fundamentals.errors import FundamentalsError, InvalidKey, QuotaExhausted, SyncPending
from tests.test_fund_data import KEY_1, KEY_2, KEY_3, Clock, _pool, _service


def ids(pool):
    return {s["slot"]: s["key_id"] for s in pool.slots()}


def roles(pool):
    return [(s["slot"], s["role"]) for s in pool.slots()]


# ------------------------------------------------------------------------------------------------------------------------ the pool
def test_a_key_the_user_chooses_makes_the_requests_and_the_others_are_standbys(tmp_path):
    pool, usage = _pool(tmp_path, (KEY_1, KEY_2, KEY_3))
    key = ids(pool)
    assert roles(pool) == [("key 1", "active"), ("key 2", "standby"), ("key 3", "standby")] and not any(s["chosen"] for s in pool.slots()) and pool.chosen_id is None
    assert pool.choose("key 2") == "key 2" and pool.chosen_id == key["key 2"]
    assert [(s["slot"], s["role"], s["chosen"]) for s in pool.slots()] == [("key 1", "standby", False), ("key 2", "active", True), ("key 3", "standby", False)]
    used: list[str] = []
    pool.call(lambda k: used.append(k) or 1, "financials", "us:AAA", credits=10)
    assert used == [KEY_2] and usage.used_today("openfilings", key["key 2"]) == 10 and usage.used_today("openfilings", key["key 1"]) == 0
    assert pool.remaining_today() == 90                                                     # what the key in use can still spend
    assert pool.choose("key 1") == "key 1" and roles(pool)[0] == ("key 1", "active")        # and back again: nothing is fixed
    pool.call(lambda k: used.append(k) or 1, "financials", "us:AAA", credits=10)
    assert used == [KEY_2, KEY_1]


def test_a_slot_is_named_by_its_number_or_its_name_in_any_case(tmp_path):
    pool, _ = _pool(tmp_path, (KEY_1, KEY_2, KEY_3))
    assert [pool.choose(text) for text in ("2", " Key 3 ", "KEY 1", 2, "key 3")] == ["key 2", "key 3", "key 1", "key 2", "key 3"]
    assert roles(pool)[2] == ("key 3", "active")


def test_choosing_the_key_already_in_use_changes_nothing_but_keeps_it_there(tmp_path):
    pool, _ = _pool(tmp_path, (KEY_1, KEY_2))
    assert pool.choose("key 1") == "key 1" and [s["chosen"] for s in pool.slots()] == [True, False]
    assert roles(pool) == [("key 1", "active"), ("key 2", "standby")]


def test_the_key_in_use_may_be_chosen_again_even_when_it_has_spent_its_allowance_because_nothing_moves(tmp_path):
    pool, _ = _pool(tmp_path, (KEY_1, KEY_2), daily=10)
    pool.call(lambda k: 1, "financials", "us:AAA", credits=10)
    assert pool.slots()[0]["state"] == "spent_day" and pool.choose("key 1") == "key 1" and [s["chosen"] for s in pool.slots()] == [True, False]
    with pytest.raises(FundamentalsError, match="usage policy forbids"):
        pool.choose("key 2")


def test_nothing_switches_by_itself_when_the_key_in_use_runs_dry_even_after_a_choice(tmp_path):
    pool, _ = _pool(tmp_path, (KEY_1, KEY_2, KEY_3), daily=20)
    pool.choose("key 2")
    used: list[str] = []
    for _ in range(2):
        pool.call(lambda k: used.append(k) or 1, "financials", "us:AAA", credits=10)
    with pytest.raises(QuotaExhausted):
        pool.call(lambda k: used.append(k) or 1, "financials", "us:AAA", credits=10)
    assert used == [KEY_2, KEY_2] and roles(pool) == [("key 1", "standby"), ("key 2", "active"), ("key 3", "standby")]


def test_a_choice_is_refused_while_the_key_in_use_has_spent_its_daily_allowance_and_allowed_once_it_resets(tmp_path):
    clock = Clock()
    pool, _ = _pool(tmp_path, (KEY_1, KEY_2), clock=clock, daily=20)
    for _ in range(2):
        pool.call(lambda k: 1, "financials", "us:AAA", credits=10)
    assert pool.slots()[0]["state"] == "spent_day"
    with pytest.raises(FundamentalsError) as refused:
        pool.choose("key 2")
    text = str(refused.value)
    assert text.startswith("Key 1 is in use and its daily allowance is spent (it resets at 00:00 UTC).") and "usage policy forbids rotating accounts or API keys to get past its quotas" in text
    assert KEY_1 not in text and KEY_2 not in text
    second = pool.slots()[1]
    assert second["can_use"] is False and second["why_not"] == text and pool.slots()[0]["can_use"] is False and pool.slots()[0]["why_not"] == ""
    assert roles(pool) == [("key 1", "active"), ("key 2", "standby")] and pool.chosen_id is None      # the refusal moved nothing
    clock.advance(days=1)                                                                   # the allowance resets
    assert pool.slots()[1]["can_use"] is True and pool.choose("key 2") == "key 2"


def test_a_choice_is_refused_while_the_key_in_use_has_spent_its_monthly_allowance_or_its_balance(tmp_path):
    clock = Clock()
    pool, usage = _pool(tmp_path, (KEY_1, KEY_2), clock=clock, daily=100, monthly=20)
    for _ in range(2):
        pool.call(lambda k: 1, "financials", "us:AAA", credits=10)
    assert pool.slots()[0]["state"] == "ready"                                              # the day has room; the month is used up
    with pytest.raises(FundamentalsError, match=r"Key 1 is in use and its monthly allowance is spent \(it resets on 2026-11-01 00:00 UTC\)"):
        pool.choose("key 2")
    broke, usage2 = _pool(tmp_path, (KEY_1, KEY_2), clock=clock, name="balance.db")
    usage2.record("openfilings", "financials", False, key_id=ids(broke)["key 1"], error="balance: not enough credits")
    assert broke.slots()[0]["state"] == "no_credits"
    with pytest.raises(FundamentalsError, match="its included credits and prepaid balance cannot cover a call"):
        broke.choose("key 2")


def test_a_key_that_is_rejected_or_out_of_credits_itself_cannot_be_chosen(tmp_path):
    pool, usage = _pool(tmp_path, (KEY_1, KEY_2, KEY_3), daily=50)
    key = ids(pool)
    usage.record("openfilings", "financials", True, key_id=key["key 3"], credits=50)       # key 3 spent today's allowance elsewhere
    with pytest.raises(FundamentalsError) as spent:
        pool.choose("key 3")
    assert str(spent.value) == "Key 3 cannot be used now: its daily allowance is spent (it resets at 00:00 UTC)."
    pool.choose("key 2")

    def fn(k):
        if k == KEY_2:
            raise InvalidKey("401")
        return "ok"

    assert pool.call(fn, "search", "X") == "ok"                                             # key 2 was rejected, so the first working key took over
    assert pool.slots()[1]["state"] == "invalid" and roles(pool)[0] == ("key 1", "active")
    with pytest.raises(FundamentalsError) as invalid:
        pool.choose("key 2")
    assert str(invalid.value) == "Key 2 was rejected as invalid or revoked."
    assert pool.slots()[1]["why_not"] == str(invalid.value) and pool.slots()[1]["can_use"] is False


def test_there_is_nothing_to_choose_with_one_key_and_an_unknown_slot_is_refused(tmp_path):
    one, _ = _pool(tmp_path, (KEY_1,))
    with pytest.raises(FundamentalsError, match="there is only one key, so there is nothing to choose"):
        one.choose("key 1")
    assert one.slots()[0]["can_use"] is False and one.slots()[0]["why_not"] == ""

    def rejected(k):
        raise InvalidKey("401")

    with pytest.raises(InvalidKey):
        one.call(rejected, "search", "X")
    assert (one.slots()[0]["role"], one.slots()[0]["can_use"], one.slots()[0]["why_not"]) == ("unusable", False, "")                      # a lone key that was rejected has no other to be swapped for
    two, _ = _pool(tmp_path, (KEY_1, KEY_2), name="two.db")
    for wrong in ("", "nonsense", KEY_2, "key 9", "0", "key", "\u00b2", "9" * 5000):                 # a superscript two and a number of five thousand digits are no slot and no crash
        with pytest.raises(FundamentalsError) as bad:
            two.choose(wrong)
        assert str(bad.value) == "There is no such key."                                    # a key typed in place of a slot is not echoed back, only refused
    assert roles(two) == [("key 1", "active"), ("key 2", "standby")]


def test_a_chosen_key_that_is_rejected_hands_over_to_the_first_working_key_in_the_list(tmp_path):
    pool, _ = _pool(tmp_path, (KEY_1, KEY_2, KEY_3))
    pool.choose("key 3")

    def fn(k):
        if k == KEY_3:
            raise InvalidKey("401")
        return k

    assert pool.call(fn, "search", "X") == KEY_1 and roles(pool) == [("key 1", "active"), ("key 2", "standby"), ("key 3", "unusable")]


def test_an_answer_that_is_still_being_collected_is_logged_as_a_request_that_did_not_succeed(tmp_path):
    pool, usage = _pool(tmp_path, (KEY_1, KEY_2))

    def pending(k):
        raise SyncPending("still collecting", retry_after=60)

    with pytest.raises(SyncPending):
        pool.call(pending, "financials", "jp:7203", credits=10)
    row = usage.recent("openfilings", 1)[0]
    assert (row["success"], row["credits_used"], row["error"]) == (0, 0, "pending: still collecting") and usage.used_today("openfilings") == 0         # nothing was charged


def test_a_saved_choice_that_matches_no_key_is_ignored(tmp_path):
    pool, _ = _pool(tmp_path, (KEY_1, KEY_2))
    pool.prefer("0123456789abcdef")
    assert roles(pool) == [("key 1", "active"), ("key 2", "standby")] and not any(s["chosen"] for s in pool.slots()) and pool.chosen_id is None
    pool.prefer(ids(pool)["key 2"])
    assert roles(pool) == [("key 1", "standby"), ("key 2", "active")]
    pool.prefer(None)
    assert roles(pool)[0] == ("key 1", "active")


# ------------------------------------------------------------------------------------------------------------------------ the service keeps the choice
def test_the_choice_is_kept_across_a_restart_as_a_hash_and_never_as_the_key(tmp_path):
    svc, _ = _service(tmp_path, keys=(KEY_1, KEY_2))
    key = ids(svc.pool("openfilings"))
    answer = svc.use_key("openfilings", "key 2")
    assert answer["provider"] == "openfilings" and answer["active"] == "key 2" and [s["role"] for s in answer["keys"]] == ["standby", "active"]
    assert svc.store.get_meta("key_choice:openfilings") == key["key 2"] and len(key["key 2"]) == 16
    again, _ = _service(tmp_path, keys=(KEY_1, KEY_2))                                        # a restart: a new service on the same files
    assert roles(again.pool("openfilings")) == [("key 1", "standby"), ("key 2", "active")]
    for suffix in ("", "-wal", "-shm"):
        path = tmp_path / "svc" / "data" / "user" / f"fundamentals.db{suffix}"
        if path.exists():
            blob = path.read_bytes()
            assert not any(k.encode() in blob for k in (KEY_1, KEY_2, KEY_3))                 # no key is in the database, only the hash of the one chosen


def test_a_replaced_key_loses_the_choice_that_was_made_for_it(tmp_path):
    svc, _ = _service(tmp_path, keys=(KEY_1, KEY_2))
    svc.use_key("openfilings", "key 2")
    swapped, _ = _service(tmp_path, keys=(KEY_1, KEY_3))                                      # the second secret now holds another key
    pool = swapped.pool("openfilings")
    assert roles(pool) == [("key 1", "active"), ("key 2", "standby")] and not any(s["chosen"] for s in pool.slots())


def test_the_service_refuses_a_choice_it_could_not_keep_and_remembers_nothing_then(tmp_path):
    svc, _ = _service(tmp_path, keys=(KEY_1, KEY_2), clock=Clock(), daily=10)
    svc.pool("openfilings").call(lambda k: 1, "financials", "us:AAA", credits=10)
    with pytest.raises(FundamentalsError, match="usage policy forbids rotating"):
        svc.use_key("openfilings", "key 2")
    assert svc.store.get_meta("key_choice:openfilings") is None
    with pytest.raises(FundamentalsError, match="only one key"):
        svc.use_key("simulated", "key 1")                                                      # the simulated world has no keys to choose between
