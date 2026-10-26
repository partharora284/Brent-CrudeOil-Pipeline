import pandas as pd
import pytest

from src.sessions import assess_session, is_completed_bar, is_market_holiday, is_session_day, next_session, today_in_market

D = pd.Timestamp


def test_weekends_and_fixed_holidays_are_not_sessions():
    assert not is_session_day("2026-10-24") and not is_session_day("2026-10-25")  # Sat, Sun
    assert is_session_day("2026-10-26")
    assert is_market_holiday("2026-12-25") and is_market_holiday("2027-01-01")
    assert is_market_holiday("2026-04-03")  # Good Friday 2026
    assert next_session("2026-10-23") == D("2026-10-26")  # Friday -> Monday
    assert next_session("2026-12-24") == D("2026-12-28")  # skips Christmas Day + weekend


def test_extra_holidays_from_environment(monkeypatch):
    monkeypatch.setenv("MARKET_HOLIDAYS", "2026-10-26, 2026-10-27")
    assert next_session("2026-10-23") == D("2026-10-28")


def test_today_is_computed_in_the_market_timezone():
    # 02:00 UTC on 26 Oct is still 25 Oct (late evening) in New York
    assert today_in_market("2026-10-26 02:00:00") == D("2026-10-25")
    assert today_in_market("2026-10-26 15:00:00") == D("2026-10-26")


def test_only_bars_before_today_are_completed():
    assert is_completed_bar("2026-10-23", today=D("2026-10-26"))
    assert not is_completed_bar("2026-10-26", today=D("2026-10-26"))  # today's bar is still being built
    assert not is_completed_bar("2026-10-27", today=D("2026-10-26"))


def test_current_session_when_latest_data_is_the_previous_session():
    r = assess_session("2026-10-23", today=D("2026-10-26"))  # Fri data, Monday
    assert r["status"] == "current" and r["is_current_session"]
    assert r["session_date"] == "2026-10-26" and r["input_date"] == "2026-10-23" and r["message"] is None
    assert not r["uses_previous_row"]


def test_closed_market_shows_latest_session_with_message():
    r = assess_session("2026-10-23", today=D("2026-10-25"))  # Sunday
    assert r["status"] == "closed" and not r["is_current_session"]
    assert r["session_date"] == "2026-10-23" and r["uses_previous_row"]
    assert "closed" in r["message"].lower() and "Fri 23 Oct 2026" in r["message"]


def test_outdated_data_never_claims_to_be_today():
    r = assess_session("2026-10-12", today=D("2026-10-26"))
    assert r["status"] == "outdated" and not r["is_current_session"]
    assert r["session_date"] == "2026-10-12" and r["uses_previous_row"]
    assert "not been updated" in r["message"]


def test_stale_input_blocks_prediction_and_names_the_input():
    last = {"brent": D("2026-10-23"), "wti": D("2026-10-14"), "dxy": D("2026-10-23")}
    r = assess_session("2026-10-23", today=D("2026-10-26"), asset_last_obs=last)
    assert r["status"] == "insufficient" and r["session_date"] is None
    assert "WTI crude" in r["message"] and r["stale_assets"] == ["wti"]


def test_input_within_allowed_lag_is_fine():
    last = {"brent": D("2026-10-23"), "wti": D("2026-10-20"), "dxy": D("2026-10-23")}  # 3 days
    assert assess_session("2026-10-23", today=D("2026-10-26"), asset_last_obs=last)["status"] == "current"
