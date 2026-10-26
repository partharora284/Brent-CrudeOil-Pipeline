"""
src/sessions.py  --  Which trading session is being predicted, and is the data good enough?
=========================================================================================

WHAT THE PREDICTION MEANS
-------------------------
The model is trained on DAILY data.  For a session S it answers one question:

    "Will Brent's settlement on session S be at or above its settlement on the
     previous session?"   (1 = Up/Flat, 0 = Down)

It is generated BEFORE session S settles, using only information up to the close of the
last COMPLETED session.  It does NOT use intraday prices and it does NOT predict the
final close of a half-finished session from partial-session data.

THE FOUR STATES
---------------
``current``       Today is a trading day and our newest completed session is the previous one.
                  -> the prediction is for TODAY's session.
``closed``        Markets are closed today (weekend/holiday).  -> show the latest completed
                  session (its prediction was made from the session before it) and say so.
``outdated``      At least one completed session is missing from our data (e.g. the data
                  provider failed).  -> a prediction for today is NOT possible; show the
                  latest available session and say so.
``insufficient``  A required input (WTI, DXY) has not updated recently, so the
                  newest feature rows would be built on a stale/forward-filled value.
                  -> no prediction at all, just a message.

A bar is only "completed" if its date is strictly BEFORE today (market time zone), so a
half-finished daily bar from a data provider can never be used as a finished session.

Holidays: weekends plus 25 Dec, 1 Jan and Good Friday are built in; add others with the
``MARKET_HOLIDAYS=2026-12-28,2027-05-31`` environment variable.  This is an approximation
of the ICE calendar, not an official exchange calendar.
"""

from __future__ import annotations

import os
from datetime import timedelta
from typing import Dict, Optional

import pandas as pd
from dateutil.easter import easter

MARKET_TZ = os.getenv("MARKET_TZ", "America/New_York")
MAX_ASSET_LAG_DAYS = int(os.getenv("MAX_ASSET_LAG_DAYS", "3"))  # secondary input may trail Brent by this many calendar days

ASSET_NAMES = {"brent": "Brent", "wti": "WTI crude", "dxy": "US Dollar Index"}


def _fmt(d) -> str:
    return pd.Timestamp(d).strftime("%a %d %b %Y")


def _extra_holidays() -> set:
    return {x.strip() for x in os.getenv("MARKET_HOLIDAYS", "").split(",") if x.strip()}


def today_in_market(now=None) -> pd.Timestamp:
    """Today's calendar date (midnight, tz-naive) in the market time zone."""
    ts = pd.Timestamp(now) if now is not None else pd.Timestamp.now(tz="UTC")
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    return ts.tz_convert(MARKET_TZ).tz_localize(None).normalize()


def is_market_holiday(d) -> bool:
    d = pd.Timestamp(d).normalize()
    if (d.month, d.day) in {(12, 25), (1, 1)}:
        return True
    if d.date() == easter(d.year) - timedelta(days=2):  # Good Friday
        return True
    return d.strftime("%Y-%m-%d") in _extra_holidays()


def is_session_day(d) -> bool:
    d = pd.Timestamp(d).normalize()
    return d.weekday() < 5 and not is_market_holiday(d)


def next_session(d) -> pd.Timestamp:
    """First trading day strictly after ``d``."""
    d = pd.Timestamp(d).normalize() + pd.Timedelta(days=1)
    while not is_session_day(d):
        d += pd.Timedelta(days=1)
    return d


def is_completed_bar(bar_date, today: Optional[pd.Timestamp] = None) -> bool:
    """A daily bar counts as a finished session only if it is dated before today (market time)."""
    today = today_in_market() if today is None else pd.Timestamp(today).normalize()
    return pd.Timestamp(bar_date).normalize() < today


def assess_session(
    latest_date,
    today: Optional[pd.Timestamp] = None,
    asset_last_obs: Optional[Dict[str, pd.Timestamp]] = None,
) -> Dict[str, object]:
    """
    Decide what the dashboard may claim, given the newest completed session ``latest_date``.

    Returns a dict with:
      status              current | closed | outdated | insufficient
      session_date        the session the displayed prediction is about (None if insufficient)
      input_date          last completed session whose data the model uses (None if insufficient)
      uses_previous_row   True when the displayed session is ``latest_date`` itself (closed/outdated),
                          so its prediction must come from the feature row BEFORE ``latest_date``
      is_current_session  True only for ``current``
      message             plain-English note for the user (None when everything is normal)
    """
    today = today_in_market() if today is None else pd.Timestamp(today).normalize()
    latest = pd.Timestamp(latest_date).normalize()
    nxt = next_session(latest)

    stale = []
    for asset, last in (asset_last_obs or {}).items():
        if asset == "brent" or last is None:
            continue
        if (latest - pd.Timestamp(last).normalize()).days > MAX_ASSET_LAG_DAYS:
            stale.append((asset, pd.Timestamp(last)))
    base = {"latest_data_date": str(latest.date()), "next_session": str(nxt.date()), "stale_assets": [a for a, _ in stale]}

    if stale:
        parts = ", ".join(f"{ASSET_NAMES.get(a, a)} (last update {_fmt(d)})" for a, d in stale)
        return {**base, "status": "insufficient", "session_date": None, "input_date": None,
                "uses_previous_row": False, "is_current_session": False,
                "message": f"There isn't enough up-to-date market data for a reliable prediction right now: {parts}."}

    if today == nxt:
        return {**base, "status": "current", "session_date": str(nxt.date()), "input_date": str(latest.date()),
                "uses_previous_row": False, "is_current_session": True, "message": None}

    shown = {"session_date": str(latest.date()), "uses_previous_row": True, "is_current_session": False}
    if today < nxt:
        return {**base, **shown, "status": "closed", "input_date": None,
                "message": f"Markets are closed today ({today:%A}). Showing the latest completed session, "
                           f"{_fmt(latest)}. The next session is {_fmt(nxt)}."}
    return {**base, **shown, "status": "outdated", "input_date": None,
            "message": f"Market data has not been updated since {_fmt(latest)}, so a prediction for today isn't "
                       f"available. Showing the latest available session, {_fmt(latest)}, instead."}
