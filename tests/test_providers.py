import numpy as np
import pandas as pd
import pytest

from src import providers
from src.ingestion import ASSET_FILES, build_master_dataframe, extend_with_provider
from src.providers import (ProviderUnavailable, overlap_difference, parse_yahoo_chart, refresh_yahoo,
                           validate_provider_frame)
from tests.conftest import make_master, write_vendor_csvs

D = pd.Timestamp
TODAY = D("2026-10-26")


def chart(dates, close, high=None, low=None, tz="America/New_York"):
    ts = [int(pd.Timestamp(d, tz=tz).tz_convert("UTC").timestamp()) for d in dates]
    return {"chart": {"error": None, "result": [{
        "meta": {"exchangeTimezoneName": tz}, "timestamp": ts,
        "indicators": {"quote": [{"open": close, "high": high or [None if c is None else c + 1 for c in close],
                                  "low": low or [None if c is None else c - 1 for c in close], "close": close,
                                  "volume": [1] * len(ts)}]}}]}}


def frame(dates, close, **kw):
    return parse_yahoo_chart(chart(dates, close, **kw))


def test_parse_uses_exchange_date_and_skips_null_closes():
    df = frame(["2026-10-21", "2026-10-22", "2026-10-23"], [80.0, None, 82.0])
    assert list(df["date"].dt.strftime("%Y-%m-%d")) == ["2026-10-21", "2026-10-22", "2026-10-23"]
    assert df["price"].isna().tolist() == [False, True, False]
    assert parse_yahoo_chart({"chart": {"result": None, "error": None}}).empty


def test_parse_raises_on_provider_error():
    with pytest.raises(ProviderUnavailable):
        parse_yahoo_chart({"chart": {"error": {"description": "No data found"}, "result": None}})


def test_validate_drops_todays_unfinished_bar_missing_values_and_duplicates():
    df = frame(["2026-10-22", "2026-10-23", "2026-10-26"], [80.0, 81.0, 82.0])  # 26th = today, unfinished
    df = pd.concat([df, df.iloc[[0]]], ignore_index=True)  # duplicate date
    df.loc[1, "price"] = np.nan
    clean, issues = validate_provider_frame(df, "brent", today=TODAY)
    assert list(clean["date"].dt.strftime("%Y-%m-%d")) == ["2026-10-22"]
    text = " ".join(issues)
    assert "today/future" in text and "missing" in text and "duplicate" in text


def test_validate_rejects_bad_ticks_inverted_range_and_brent_without_high_low():
    df = frame(["2026-10-20", "2026-10-21", "2026-10-22", "2026-10-23"], [80.0, 80.5, 160.0, 81.0])
    df.loc[3, "high"], df.loc[3, "low"] = 80.0, 82.0  # High < Low
    clean, issues = validate_provider_frame(df, "wti", today=TODAY)
    assert list(clean["price"]) == [80.0, 80.5]
    df2 = frame(["2026-10-22"], [80.0])
    df2["high"] = np.nan
    assert validate_provider_frame(df2, "brent", today=TODAY)[0].empty          # Brent needs High/Low
    assert len(validate_provider_frame(df2, "wti", today=TODAY)[0]) == 1      # others do not


def test_overlap_difference():
    csv = pd.DataFrame({"brent_price": [80.0, 81.0, 82.0, 83.0]}, index=pd.to_datetime(["2026-10-19", "2026-10-20", "2026-10-21", "2026-10-22"]))
    same = pd.DataFrame({"date": csv.index, "price": csv["brent_price"].to_numpy() * 1.001})
    assert overlap_difference(csv, same, "brent") == pytest.approx(0.001)
    assert overlap_difference(csv, same.iloc[:2], "brent") is None  # < 3 common dates -> cannot verify


def test_master_extends_csv_with_provider_rows_only_after_csv_end(tmp_path):
    m = make_master(60)
    write_vendor_csvs(tmp_path, m)
    last = m.index[-1]
    nxt = pd.bdate_range(last + pd.Timedelta(days=1), periods=2)
    prov = {}
    for a in ASSET_FILES:
        px = float(m[f"{a}_price"].iloc[-1])
        dates = [m.index[-2], *nxt]  # first row overlaps the CSV -> must be ignored (CSV is never overwritten)
        prov[a] = pd.DataFrame({"price": [999.0, px * 1.01, px * 1.02], "open": px, "high": px * 1.03, "low": px * 0.99}, index=dates)
    out = build_master_dataframe(tmp_path, provider_frames=prov)
    assert list(out.index[-2:]) == list(nxt) and len(out) == 62
    assert out.loc[m.index[-2], "brent_price"] == pytest.approx(m["brent_price"].iloc[-2], abs=0.01)  # CSV wins on overlap
    assert out["brent_price"].iloc[-1] == pytest.approx(float(m["brent_price"].iloc[-1]) * 1.02)


def test_missing_provider_rows_are_not_invented_for_non_price_columns(tmp_path):
    m = make_master(60)
    write_vendor_csvs(tmp_path, m)
    nxt = pd.bdate_range(m.index[-1] + pd.Timedelta(days=1), periods=1)
    prov = {a: pd.DataFrame({"price": [float(m[f"{a}_price"].iloc[-1])], "open": [np.nan], "high": [np.nan], "low": [np.nan]}, index=nxt)
            for a in ("wti", "dxy")}
    prov["brent"] = pd.DataFrame({"price": [80.0], "open": [80.0], "high": [81.0], "low": [79.0]}, index=nxt)
    out = build_master_dataframe(tmp_path, provider_frames=prov)
    assert np.isnan(out["wti_high"].iloc[-1]) and np.isnan(out["brent_vol"].iloc[-1])  # not forward-filled


def test_extend_with_provider_noop_without_newer_rows():
    csv = pd.DataFrame({"brent_price": [1.0]}, index=pd.to_datetime(["2026-10-23"]))
    prov = pd.DataFrame({"price": [1.0], "open": 1.0, "high": 1.0, "low": 1.0}, index=pd.to_datetime(["2026-10-22"]))
    out, n = extend_with_provider(csv, prov, "brent")
    assert n == 0 and out.equals(csv)


# ---- refresh_yahoo end-to-end with a fake provider and an in-memory store --------------------------------------- #
class FakeProvider:
    assets = ("brent", "wti", "dxy")

    def __init__(self, csv_master, factor=None, fail=()):
        self.m, self.factor, self.fail = csv_master, factor or {}, fail

    def fetch_ohlc(self, asset, start):
        if asset in self.fail:
            raise ProviderUnavailable("boom")
        px = self.m[f"{asset}_price"] * self.factor.get(asset, 1.0)
        df = px[px.index >= start].rename("price").to_frame()
        extra = pd.bdate_range(self.m.index[-1] + pd.Timedelta(days=1), periods=2)
        add = pd.DataFrame({"price": [float(px.iloc[-1]) * 1.01, float(px.iloc[-1]) * 1.02]}, index=extra)
        df = pd.concat([df, add])
        df["open"], df["high"], df["low"] = df["price"], df["price"] * 1.01, df["price"] * 0.99
        return df.rename_axis("date").reset_index()


@pytest.fixture()
def fake_store(monkeypatch):
    tables = {}

    def persist(df, table, db_path=None, **kw):
        tables[table] = df.copy()

    def read(table, db_path=None):
        if table not in tables:
            raise FileNotFoundError(table)
        return tables[table].copy()

    monkeypatch.setattr(providers, "persist_dataframe", persist)
    monkeypatch.setattr(providers, "read_table", read)
    return tables


def test_refresh_stores_validated_rows_and_reports_per_asset_failures(tmp_path, fake_store):
    m = make_master(60)
    write_vendor_csvs(tmp_path, m)
    today = pd.bdate_range(m.index[-1] + pd.Timedelta(days=1), periods=3)[-1]  # later than both fake bars
    st = refresh_yahoo("x.duckdb", tmp_path, today=today + pd.Timedelta(days=1), provider=FakeProvider(m, fail=("wti",)))
    assert st["brent"]["ok"] and st["brent"]["rows_new"] >= 2 and st["wti"]["ok"] is False and "boom" in st["wti"]["message"]
    assert "raw_provider_yahoo_brent" in fake_store and "raw_provider_yahoo_wti" not in fake_store
    again = refresh_yahoo("x.duckdb", tmp_path, today=today + pd.Timedelta(days=1), provider=FakeProvider(m))
    assert again["brent"]["rows_new"] == 0                                    # idempotent: no duplicate rows
    assert not fake_store["raw_provider_yahoo_brent"].index.duplicated().any()


def test_refresh_rejects_a_series_that_does_not_match_the_existing_history(tmp_path, fake_store):
    m = make_master(60)
    write_vendor_csvs(tmp_path, m)
    today = m.index[-1] + pd.Timedelta(days=10)
    st = refresh_yahoo("x.duckdb", tmp_path, today=today, provider=FakeProvider(m, factor={"dxy": 1.5}))
    assert st["dxy"]["ok"] is False and "differ" in st["dxy"]["message"]
    assert "raw_provider_yahoo_dxy" not in fake_store                          # nothing from the mismatching symbol is kept
