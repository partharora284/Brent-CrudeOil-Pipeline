import hashlib

import pandas as pd
import pytest

from src.ingestion import (DataValidationError, build_master_dataframe, clean_financial_csv, run_ingestion,
                           validate_raw_frame)
from src.store import read_table
from tests.conftest import make_master, write_vendor_csvs


def raw(rows):
    return pd.DataFrame(rows, columns=["Date", "Price", "Open", "High", "Low"])


def test_validation_flags_missing_columns():
    rep = validate_raw_frame(pd.DataFrame({"Date": ["01-01-2020"], "Price": ["1"]}), "brent")
    assert rep.errors and "missing required columns" in rep.errors[0]


def test_validation_flags_future_dates_and_bad_dates_and_dupes():
    r = raw([["01-01-2020", "10", "10", "11", "9"], ["01-01-2020", "11", "10", "12", "9"],
             ["not-a-date", "1", "1", "1", "1"], ["01-01-2099", "12", "10", "12", "9"]])
    rep = validate_raw_frame(r, "brent")
    assert any("future" in e for e in rep.errors)
    assert any("unparseable" in w for w in rep.warnings)
    assert any("duplicate" in w for w in rep.warnings)


def test_validation_warns_non_numeric_and_inverted_range():
    rep = validate_raw_frame(raw([["01-01-2020", "abc", "1", "1", "2"], ["02-01-2020", "5", "5", "4", "6"]]), "wti")
    assert any("non-numeric" in w for w in rep.warnings)
    assert any("High < Low" in w for w in rep.warnings)


def test_clean_csv_sorts_ascending_and_parses_numbers(tmp_path):
    f = tmp_path / "x.csv"
    f.write_text('"Date","Price","Open","High","Low","Vol.","Change %"\n'
                 '"03-01-2020","1,385.00","1","2","1","2.5K","1.50%"\n'
                 '"02-01-2020","1,380.00","1","2","1","","-0.10%"\n', encoding="utf-8-sig")
    df = clean_financial_csv(f, "wti")
    assert df.index.is_monotonic_increasing
    assert df["wti_price"].iloc[-1] == 1385.0 and df["wti_vol"].iloc[-1] == 2500
    assert df["wti_change_pct"].iloc[-1] == pytest.approx(0.015)


def test_clean_csv_raises_on_schema_error(tmp_path):
    f = tmp_path / "bad.csv"
    f.write_text("Foo,Bar\n1,2\n")
    with pytest.raises(DataValidationError):
        clean_financial_csv(f, "brent")


def test_master_uses_brent_calendar_and_ffills_only_secondary(tmp_path):
    m = make_master(60)
    write_vendor_csvs(tmp_path, m)
    # drop a WTI day -> must be forward-filled; add a WTI-only extra day -> must not appear
    wti = tmp_path / "Crude_Oil_WTI_Futures_Historical_Data.csv"
    d = pd.read_csv(wti, dtype=str)
    missing_date = m.index[30].strftime("%d-%m-%Y")
    d = d[d["Date"] != missing_date]
    extra = pd.DataFrame([{"Date": "01-01-2019", "Price": "1", "Open": "1", "High": "1", "Low": "1", "Vol.": "", "Change %": "0%"}])
    pd.concat([extra, d]).to_csv(wti, index=False)
    out = build_master_dataframe(tmp_path)
    assert out.index.equals(m.index)  # Brent's calendar exactly
    assert out["wti_price"].iloc[30] == out["wti_price"].iloc[29]  # ffilled from the PAST


def test_run_ingestion_keeps_raw_untouched_and_separates_layers(tmp_path):
    m = make_master(80)
    raw_dir = tmp_path / "raw"; raw_dir.mkdir()
    write_vendor_csvs(raw_dir, m)
    before = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in raw_dir.glob("*.csv")}
    db = tmp_path / "s.duckdb"
    run_ingestion(save=True, db_path=db, raw_dir=raw_dir)
    after = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in raw_dir.glob("*.csv")}
    assert before == after
    raw_t = read_table("raw_brent", db)  # verbatim text snapshot
    assert "source_sha256" in raw_t.columns and raw_t["Price"].dtype == object or str(raw_t["Price"].dtype).startswith("str")
    assert len(read_table("master_aligned", db)) == 80
    assert (tmp_path / "quality" / "ingestion_report.json").exists()
