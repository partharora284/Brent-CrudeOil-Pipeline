"""
src/store.py  --  Intermediate feature store (DuckDB)
=====================================================

One embedded DuckDB file (no server) holds every layer of the data:

    raw_<asset>            verbatim snapshot of each vendor CSV (all strings)
    raw_provider_<n>_<a>   rows downloaded from an optional live provider
    master_aligned         cleaned + calendar-aligned prices (Brent calendar)
    feature_matrix         labelled model features + target (train/eval)
    inference_features     all complete feature rows, incl. unlabelled tail
    predictions            out-of-sample test signals + live API signals
    ingestion_log          one row per asset per ingestion run

DuckDB allows ONE writer process at a time; readers open the file read-only
and close it immediately (see ``_connect``), so the API, dashboard and the
scheduled ingestion job can safely share the file as long as writes are short.
"""

from __future__ import annotations

import logging
import re
import time
from pathlib import Path
from typing import Optional

import duckdb
import pandas as pd

from src.config import FEATURE_STORE_PATH

logger = logging.getLogger(__name__)
_IDENT = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")

PREDICTIONS_DDL = """
CREATE TABLE IF NOT EXISTS predictions (
    date DATE, model_version VARCHAR, split VARCHAR,
    prob_up DOUBLE, sentiment VARCHAR, actual INTEGER, generated_at TIMESTAMP
)"""


def _check_ident(name: str) -> str:
    if not _IDENT.match(name):
        raise ValueError(f"Illegal table name: {name!r}")
    return name


def _connect(db_path: Path, read_only: bool = False, retries: int = 5) -> duckdb.DuckDBPyConnection:
    """Open the store, retrying briefly if another process holds the write lock."""
    if db_path.exists() and db_path.stat().st_size >= 16:
        with open(db_path, "rb") as fh:
            if fh.read(15) == b"SQLite format 3":
                raise RuntimeError(
                    f"{db_path} is an old SQLite feature store, but the pipeline now uses DuckDB. "
                    "Set FEATURE_STORE_PATH=data/feature_store.duckdb in your .env (or delete the "
                    "line) and re-run `python -m src.features`."
                )
    last: Optional[Exception] = None
    for i in range(retries):
        try:
            return duckdb.connect(str(db_path), read_only=read_only)
        except duckdb.IOException as exc:  # locked by another process
            last = exc
            time.sleep(0.2 * (i + 1))
        except duckdb.Error as exc:
            raise RuntimeError(
                f"{db_path} is not a usable DuckDB file ({exc}). If it is an old SQLite "
                "store, set FEATURE_STORE_PATH=data/feature_store.duckdb in .env and re-run "
                "`python -m src.features`."
            ) from exc
    raise RuntimeError(f"Feature store {db_path} is locked by another process: {last}")


def persist_dataframe(
    df: pd.DataFrame,
    table: str,
    db_path: Optional[str | Path] = None,
    index_label: str = "date",
    index: bool = True,
) -> Path:
    """Write ``df`` to ``table`` (replacing it). A DatetimeIndex becomes a DATE column."""
    _check_ident(table)
    db_path = Path(db_path) if db_path else FEATURE_STORE_PATH
    db_path.parent.mkdir(parents=True, exist_ok=True)
    out = df.copy()
    if index:
        out.index = pd.to_datetime(out.index).normalize()
        out = out.rename_axis(index_label).reset_index()
        out[index_label] = out[index_label].astype("datetime64[ns]")
    con = _connect(db_path)
    try:
        con.register("_df", out)
        con.execute(f'CREATE OR REPLACE TABLE "{table}" AS SELECT * FROM _df')
        if index:
            con.execute(f'ALTER TABLE "{table}" ALTER "{index_label}" TYPE DATE')
        con.unregister("_df")
    finally:
        con.close()
    logger.info("Persisted %d rows to %s [%s]", len(out), db_path, table)
    return db_path


def read_table(table: str, db_path: Optional[str | Path] = None) -> pd.DataFrame:
    """Read a table back; if it has a ``date`` column it becomes the sorted index."""
    _check_ident(table)
    db_path = Path(db_path) if db_path else FEATURE_STORE_PATH
    if not db_path.exists():
        raise FileNotFoundError(
            f"Feature store not found at {db_path}. Run `python -m src.features` first."
        )
    con = _connect(db_path, read_only=True)
    try:
        if table not in list_tables(con):
            raise FileNotFoundError(
                f"Table '{table}' not found in {db_path}. Run `python -m src.features` first."
            )
        df = con.execute(f'SELECT * FROM "{table}"').df()
    finally:
        con.close()
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])
        df = df.set_index("date").sort_index()
    return df


def list_tables(con: duckdb.DuckDBPyConnection) -> list[str]:
    return [r[0] for r in con.execute("SELECT table_name FROM information_schema.tables").fetchall()]


def upsert_predictions(df: pd.DataFrame, db_path: Optional[str | Path] = None, replace_split: Optional[str] = None) -> None:
    """
    Insert rows into ``predictions`` (columns: date, model_version, split, prob_up,
    sentiment, actual, generated_at).  Existing rows with the same
    (date, model_version, split) are replaced.  ``replace_split`` first clears a
    whole split (used when training rewrites the out-of-sample test signals).
    """
    db_path = Path(db_path) if db_path else FEATURE_STORE_PATH
    db_path.parent.mkdir(parents=True, exist_ok=True)
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"]).astype("datetime64[ns]")
    df["generated_at"] = pd.to_datetime(df["generated_at"]).astype("datetime64[ns]")
    df["actual"] = df["actual"].astype("Int64")
    cols = ["date", "model_version", "split", "prob_up", "sentiment", "actual", "generated_at"]
    df = df[cols]
    con = _connect(db_path)
    try:
        con.execute(PREDICTIONS_DDL)
        if replace_split:
            con.execute("DELETE FROM predictions WHERE split = ?", [replace_split])
        con.register("_p", df)
        con.execute(
            "DELETE FROM predictions WHERE (date, model_version, split) IN "
            "(SELECT CAST(date AS DATE), model_version, split FROM _p)"
        )
        con.execute("INSERT INTO predictions SELECT CAST(date AS DATE), model_version, split, "
                    "prob_up, sentiment, actual, generated_at FROM _p")
        con.unregister("_p")
    finally:
        con.close()


def append_rows(df: pd.DataFrame, table: str, db_path: Optional[str | Path] = None) -> None:
    """Append rows to an existing table (create it from ``df`` if missing). No index."""
    _check_ident(table)
    db_path = Path(db_path) if db_path else FEATURE_STORE_PATH
    db_path.parent.mkdir(parents=True, exist_ok=True)
    con = _connect(db_path)
    try:
        con.register("_a", df)
        if table in list_tables(con):
            con.execute(f'INSERT INTO "{table}" BY NAME SELECT * FROM _a')
        else:
            con.execute(f'CREATE TABLE "{table}" AS SELECT * FROM _a')
        con.unregister("_a")
    finally:
        con.close()
