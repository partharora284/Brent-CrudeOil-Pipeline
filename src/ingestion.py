"""
src/ingestion.py  --  LAYER 2: Data Extraction & Schema Normalizer
===================================================================

Responsibilities
----------------
1. Read the three raw CSVs (Brent, WTI, DXY) exported
   from a market-data vendor (Investing.com style: "Date, Price, Open, High,
   Low, Vol., Change %").
2. Normalise each file to a clean, numeric, chronologically sorted schema with
   asset-prefixed column names (``brent_price``, ``wti_high`` ...).
3. Align all three assets on ONE trading calendar and return a single
   "master" DataFrame.
4. Validate every file, keep a verbatim raw snapshot, and persist the cleaned,
   aligned table into the intermediate feature store (DuckDB, see src/store.py) so the result can be inspected without re-running the pipeline.

Quantitative-finance rationale
------------------------------
* **Brent is the master calendar.**  We are predicting Brent, so the only dates
  that matter are dates on which Brent actually settled.
* **LEFT join, never outer join.**  An outer join would create rows for days
  when Brent was closed (e.g. UK bank holidays) but WTI / DXY traded.  Those
  rows would carry a stale Brent price, so the "return" on that day would be a
  fake 0.00% -- a classic source of *fake-zero-return leakage* that makes a
  model look smarter than it is.  Left-joining onto Brent's dates removes
  those rows entirely.
* **Forward-fill only secondary assets.**  If WTI was closed on a day Brent
  traded, the last known WTI settlement is the best information a trader
  would actually have had that day.  Forward-filling uses only PAST values,
  so it cannot leak future information.

Run standalone::

    python -m src.ingestion
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from src.config import (  # noqa: F401  (re-exported: other modules import these from here)
    FEATURE_STORE_PATH,
    PROJECT_ROOT,
    RAW_DATA_DIR,
    resolve_path,
)
from src.store import append_rows, persist_dataframe, read_table  # noqa: F401

logger = logging.getLogger(__name__)

# asset prefix -> human-readable stem of the vendor CSV file name.
# Matching is tolerant: "Brent Oil Futures Historical Data.csv" and
# "Brent_Oil_Futures_Historical_Data.csv" are both accepted.
ASSET_FILES: Dict[str, str] = {
    "brent": "Brent Oil Futures Historical Data",
    "wti": "Crude Oil WTI Futures Historical Data",
    "dxy": "US Dollar Index Historical Data",
}
REQUIRED_RAW_COLUMNS = ["Date", "Price", "Open", "High", "Low"]
DATE_FORMAT = "%d-%m-%Y"          # vendor format (day-month-year)
MAX_GAP_DAYS = 10                 # calendar-day gap that triggers a warning
STALE_SECONDARY_DAYS = 5          # secondary asset older than Brent by this many days -> warning


class DataValidationError(ValueError):
    """Raised when a source file is unusable (missing columns, no valid dates ...)."""


@dataclass
class QualityReport:
    """Outcome of validating one asset file. Errors abort ingestion; warnings are logged + stored."""

    asset: str
    source: str
    sha256: str = ""
    rows_raw: int = 0
    rows_valid: int = 0
    start: Optional[str] = None
    end: Optional[str] = None
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return self.__dict__.copy()


# --------------------------------------------------------------------------- #
# Helpers                                                                     #
# --------------------------------------------------------------------------- #
def _normalise_name(name: str) -> str:
    """Lower-case a file stem and collapse spaces/underscores for fuzzy matching."""
    return re.sub(r"[\s_]+", " ", name).strip().lower()


def find_csv(raw_dir: Path, stem: str) -> Path:
    """Locate a vendor CSV in ``raw_dir`` regardless of space/underscore naming."""
    target = _normalise_name(stem)
    for candidate in sorted(Path(raw_dir).glob("*.csv")):
        if _normalise_name(candidate.stem) == target:
            return candidate
    raise FileNotFoundError(
        f"Could not find '{stem}.csv' (or its underscore variant) in {raw_dir}. "
        "Copy the three historical CSVs into data/raw/ and try again."
    )


def _parse_volume(v) -> float:
    """
    Convert vendor volume strings ('34.55K', '1.2M', '-', '') to floats.

    Volume is often blank for index products (DXY) and for thinly traded
    sessions; blanks become NaN and are handled by the forward-fill step.
    """
    if v is None or (not isinstance(v, str) and pd.isna(v)) or v == "-":
        return np.nan
    v_str = str(v).strip().upper()
    if v_str in {"", "-"}:
        return np.nan
    try:
        if "K" in v_str:
            return float(v_str.replace("K", "").replace(",", "")) * 1_000
        if "M" in v_str:
            return float(v_str.replace("M", "").replace(",", "")) * 1_000_000
        if "B" in v_str:
            return float(v_str.replace("B", "").replace(",", "")) * 1_000_000_000
        return float(v_str.replace(",", ""))
    except ValueError:
        return np.nan


# --------------------------------------------------------------------------- #
# 0. Raw loading + validation                                                 #
# --------------------------------------------------------------------------- #
def _sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_raw_csv(filepath: str | Path) -> pd.DataFrame:
    """Read a vendor CSV verbatim (every column as text). The file itself is never modified."""
    return pd.read_csv(filepath, encoding="utf-8-sig", dtype=str, keep_default_na=False)


def validate_raw_frame(raw: pd.DataFrame, asset: str, source: str = "", today: Optional[pd.Timestamp] = None) -> QualityReport:
    """
    Check schema, dates, numeric values, duplicates and gaps of one raw asset frame.

    Hard errors (ingestion stops): missing required columns, empty file, no parseable
    dates, future-dated rows.  Everything else is recorded as a warning and handled
    downstream (unparseable rows are dropped, duplicate dates keep the last row).
    """
    rep = QualityReport(asset=asset, source=source, rows_raw=len(raw))
    missing = [c for c in REQUIRED_RAW_COLUMNS if c not in raw.columns]
    if missing:
        rep.errors.append(f"missing required columns {missing}; found {list(raw.columns)}")
        return rep
    if raw.empty:
        rep.errors.append("file has no data rows")
        return rep

    dates = pd.to_datetime(raw["Date"], format=DATE_FORMAT, errors="coerce")
    n_bad_dates = int(dates.isna().sum())
    if n_bad_dates == len(raw):
        rep.errors.append(f"no parseable dates (expected {DATE_FORMAT}); first value: {raw['Date'].iloc[0]!r}")
        return rep
    if n_bad_dates:
        rep.warnings.append(f"{n_bad_dates} rows with unparseable dates will be dropped")

    today = (today or pd.Timestamp.today()).normalize()
    n_future = int((dates > today).sum())
    if n_future:
        rep.errors.append(f"{n_future} rows are dated in the future (latest {dates.max().date()})")

    num = {c: pd.to_numeric(raw[c].astype(str).str.replace(",", "", regex=False), errors="coerce")
           for c in ["Price", "Open", "High", "Low"]}
    for c, v in num.items():
        n_nan = int(v[dates.notna()].isna().sum())
        if n_nan:
            rep.warnings.append(f"{n_nan} non-numeric/missing values in '{c}'")
    if (num["Price"] <= 0).any():
        rep.warnings.append(f"{int((num['Price'] <= 0).sum())} non-positive prices")
    inverted = int((num["High"] < num["Low"]).sum())
    if inverted:
        rep.warnings.append(f"{inverted} rows where High < Low")

    valid_dates = dates.dropna()
    n_dup = int(valid_dates.duplicated().sum())
    if n_dup:
        rep.warnings.append(f"{n_dup} duplicate dates (last occurrence kept)")
    uniq = valid_dates.drop_duplicates().sort_values().reset_index(drop=True)
    gaps = uniq.diff().dt.days
    big = gaps[gaps > MAX_GAP_DAYS]
    if len(big):
        spans = [f"{uniq[i - 1].date()}->{uniq[i].date()}" for i in big.index[:3]]
        rep.warnings.append(f"{len(big)} gaps > {MAX_GAP_DAYS} calendar days (e.g. {', '.join(spans)})")

    rep.rows_valid = int(len(uniq))
    rep.start, rep.end = str(uniq.min().date()), str(uniq.max().date())
    return rep


# --------------------------------------------------------------------------- #
# 1. Single-asset cleaner                                                     #
# --------------------------------------------------------------------------- #
def clean_financial_csv(filepath: str | Path, prefix: str) -> pd.DataFrame:
    """
    Load and standardise ONE asset CSV.

    Steps
    -----
    1. Parse ``Date`` (day-month-year) and sort ascending -- vendor files are
       newest-first, but every rolling/shift operation downstream assumes
       oldest-first.
    2. Strip thousands separators from price columns (e.g.
       ``'1,385.00'``) and cast to float.
    3. Parse volume strings (``'34.55K'``) to numbers.
    4. Convert ``Change %`` text to a decimal fraction.
    5. Keep only the OHLCV + change columns and prefix them with the asset
       name (``brent_price``, ``wti_open`` ...) so the assets can live side by
       side in one frame without column collisions.

    Parameters
    ----------
    filepath : path to the vendor CSV.
    prefix   : asset tag used as column prefix ('brent', 'wti', 'dxy').

    Returns
    -------
    DataFrame indexed by ``Date`` (ascending, unique) with prefixed columns.
    """
    raw = read_raw_csv(filepath)
    report = validate_raw_frame(raw, prefix, source=str(filepath))
    for w in report.warnings:
        logger.warning("[%s] %s", prefix, w)
    if report.errors:
        raise DataValidationError(f"[{prefix}] {Path(filepath).name}: " + "; ".join(report.errors))
    df = raw.copy()

    # 1. Dates -> DatetimeIndex, chronological order, de-duplicated
    df["Date"] = pd.to_datetime(df["Date"], format=DATE_FORMAT, errors="coerce")
    df = df[df["Date"].notna()].set_index("Date").sort_index(ascending=True)
    df = df[~df.index.duplicated(keep="last")]  # one settlement per day

    # 2. Numeric price columns (handle '1,385.00' style strings)
    for col in ["Price", "Open", "High", "Low"]:
        if col in df.columns:
            df[col] = pd.to_numeric(
                df[col].astype(str).str.replace(",", "", regex=False), errors="coerce"
            )

    # 3. Volume
    if "Vol." in df.columns:
        df["Vol."] = df["Vol."].apply(_parse_volume)

    # 4. Change % -> decimal ("2.55%" -> 0.0255)
    if "Change %" in df.columns:
        df["Change %"] = (
            pd.to_numeric(
                df["Change %"].astype(str).str.replace("%", "", regex=False), errors="coerce"
            )
            / 100.0
        )

    # 5. Keep & rename columns with the asset prefix
    cols_to_keep = ["Price", "Open", "High", "Low", "Vol.", "Change %"]
    df = df[[c for c in cols_to_keep if c in df.columns]].copy()
    df.columns = [
        f"{prefix}_{c.lower().replace('.', '').replace(' ', '_').replace('%', 'pct')}"
        for c in df.columns
    ]
    return df


# --------------------------------------------------------------------------- #
# 2. Multi-asset calendar alignment                                           #
# --------------------------------------------------------------------------- #
def extend_with_provider(csv_df: pd.DataFrame, prov: Optional[pd.DataFrame], prefix: str) -> tuple[pd.DataFrame, int]:
    """
    Append validated live-provider rows that are NEWER than the CSV's last date.

    The CSV history is never overwritten.  Provider rows only supply price/open/high/low; the other
    CSV columns (volume, change %) stay NaN for those rows - they are not used by any feature and
    are deliberately not invented.
    """
    if prov is None or prov.empty:
        return csv_df, 0
    new = prov[prov.index > csv_df.index.max()]
    if new.empty:
        return csv_df, 0
    add = pd.DataFrame({f"{prefix}_{c}": new[c] for c in ("price", "open", "high", "low")}, index=new.index)
    return pd.concat([csv_df, add]).sort_index(), len(add)


def _build_master(raw_dir: Path, provider_frames: Optional[Dict[str, pd.DataFrame]] = None) -> tuple[pd.DataFrame, dict]:
    """Core of ``build_master_dataframe``; also returns each asset's last REAL observation date."""
    logger.info("Ingesting all three datasets from %s ...", raw_dir)
    frames: Dict[str, pd.DataFrame] = {}
    provider_rows: Dict[str, int] = {}
    for prefix, stem in ASSET_FILES.items():
        df = clean_financial_csv(find_csv(raw_dir, stem), prefix)
        frames[prefix], provider_rows[prefix] = extend_with_provider(df, (provider_frames or {}).get(prefix), prefix)
    last_observation = {prefix: f.index.max() for prefix, f in frames.items()}

    # Brent = absolute truth for the calendar.  LEFT join everything else onto it.
    # Left join prevents 0% return leakage on holidays: dates on which only WTI
    # or DXY traded never enter the dataset.
    master_df = frames["brent"].copy()
    for prefix in ("wti", "dxy"):
        master_df = master_df.join(frames[prefix], how="left")

    # DXY is an index and has no meaningful volume -> drop if present.
    if "dxy_vol" in master_df.columns:
        master_df = master_df.drop(columns=["dxy_vol"])

    # Forward-fill ONLY the settlement PRICES: a secondary market that was closed on a day Brent
    # traded keeps its last settlement (information available at that time).  Other columns are
    # never filled, so no value is ever silently substituted.  Rows still missing a required
    # input (leading history) are dropped.
    rows_before = len(master_df)
    brent_last = frames["brent"].index.max()
    for prefix in ("wti", "dxy"):
        lag = (brent_last - frames[prefix].index.max()).days
        if lag > STALE_SECONDARY_DAYS:
            logger.warning(
                "[%s] last observation %s is %d days older than Brent (%s): its values are "
                "forward-filled and STALE for the most recent rows.",
                prefix, frames[prefix].index.max().date(), lag, brent_last.date(),
            )
    price_cols = [c for c in master_df.columns if c.endswith("_price")]
    master_df[price_cols] = master_df[price_cols].ffill()
    required = ["brent_price", "brent_high", "brent_low", "wti_price", "dxy_price"]
    missing = set(required) - set(master_df.columns)
    if missing:
        raise ValueError(f"Master dataset is missing required columns: {sorted(missing)}")
    master_df = master_df.dropna(subset=required)
    logger.info(
        "Master dataset: %d rows (%d dropped for incomplete history), %s -> %s",
        len(master_df),
        rows_before - len(master_df),
        master_df.index.min().date(),
        master_df.index.max().date(),
    )
    return master_df, {"last_observation": last_observation, "provider_rows": provider_rows}


def build_master_dataframe(
    raw_dir: Optional[str | Path] = None, provider_frames: Optional[Dict[str, pd.DataFrame]] = None
) -> pd.DataFrame:
    """
    Ingest all three datasets and align them on Brent's trading calendar.

    ``provider_frames`` (optional, from ``src.providers.load_provider_frames``) extends each asset
    past the end of its CSV with validated live rows.

    Returns
    -------
    DataFrame indexed by date with columns such as ``brent_price``,
    ``wti_price``, ``dxy_price`` (plus open/high/low/etc.).
    """
    return _build_master(Path(raw_dir) if raw_dir else RAW_DATA_DIR, provider_frames)[0]


# --------------------------------------------------------------------------- #
# 4. Entry point                                                              #
# --------------------------------------------------------------------------- #
def persist_raw_snapshots(raw_dir: Path, db_path: Optional[Path], reports: Dict[str, QualityReport]) -> None:
    """
    Store a verbatim copy of each vendor CSV (all text) in ``raw_<asset>`` with its
    SHA-256 and ingestion time.  The CSV files on disk are never touched, so the
    raw layer stays immutable and separate from the cleaned ``master_aligned`` table.
    """
    now = datetime.now(timezone.utc).replace(tzinfo=None)
    for prefix, stem in ASSET_FILES.items():
        path = find_csv(raw_dir, stem)
        snap = read_raw_csv(path)
        snap["source_file"] = path.name
        snap["source_sha256"] = reports[prefix].sha256
        snap["ingested_at"] = now
        persist_dataframe(snap, f"raw_{prefix}", db_path, index=False)


def run_ingestion(
    save: bool = True,
    db_path: Optional[str | Path] = None,
    raw_dir: Optional[str | Path] = None,
    report_dir: Optional[str | Path] = None,
    provider_status: Optional[Dict[str, dict]] = None,
) -> pd.DataFrame:
    """
    CSV (+ validated live-provider rows already stored in the feature store) -> validate -> raw
    snapshot + cleaned/aligned master table + ``data_status``.

    Works fully offline.  Live downloads are done separately by ``src.providers.refresh_yahoo``
    (see ``python -m src.scheduled_ingest``); pass its result as ``provider_status`` so the outcome
    is recorded in ``data_status`` for the API.
    """
    raw_dir = Path(raw_dir) if raw_dir else RAW_DATA_DIR
    db_path = Path(db_path) if db_path else FEATURE_STORE_PATH
    reports: Dict[str, QualityReport] = {}
    for prefix, stem in ASSET_FILES.items():
        path = find_csv(raw_dir, stem)
        rep = validate_raw_frame(read_raw_csv(path), prefix, source=path.name)
        rep.sha256 = _sha256(path)
        reports[prefix] = rep
    from src.providers import load_provider_frames

    master_df, meta = _build_master(raw_dir, load_provider_frames(db_path))
    if save:
        persist_raw_snapshots(raw_dir, db_path, reports)
        persist_dataframe(master_df, table="master_aligned", db_path=db_path)
        ts = datetime.now(timezone.utc).replace(tzinfo=None)
        persist_dataframe(pd.DataFrame([{
            "asset": a,
            "last_observation": str(meta["last_observation"][a].date()),
            "provider_rows_used": int(meta["provider_rows"][a]),
            "provider_status": ("not_used" if not provider_status else "ok" if provider_status.get(a, {}).get("ok") else "failed"),
            "provider_message": str((provider_status or {}).get(a, {}).get("message", "")),
            "checked_at": ts,
        } for a in ASSET_FILES]), "data_status", db_path, index=False)
        log = pd.DataFrame([{**{k: v for k, v in r.to_dict().items() if k not in ("errors", "warnings")},
                             "n_warnings": len(r.warnings), "n_errors": len(r.errors), "run_at": ts}
                            for r in reports.values()])
        append_rows(log, "ingestion_log", db_path)
        report_dir = Path(report_dir) if report_dir else db_path.parent / "quality"  # next to the store -> test-safe
        report_dir.mkdir(parents=True, exist_ok=True)
        (report_dir / "ingestion_report.json").write_text(json.dumps(
            {"run_at": ts.isoformat(), "master_rows": len(master_df),
             "master_start": str(master_df.index.min().date()), "master_end": str(master_df.index.max().date()),
             "assets": {k: v.to_dict() for k, v in reports.items()}}, indent=2))
    return master_df


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    df = run_ingestion(save=True)
    print("\n--- True-Calendar Data Extraction Complete ---")
    print(f"Master Dataset Shape: {df.shape}")
    print(df.tail())
