"""
src/features.py  --  LAYER 2: Feature Engineering Engine
=========================================================

Turns the aligned master DataFrame (see ``src/ingestion.py``) into a model-ready
feature matrix and the binary 3-day classification target, then persists both
into the intermediate feature store (SQLite).

Feature families (all computed from PAST data only -- no lookahead)
--------------------------------------------------------------------
1. Cross-asset spreads
     * Brent-WTI spread        : seaborne vs. landlocked crude (transport / quality premium)
     * Brent-DXY z-score gap   : oil vs. dollar strength, z-scored so scales are comparable
2. Log returns (stationary inputs) for Brent, WTI, DXY
3. Rolling 14-day correlations (macro-regime detectors)
4. Volatility / stress: 7d & 30d realised vol, intraday range ratio
5. Trend / mean-reversion: price relative to EMA20, SMA50, SMA200
6. Momentum "memory" lags
7. Calendar context: day-of-week, month

Target
------
``target_{H}d_class`` = 1 if Brent's settlement H trading days after the feature date is >=
the settlement on the feature date (Up/Flat), else 0 (Down).  Default H = 1, i.e. the
NEXT session: the features of the last completed session predict whether the coming
(current) session will settle at or above the previous settlement.  See ``src/sessions.py``.

IMPORTANT (bug fixed vs. the prototype notebook)
------------------------------------------------
The notebook built the label with ``np.where(future_return >= 0, 1, 0)``.
For the last H rows ``future_return`` is NaN, and ``NaN >= 0`` is False, so
those rows silently became label **0 ("Down")** and survived ``dropna()``.
Here the label stays NaN where the future is unknown, so those rows are
correctly removed from training.  They are, however, kept in the
``inference_features`` table because they are exactly the rows we want to
score live.

Run standalone::

    python -m src.features
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd

from src.config import FEATURE_STORE_PATH, PROJECT_ROOT  # noqa: F401  (also loads .env)
from src.ingestion import run_ingestion
from src.store import persist_dataframe, read_table

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
# Constants shared with train.py / api.py / explainer.py                      #
# --------------------------------------------------------------------------- #
HORIZON: int = int(os.getenv("FORECAST_HORIZON", "1"))
TARGET_COL: str = f"target_{HORIZON}d_class"
LOOKBACK_DAYS: int = int(os.getenv("LOOKBACK_DAYS", "60"))

FEATURE_TABLE = "feature_matrix"        # labelled rows (training / evaluation)
INFERENCE_TABLE = "inference_features"  # all complete feature rows incl. unlabelled tail

FEATURE_COLUMNS = [
    # 1. cross-asset spreads
    "spread_brent_wti",
    "spread_brent_dxy",
    # 2. log returns
    "brent_log_return",
    "wti_log_return",
    "dxy_log_return",
    # 3. rolling correlations
    "rolling_corr_brent_dxy_14d",
    "rolling_corr_brent_wti_14d",
    # 4. volatility & stress
    "brent_volatility_7d",
    "brent_volatility_30d",
    "high_low_spread_ratio",
    # 5. trend / mean reversion
    "distance_ema20",
    "distance_sma50",
    "distance_sma200",
    # 6. momentum lags
    "brent_return_lag1",
    "brent_return_lag2",
    "brent_return_lag3",
    "wti_return_lag1",
    "dxy_return_lag1",
    # 7. calendar
    "day_of_week",
    "month_of_year",
]

TARGET_DEFINITION = (
    (
        "1 if Brent's settlement on the NEXT trading session >= the settlement of the session the features "
        "describe (Up/Flat), else 0 (Down). Features use data up to that completed session only, so the "
        "prediction is made before the next session settles. NaN (unlabelled) until that session has settled."
    )
    if HORIZON == 1
    else (
        f"1 if Brent settlement {HORIZON} trading days ahead >= today's settlement (Up/Flat), "
        f"else 0 (Down); NaN (unlabelled) when the future price is not yet known."
    )
)
# Fingerprint of (feature names + order + target + horizon).  Stored with every
# registered model and re-checked at load time so training and inference can
# never silently use different schemas.
FEATURE_SCHEMA_HASH: str = hashlib.sha256(
    json.dumps({"features": FEATURE_COLUMNS, "target": TARGET_COL, "horizon": HORIZON}).encode()
).hexdigest()[:16]


# --------------------------------------------------------------------------- #
# 1. Feature engineering                                                      #
# --------------------------------------------------------------------------- #
def engineer_features(master_df: pd.DataFrame) -> pd.DataFrame:
    """
    Build the feature matrix from the aligned master DataFrame.

    Every statistic uses ``rolling``/``ewm``/``shift(+k)`` so row *t* only sees
    information available at the close of day *t*.  Leading rows will contain
    NaN while the longest window (252 days for the Brent-DXY z-score) fills up.
    """
    f = pd.DataFrame(index=master_df.index)
    brent, wti = master_df["brent_price"], master_df["wti_price"]
    dxy = master_df["dxy_price"]

    # ---- 1. Cross-asset spreads & arbitrage proxies ----------------------- #
    f["spread_brent_wti"] = brent - wti                  # seaborne vs. domestic crude

    # Brent and DXY live on different scales, so z-score each over ~1 trading
    # year before differencing; the gap says "oil rich/cheap relative to USD".
    brent_z = (brent - brent.rolling(252).mean()) / brent.rolling(252).std()
    dxy_z = (dxy - dxy.rolling(252).mean()) / dxy.rolling(252).std()
    f["spread_brent_dxy"] = brent_z - dxy_z

    # ---- 2. Log returns (additive, approx. stationary) -------------------- #
    for asset, px in (("brent", brent), ("wti", wti), ("dxy", dxy)):
        f[f"{asset}_log_return"] = np.log(px / px.shift(1))

    # ---- 3. Rolling correlations (macro regime) --------------------------- #
    brent_ret = f["brent_log_return"]
    f["rolling_corr_brent_dxy_14d"] = brent_ret.rolling(14).corr(f["dxy_log_return"])
    f["rolling_corr_brent_wti_14d"] = brent_ret.rolling(14).corr(f["wti_log_return"])

    # ---- 4. Volatility & market stress ------------------------------------ #
    f["brent_volatility_7d"] = brent_ret.rolling(7).std()
    f["brent_volatility_30d"] = brent_ret.rolling(30).std()
    f["high_low_spread_ratio"] = (master_df["brent_high"] - master_df["brent_low"]) / brent

    # ---- 5. Multi-scale trend / mean-reversion ratios --------------------- #
    f["distance_ema20"] = brent / brent.ewm(span=20, adjust=False).mean()
    f["distance_sma50"] = brent / brent.rolling(50).mean()
    f["distance_sma200"] = brent / brent.rolling(200).mean()

    # ---- 6. Momentum lags -------------------------------------------------- #
    f["brent_return_lag1"] = f["brent_log_return"].shift(1)
    f["brent_return_lag2"] = f["brent_log_return"].shift(2)
    f["brent_return_lag3"] = f["brent_log_return"].shift(3)
    f["wti_return_lag1"] = f["wti_log_return"].shift(1)
    f["dxy_return_lag1"] = f["dxy_log_return"].shift(1)

    # ---- 7. Calendar context ----------------------------------------------- #
    f["day_of_week"] = f.index.dayofweek
    f["month_of_year"] = f.index.month

    return f[FEATURE_COLUMNS]


# --------------------------------------------------------------------------- #
# 2. Target construction                                                      #
# --------------------------------------------------------------------------- #
def build_target(master_df: pd.DataFrame, horizon: int = HORIZON) -> pd.Series:
    """
    Binary label: 1 if Brent is Up/Flat ``horizon`` trading days ahead, 0 if Down.

    The label is NaN wherever the future price is unknown (last ``horizon`` rows),
    so those rows can never be mistaken for genuine "Down" observations.
    """
    price = master_df["brent_price"]
    future_return = price.shift(-horizon) / price - 1.0
    target = (future_return >= 0).astype(float).where(future_return.notna())
    target.name = f"target_{horizon}d_class"
    return target


# --------------------------------------------------------------------------- #
# 3. Dataset assembly                                                         #
# --------------------------------------------------------------------------- #
def build_datasets(master_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Returns
    -------
    labelled : features + target, NaN rows dropped  -> training/evaluation matrix
    live     : features only, for every row with a complete feature vector
               (includes the most recent days whose 3-day outcome is still unknown)
    """
    features = engineer_features(master_df)
    target = build_target(master_df)

    live = features.dropna().copy()
    labelled = pd.concat([features, target], axis=1).dropna().copy()
    labelled[TARGET_COL] = labelled[TARGET_COL].astype(int)
    return labelled, live


def run_feature_pipeline(
    db_path: Optional[str | Path] = None,
    master_df: Optional[pd.DataFrame] = None,
    provider_status: Optional[dict] = None,
) -> pd.DataFrame:
    """
    End-to-end: ingest (raw -> validated/cleaned) -> engineer -> persist to the store.

    ``master_df`` may be supplied (tests, or a caller that already ingested); otherwise
    the CSVs (+ stored live rows) are ingested and the raw + cleaned tables are written first.
    ``provider_status`` is the result of ``src.providers.refresh_yahoo`` (recorded in ``data_status``).
    Returns the labelled feature matrix that was written to ``feature_matrix``.
    """
    if master_df is None:
        master_df = run_ingestion(save=True, db_path=db_path, provider_status=provider_status)
    labelled, live = build_datasets(master_df)

    persist_dataframe(labelled, FEATURE_TABLE, db_path)
    persist_dataframe(live, INFERENCE_TABLE, db_path)

    logger.info(
        "Feature matrix: %d labelled rows x %d features | %d live rows (latest %s)",
        len(labelled),
        len(FEATURE_COLUMNS),
        len(live),
        live.index.max().date(),
    )
    return labelled


# --------------------------------------------------------------------------- #
# 4. Feature-store readers (used by train.py and api.py)                      #
# --------------------------------------------------------------------------- #
def load_feature_matrix(db_path: Optional[str | Path] = None) -> Tuple[pd.DataFrame, pd.Series]:
    """Load the labelled matrix from the store and split into (X, y)."""
    df = read_table(FEATURE_TABLE, db_path or FEATURE_STORE_PATH)
    return df[FEATURE_COLUMNS], df[TARGET_COL].astype(int)


def load_recent_context(
    n_days: int = LOOKBACK_DAYS,
    db_path: Optional[str | Path] = None,
    as_of: Optional[str | pd.Timestamp] = None,
) -> pd.DataFrame:
    """
    Fetch the most recent ``n_days`` complete feature rows ('Fetch Recent 60D Context').

    ``as_of`` caps the slice at that date (inclusive) so a historical reading can be
    reproduced without any row from after that date.
    """
    df = read_table(INFERENCE_TABLE, db_path or FEATURE_STORE_PATH)
    if as_of is not None:
        df = df.loc[: pd.Timestamp(as_of)]
    return df[FEATURE_COLUMNS].tail(n_days)


def load_labels(db_path: Optional[str | Path] = None) -> pd.Series:
    """Realised target (0/1) for every date whose outcome is already known."""
    df = read_table(FEATURE_TABLE, db_path or FEATURE_STORE_PATH)
    return df[TARGET_COL].astype(int)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    data = run_feature_pipeline()
    print("\n--- Feature Engineering & Target Complete ---")
    print(f"Final Feature Matrix (X) Shape: {data[FEATURE_COLUMNS].shape}")
    print(f"Class balance (share Up/Flat): {data[TARGET_COL].mean():.3f}")
    print(f"Persisted to: {FEATURE_STORE_PATH}")
