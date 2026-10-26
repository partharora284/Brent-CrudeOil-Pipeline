"""Shared fixtures: deterministic synthetic market data, isolated from real data/models."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

VENDOR_COLS = ["Date", "Price", "Open", "High", "Low", "Vol.", "Change %"]


def make_master(n: int = 900, seed: int = 0) -> pd.DataFrame:
    """Aligned master frame (random walks) with the columns the feature engine needs."""
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2018-01-01", periods=n)

    def walk(start, vol):
        return start * np.exp(np.cumsum(rng.normal(0, vol, n)))

    brent = walk(70, 0.015)
    df = pd.DataFrame(index=idx)
    df["brent_price"] = brent
    df["brent_high"] = brent * 1.01
    df["brent_low"] = brent * 0.99
    df["wti_price"] = brent * 0.95 * np.exp(rng.normal(0, 0.003, n))
    df["dxy_price"] = walk(95, 0.003)
    return df


def write_vendor_csvs(raw_dir, master: pd.DataFrame) -> None:
    """Write the three vendor-format CSVs (newest first, dd-mm-yyyy) for ingestion tests."""
    stems = {"brent": "Brent_Oil_Futures_Historical_Data", "wti": "Crude_Oil_WTI_Futures_Historical_Data",
             "dxy": "US_Dollar_Index_Historical_Data"}
    for a, stem in stems.items():
        px = master[f"{a}_price"]
        d = pd.DataFrame({
            "Date": master.index.strftime("%d-%m-%Y"),
            "Price": [f"{v:,.2f}" for v in px],
            "Open": [f"{v:,.2f}" for v in px],
            "High": [f"{v * 1.01:,.2f}" for v in px],
            "Low": [f"{v * 0.99:,.2f}" for v in px],
            "Vol.": ["1.5K"] * len(px),
            "Change %": ["0.10%"] * len(px),
        })[VENDOR_COLS].iloc[::-1]
        d.to_csv(raw_dir / f"{stem}.csv", index=False)


@pytest.fixture(scope="session")
def master_df() -> pd.DataFrame:
    return make_master()


@pytest.fixture(scope="session")
def trained_env(tmp_path_factory, master_df):
    """Synthetic feature store + a fast training run, entirely inside a tmp dir."""
    from src.features import run_feature_pipeline
    from src.train import run_training

    root = tmp_path_factory.mktemp("env")
    db, models = root / "store.duckdb", root / "models"
    run_feature_pipeline(db, master_df=master_df)
    meta = run_training(db_path=db, model_dir=models, n_search_iter=2, n_splits=3)
    return {"db": db, "models": models, "meta": meta, "root": root}
