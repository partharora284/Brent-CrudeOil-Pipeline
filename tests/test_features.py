import numpy as np
import pandas as pd

from src.features import FEATURE_COLUMNS, HORIZON, TARGET_COL, build_datasets, build_target, engineer_features
from src.train import EMBARGO, chronological_split


def test_target_alignment_and_unknown_tail_is_nan(master_df):
    t = build_target(master_df)
    px = master_df["brent_price"]
    i = 100
    assert t.iloc[i] == float(px.iloc[i + HORIZON] >= px.iloc[i])
    assert t.iloc[-HORIZON:].isna().all()  # unknown future is NOT labelled "Down"


def test_labelled_excludes_tail_but_live_includes_it(master_df):
    labelled, live = build_datasets(master_df)
    assert labelled.index.max() < live.index.max()
    assert live.index.max() == master_df.index.max()
    assert set(labelled[TARGET_COL].unique()) <= {0, 1}


def test_features_have_no_lookahead(master_df):
    """Features at date t must be identical whether or not later data exists."""
    full = engineer_features(master_df)
    cut = 600
    part = engineer_features(master_df.iloc[:cut])
    pd.testing.assert_frame_equal(full.iloc[:cut], part, check_exact=False, rtol=1e-12, atol=1e-12)


def test_feature_schema_order_is_stable(master_df):
    f = engineer_features(master_df)
    assert list(f.columns) == FEATURE_COLUMNS
    labelled, live = build_datasets(master_df)
    assert list(live.columns) == FEATURE_COLUMNS
    assert list(labelled.columns) == FEATURE_COLUMNS + [TARGET_COL]
    assert np.isfinite(live.to_numpy()).all()


def test_chronological_split_has_embargo_and_no_overlap(master_df):
    labelled, _ = build_datasets(master_df)
    X, y = labelled[FEATURE_COLUMNS], labelled[TARGET_COL]
    Xtr, Xte, ytr, yte = chronological_split(X, y)
    assert Xtr.index.max() < Xte.index.min()
    gap_rows = len(labelled.loc[(labelled.index > Xtr.index.max()) & (labelled.index < Xte.index.min())])
    assert gap_rows == EMBARGO  # exactly EMBARGO rows purged between train and test
    # the last training label looks HORIZON sessions ahead; that window must end before the first test row
    pos = {d: i for i, d in enumerate(master_df.index)}
    assert pos[Xtr.index.max()] + HORIZON < pos[Xte.index.min()]


def test_target_is_the_next_session_not_three_days_ahead(master_df):
    """Current-session model: row t is labelled with whether session t+1 settles >= session t."""
    assert HORIZON == 1 and TARGET_COL == "target_1d_class"  # a FORECAST_HORIZON=3 in .env would silently break the dashboard's meaning
    t = build_target(master_df)
    px = master_df["brent_price"]
    assert t.iloc[50] == float(px.iloc[51] >= px.iloc[50])
    assert t.iloc[-1:].isna().all() and not t.iloc[:-1].isna().any()
