import numpy as np
import pandas as pd

from src.experiments import (EXTRA_COLUMNS, MIN_TRAIN, N_BLOCKS, extra_features, holdout_split, qualifies,
                             walk_forward_blocks)
from src.train import EMBARGO, chronological_split


def test_extra_features_only_use_past_data_and_are_well_formed(master_df):
    full = extra_features(master_df)
    assert list(full.columns) == EXTRA_COLUMNS
    assert full.dropna().shape[0] > len(full) - 40 and np.isfinite(full.dropna().to_numpy()).all()
    assert full["brent_rsi_14"].dropna().between(0, 100).all()
    part = extra_features(master_df.iloc[:-30])           # truncating the future must not change earlier values
    pd.testing.assert_frame_equal(part.dropna(), full.loc[part.dropna().index], check_exact=False, atol=1e-12)


def test_walk_forward_blocks_are_chronological_purged_and_stay_in_the_training_region():
    n = 2232
    blocks = walk_forward_blocks(n)
    assert len(blocks) == N_BLOCKS and blocks[0][0] + EMBARGO >= MIN_TRAIN
    for (te, a, b), nxt in zip(blocks, blocks[1:] + [None]):
        assert a - te == EMBARGO and a < b <= n               # purge gap; test block inside the training region
        if nxt:
            assert nxt[1] == b                                # consecutive, non-overlapping test blocks


def test_holdout_matches_the_projects_own_chronological_split():
    X = pd.DataFrame({"a": range(1000)})
    y = pd.Series(range(1000))
    Xtr, Xte, *_ = chronological_split(X, y)
    n_region, first = holdout_split(len(X))
    assert len(Xtr) == n_region and Xte.index[0] == first


def test_selection_rule_rejects_candidates_that_do_not_beat_always_up():
    good = dict(mean_edge=0.02, mean_auc=0.53, blocks_edge_pos=5, blocks_auc_pos=5)
    assert qualifies(good)
    assert not qualifies({**good, "mean_edge": -0.01})
    assert not qualifies({**good, "mean_auc": 0.49})
    assert not qualifies({**good, "blocks_edge_pos": 3})
