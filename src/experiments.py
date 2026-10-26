"""
src/experiments.py  --  Reproducible model/feature comparison (does NOT change the production system)
=======================================================================================================

    python -m src.experiments audit      # data / leakage checks (no model, no holdout)
    python -m src.experiments baseline   # current features + models on the project's chronological holdout (metrics only)
    python -m src.experiments walkforward   # compare feature sets x models on the TRAINING region only
    python -m src.experiments final --features base|extended --model <name> --param <index>
                                           # evaluate ONE pre-chosen candidate on the untouched holdout

Rules this script enforces
--------------------------
* The final holdout is the project's own split (last 20% of rows after a 1-row embargo, see ``src/train.py``).
  ``walkforward`` never reads those rows; only ``baseline`` and ``final`` do, and neither selects anything.
* Walk-forward = expanding training window, 6 consecutive chronological test blocks, 1-row embargo, no shuffling.
* Model/feature selection uses ONLY walk-forward results and a rule fixed in advance (``qualifies``).
* A candidate must also pass the project's existing gate on the holdout (beat BOTH baselines, ROC-AUC > 0.5);
  no threshold is lowered anywhere.
"""

from __future__ import annotations

import argparse
import sys
import warnings
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.features import FEATURE_COLUMNS, TARGET_COL, build_datasets, engineer_features
from src.train import EMBARGO, RANDOM_STATE, TRAIN_FRACTION, baseline_metrics, classification_metrics, validation_gate

warnings.filterwarnings("ignore")

N_BLOCKS = 6        # chronological walk-forward test blocks inside the training region
MIN_TRAIN = 500     # rows in the first training window (~2 years)

# Additional features: only Brent, WTI, DXY prices, all backward-looking.
EXTRA_COLUMNS = [
    "brent_ret_5d", "brent_ret_10d", "brent_ret_20d",   # multi-period momentum
    "brent_rsi_14",                                      # overbought / oversold
    "vol_ratio_7_30",                                    # is short-term volatility rising vs. the monthly level?
    "spread_brent_wti_chg_5d",                           # change in the Brent-WTI gap over a week
    "dxy_ret_5d",                                        # weekly dollar move
]


# --------------------------------------------------------------------------- #
# Data                                                                        #
# --------------------------------------------------------------------------- #
def load_master() -> pd.DataFrame:
    """Aligned master table from the feature store if present (includes live rows), else straight from the CSVs."""
    try:
        from src.store import read_table

        return read_table("master_aligned")
    except Exception:  # noqa: BLE001  (no store yet / DuckDB unavailable)
        from src.ingestion import build_master_dataframe

        return build_master_dataframe()


def extra_features(master: pd.DataFrame) -> pd.DataFrame:
    b, w, d = master["brent_price"], master["wti_price"], master["dxy_price"]
    r = np.log(b / b.shift(1))
    delta = b.diff()
    up, down = delta.clip(lower=0).ewm(alpha=1 / 14, adjust=False).mean(), (-delta.clip(upper=0)).ewm(alpha=1 / 14, adjust=False).mean()
    f = pd.DataFrame(index=master.index)
    for n in (5, 10, 20):
        f[f"brent_ret_{n}d"] = np.log(b / b.shift(n))
    f["brent_rsi_14"] = 100 - 100 / (1 + up / down.replace(0, np.nan))
    f["vol_ratio_7_30"] = r.rolling(7).std() / r.rolling(30).std()
    spread = b - w
    f["spread_brent_wti_chg_5d"] = spread - spread.shift(5)
    f["dxy_ret_5d"] = np.log(d / d.shift(5))
    return f[EXTRA_COLUMNS]


def dataset() -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """(features incl. extras, labels, master) for all labelled rows, chronological."""
    master = load_master()
    labelled, _ = build_datasets(master)
    X = labelled[FEATURE_COLUMNS].join(extra_features(master)).dropna()
    y = labelled.loc[X.index, TARGET_COL].astype(int)
    return X, y, master


def holdout_split(n: int) -> Tuple[int, int]:
    """(rows in the training region, first holdout row) - identical to ``src.train.chronological_split``."""
    split = int(n * TRAIN_FRACTION)
    return split, split + EMBARGO


# --------------------------------------------------------------------------- #
# Candidate models (small, conservative grids; no random search on the test side)  #
# --------------------------------------------------------------------------- #
def _logit(C):
    return lambda: Pipeline([("s", StandardScaler()), ("m", LogisticRegression(C=C, max_iter=2000))])


def candidates() -> Dict[str, List[Tuple[str, callable]]]:
    c: Dict[str, List[Tuple[str, callable]]] = {"logistic_regression": [(f"C={C}", _logit(C)) for C in (0.01, 0.1, 1.0)]}
    try:
        import xgboost as xgb

        xgb.XGBClassifier  # raises if only a placeholder module is present
        c["xgboost"] = [(f"depth={d},n={n},lr={lr}", (lambda d=d, n=n, lr=lr: xgb.XGBClassifier(
            max_depth=d, n_estimators=n, learning_rate=lr, subsample=0.8, colsample_bytree=0.8, min_child_weight=5,
            reg_lambda=5.0, tree_method="hist", random_state=RANDOM_STATE, n_jobs=1, eval_metric="logloss")))
            for d, n, lr in ((2, 100, 0.05), (3, 200, 0.03), (2, 300, 0.02))]
    except Exception:  # noqa: BLE001
        print("NOTE: xgboost not installed here - skipped.", file=sys.stderr)
    try:
        import lightgbm as lgb

        lgb.LGBMClassifier  # raises if only a placeholder module is present
        c["lightgbm"] = [(f"leaves={l},n={n},lr={lr}", (lambda l=l, n=n, lr=lr: lgb.LGBMClassifier(
            num_leaves=l, n_estimators=n, learning_rate=lr, subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
            min_child_samples=40, reg_lambda=5.0, random_state=RANDOM_STATE, n_jobs=1, verbose=-1)))
            for l, n, lr in ((4, 100, 0.05), (8, 200, 0.03), (4, 300, 0.02))]
    except Exception:  # noqa: BLE001
        print("NOTE: lightgbm not installed here - skipped (sklearn HistGradientBoosting used as a stand-in).", file=sys.stderr)
        c["hist_gbm_standin"] = [(f"depth={d},n={n},lr={lr}", (lambda d=d, n=n, lr=lr: HistGradientBoostingClassifier(
            max_depth=d, max_iter=n, learning_rate=lr, min_samples_leaf=40, l2_regularization=5.0, random_state=RANDOM_STATE)))
            for d, n, lr in ((2, 100, 0.05), (3, 200, 0.03), (2, 300, 0.02))]
    return c


# --------------------------------------------------------------------------- #
# Walk-forward                                                                #
# --------------------------------------------------------------------------- #
def walk_forward_blocks(n_region: int) -> List[Tuple[int, int, int]]:
    """[(train_end, test_start, test_end)] expanding window, equal consecutive test blocks, EMBARGO rows purged."""
    size = (n_region - MIN_TRAIN - EMBARGO) // N_BLOCKS
    out = []
    for k in range(N_BLOCKS):
        t0 = MIN_TRAIN + EMBARGO + k * size
        out.append((t0 - EMBARGO, t0, t0 + size))
    return out


def walk_forward(make, X: pd.DataFrame, y: pd.Series, cols: List[str], n_region: int) -> Dict[str, object]:
    rows, probs_all, y_all = [], [], []
    for te, a, b in walk_forward_blocks(n_region):
        m = make().fit(X[cols].iloc[:te], y.iloc[:te])
        p = m.predict_proba(X[cols].iloc[a:b])[:, 1]
        yt = y.iloc[a:b]
        pred = (p >= 0.5).astype(int)
        rows.append({"start": X.index[a].date(), "end": X.index[b - 1].date(), "n": b - a, "acc": float((pred == yt).mean()),
                     "always_up": float(yt.mean()), "auc": float(roc_auc_score(yt, p)), "bal_acc": float(balanced_accuracy_score(yt, pred))})
        probs_all.append(p)
        y_all.append(yt.to_numpy())
    pooled_p, pooled_y = np.concatenate(probs_all), np.concatenate(y_all)
    df = pd.DataFrame(rows)
    df["edge"] = df["acc"] - df["always_up"]
    return {"blocks": df, "mean_acc": df["acc"].mean(), "mean_edge": df["edge"].mean(), "mean_auc": df["auc"].mean(),
            "blocks_edge_pos": int((df["edge"] > 0).sum()), "blocks_auc_pos": int((df["auc"] > 0.5).sum()),
            "pooled_acc": float(((pooled_p >= 0.5) == pooled_y).mean()), "pooled_auc": float(roc_auc_score(pooled_y, pooled_p)),
            "pooled_always_up": float(pooled_y.mean())}


def qualifies(r: Dict[str, object]) -> bool:
    """Selection rule fixed in advance: positive mean edge over always-up AND AUC>0.5 on average, each in >= 4 of 6 blocks."""
    return bool(r["mean_edge"] > 0 and r["mean_auc"] > 0.5 and r["blocks_edge_pos"] >= 4 and r["blocks_auc_pos"] >= 4)


# --------------------------------------------------------------------------- #
# Commands                                                                    #
# --------------------------------------------------------------------------- #
def cmd_audit() -> None:
    X, y, master = dataset()
    print(f"master rows {len(master)}  {master.index.min().date()} -> {master.index.max().date()}")
    print(f"duplicate dates in master: {int(master.index.duplicated().sum())} | monotonic increasing: {master.index.is_monotonic_increasing}")
    gaps = master.index.to_series().diff().dt.days.dropna()
    print(f"calendar gaps between consecutive rows: max {int(gaps.max())} days; gaps > 4 days: {int((gaps > 4).sum())}; weekend-dated rows: {int((master.index.dayofweek >= 5).sum())}")
    print(f"labelled rows {len(y)} | NaN/inf in features: {int(np.isinf(X.to_numpy()).sum() + X.isna().sum().sum())} | features {X.shape[1]} (20 base + {len(EXTRA_COLUMNS)} extra)")
    print(f"target {TARGET_COL}: Up/Flat share {y.mean():.3f}; horizon 1 -> labels do not overlap; embargo {EMBARGO} row(s)")
    # Look-ahead check: features recomputed on truncated history must equal the full-history values
    full = pd.concat([engineer_features(master), extra_features(master)], axis=1)
    worst = 0.0
    for cut in (-400, -200, -50):
        part = master.iloc[:cut]
        pf = pd.concat([engineer_features(part), extra_features(part)], axis=1)
        worst = max(worst, float((pf.dropna() - full.loc[pf.dropna().index]).abs().max().max()))
    print(f"look-ahead check (features on truncated history vs. full history): max abs difference {worst:.2e}  -> {'OK' if worst < 1e-9 else 'LEAK!'}")
    n_region, first_test = holdout_split(len(X))
    print(f"train region rows {n_region} ({X.index[0].date()} -> {X.index[n_region - 1].date()}); holdout {len(X) - first_test} rows ({X.index[first_test].date()} -> {X.index[-1].date()})")
    for te, a, b in walk_forward_blocks(n_region):
        print(f"  walk-forward block: train rows [0,{te}) | test {X.index[a].date()} -> {X.index[b - 1].date()} ({b - a} rows)")


def _metrics_row(name: str, y: pd.Series, p: np.ndarray) -> Dict[str, object]:
    m = classification_metrics(y, p)
    m["balanced_accuracy"] = float(balanced_accuracy_score(y, (p >= 0.5).astype(int)))
    m["model"] = name
    return m


def _print_metrics(rows: List[Dict[str, object]]) -> None:
    t = pd.DataFrame([{k: r[k] for k in ("model", "standard_accuracy", "balanced_accuracy", "precision", "recall", "f1", "roc_auc")} for r in rows])
    print(t.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    for r in rows:
        print(f"  confusion [[TN,FP],[FN,TP]] {r['model']:<28} {r['confusion_matrix']}")


def cmd_baseline() -> None:
    """Current 20 features, project-style models, project holdout. For reference only - nothing is selected from this."""
    X, y, _ = dataset()
    n_region, first = holdout_split(len(X))
    Xtr, ytr, Xte, yte = X[FEATURE_COLUMNS].iloc[:n_region], y.iloc[:n_region], X[FEATURE_COLUMNS].iloc[first:], y.iloc[first:]
    print(f"holdout class distribution: Up/Flat {yte.mean():.3f} / Down {1 - yte.mean():.3f}  (n={len(yte)}); train Up/Flat {ytr.mean():.3f}")
    base = baseline_metrics(ytr, yte)
    rows = [{**_metrics_row(f"naive: {k}", yte, np.full(len(yte), float(v["predicts_class"]))), } for k, v in base.items()]
    for name, makers in candidates().items():
        i = 2 if name == "logistic_regression" else 1   # logistic: the project's default C=1; boosted trees: middle setting
        rows.append(_metrics_row(f"{name} ({makers[i][0]})", yte, makers[i][1]().fit(Xtr, ytr).predict_proba(Xte)[:, 1]))
    _print_metrics(rows)


def cmd_walkforward() -> None:
    X, y, _ = dataset()
    n_region, _ = holdout_split(len(X))
    print(f"Walk-forward on the training region only: rows 0..{n_region - 1} ({X.index[0].date()} -> {X.index[n_region - 1].date()}); holdout untouched.\n")
    sets = {"base": FEATURE_COLUMNS, "extended": FEATURE_COLUMNS + EXTRA_COLUMNS}
    summary = []
    for fs, cols in sets.items():
        for model, makers in candidates().items():
            for i, (label, make) in enumerate(makers):
                r = walk_forward(make, X, y, cols, n_region)
                summary.append({"features": fs, "model": model, "param#": i, "setting": label, "mean_acc": r["mean_acc"], "mean_always_up": r["blocks"]["always_up"].mean(),
                                "mean_edge": r["mean_edge"], "blocks_edge>0": f"{r['blocks_edge_pos']}/{N_BLOCKS}", "mean_auc": r["mean_auc"],
                                "blocks_auc>0.5": f"{r['blocks_auc_pos']}/{N_BLOCKS}", "pooled_acc": r["pooled_acc"], "pooled_auc": r["pooled_auc"],
                                "qualifies": qualifies(r)})
                if i == 0 and model == "logistic_regression":
                    print(f"[{fs}] per-block detail for {model} {label}:\n{r['blocks'].to_string(index=False, float_format=lambda v: f'{v:.3f}')}\n")
    s = pd.DataFrame(summary)
    print(s.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    q = s[s["qualifies"]]
    print("\nQUALIFYING candidates (fixed rule: mean edge>0, mean AUC>0.5, each in >=4 of 6 blocks):", "none" if q.empty else "")
    if not q.empty:
        print(q.sort_values("mean_edge", ascending=False).head(3).to_string(index=False, float_format=lambda v: f"{v:.3f}"))


def cmd_final(features: str, model: str, param: int) -> None:
    X, y, _ = dataset()
    n_region, first = holdout_split(len(X))
    cols = FEATURE_COLUMNS if features == "base" else FEATURE_COLUMNS + EXTRA_COLUMNS
    makers = candidates()[model]
    m = makers[param][1]().fit(X[cols].iloc[:n_region], y.iloc[:n_region])
    yte = y.iloc[first:]
    p = m.predict_proba(X[cols].iloc[first:])[:, 1]
    base = baseline_metrics(y.iloc[:n_region], yte)
    sel = _metrics_row(f"{model} [{features}, {makers[param][0]}]", yte, p)
    rows = [_metrics_row(f"naive: {k}", yte, np.full(len(yte), float(v["predicts_class"]))) for k, v in base.items()] + [sel]
    print(f"FINAL holdout {X.index[first].date()} -> {X.index[-1].date()} (n={len(yte)}), evaluated once.")
    _print_metrics(rows)
    gate = validation_gate(sel, base)
    print(f"\nExisting validation gate: {'PASSED' if gate['passed'] else 'FAILED'} (edge over best baseline {gate['edge_over_best_baseline'] * 100:+.2f} pp, checks {gate['checks']})")
    print(f"60% accuracy target reached on the untouched holdout: {'YES' if sel['standard_accuracy'] >= 0.60 else 'NO'} ({sel['standard_accuracy']:.2%})")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    sub = ap.add_subparsers(dest="cmd", required=True)
    for c in ("audit", "baseline", "walkforward"):
        sub.add_parser(c)
    f = sub.add_parser("final")
    f.add_argument("--features", choices=("base", "extended"), required=True)
    f.add_argument("--model", required=True)
    f.add_argument("--param", type=int, required=True)
    a = ap.parse_args(argv)
    {"audit": cmd_audit, "baseline": cmd_baseline, "walkforward": cmd_walkforward}.get(a.cmd, lambda: cmd_final(a.features, a.model, a.param))()
    return 0


if __name__ == "__main__":
    sys.exit(main())
