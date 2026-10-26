"""
src/train.py  --  LAYER 3: Model Zoo Benchmarking & Temporal Validation
========================================================================

Pipeline
--------
1. Load the labelled feature matrix from the feature store (via features.py).
2. STRICT chronological 80/20 split -- performed BEFORE any tuning, so the
   test set never influences hyper-parameter selection.
3. Benchmarks on the held-out period:
     * "Always Up" naive baseline (what you get for free in a drifting market)
     * Scaled Logistic Regression (simple linear benchmark)
4. Model zoo, all tuned/cross-validated on the TRAINING data only with a rolling-origin
   ``TimeSeriesSplit(n_splits=5, gap=EMBARGO)``:
     * Always-Up and train-majority-class baselines
     * Logistic Regression, Ridge classifier (scores, not calibrated probabilities)
     * XGBoost (RandomizedSearchCV, the original model) and LightGBM (small search)
5. Metrics on the SAME held-out test window for every model: accuracy, precision,
   recall, F1, confusion matrix, ROC-AUC, plus the original Standard Accuracy and
   High-Confidence Trade Accuracy (trade only if P(Up) >= 0.55 or <= 0.45).
6. SELECTION uses mean TRAIN cross-validation accuracy only (the test set never
   picks the model).  A validation GATE then checks the pick against the test
   baselines; only a model that passes is promoted to "approved" in the registry
   (``src/registry.py``).  A failing model is stored as "rejected" and the
   previously approved model keeps serving.

Quant-finance safeguards
------------------------
* **Chronological split, never shuffled**: shuffling lets the model train on
  tomorrow to predict yesterday.
* **Embargo / purge**: each label looks ``HORIZON`` days into the future, so
  the last ``HORIZON`` training labels overlap the start of the test window.
  We (a) skip ``HORIZON`` rows between train and test and (b) pass
  ``gap=HORIZON`` to TimeSeriesSplit so validation folds do not overlap their
  training folds either.  Set ``EMBARGO`` to 0 below to mimic the notebook.
* **Accuracy vs. baseline**: oil drifts, so "Always Up" can score >50%.
  A model only adds value if it beats that number.

Run standalone::

    python -m src.train
"""

from __future__ import annotations

import json
import logging
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import joblib  # noqa: F401
import lightgbm as lgb
import numpy as np
import pandas as pd
import sklearn
import xgboost as xgb
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, TimeSeriesSplit, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.config import FEATURE_STORE_PATH, MODEL_DIR, PROJECT_ROOT, resolve_path  # noqa: F401
from src.features import (
    FEATURE_COLUMNS,
    FEATURE_SCHEMA_HASH,
    HORIZON,
    TARGET_COL,
    TARGET_DEFINITION,
    load_feature_matrix,
    run_feature_pipeline,
)
from src.models import RidgeScoreClassifier
from src.preprocess import FeaturePreprocessor
from src.registry import ModelRegistry
from src.store import upsert_predictions

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
# Configuration                                                               #
# --------------------------------------------------------------------------- #
MODEL_PATH: Path = MODEL_DIR / "best_model.joblib"
PREPROCESSOR_PATH: Path = MODEL_DIR / "preprocessor.joblib"
METRICS_PATH: Path = MODEL_DIR / "metrics.json"

TRAIN_FRACTION = float(os.getenv("TRAIN_FRACTION", "0.8"))
CONF_UPPER = float(os.getenv("CONF_UPPER", "0.55"))
CONF_LOWER = float(os.getenv("CONF_LOWER", "0.45"))
N_SEARCH_ITER = int(os.getenv("N_SEARCH_ITER", "20"))
RANDOM_STATE = int(os.getenv("RANDOM_STATE", "42"))
EMBARGO = HORIZON  # rows skipped between train and test (see module docstring)

MODEL_ORDER = ["logistic_regression", "ridge_classifier", "xgboost", "lightgbm"]

LGBM_PARAM_DIST = {
    "n_estimators": [100, 200, 300],
    "learning_rate": [0.01, 0.05],
    "num_leaves": [7, 15, 31],
    "min_child_samples": [20, 50],
    "subsample": [0.7, 0.9],
    "colsample_bytree": [0.7, 0.9],
    "reg_lambda": [1.0, 10.0],
}
RIDGE_PARAM_GRID = {"ridge__alpha": [1.0, 10.0, 100.0, 1000.0]}

XGB_PARAM_DIST = {
    "n_estimators": [100, 200, 300],
    "learning_rate": [0.01, 0.05, 0.1],
    "max_depth": [3, 4, 5],
    "subsample": [0.7, 0.8, 0.9],
    "colsample_bytree": [0.7, 0.8, 0.9],
    "reg_lambda": [1.0, 5.0, 10.0],
}


# --------------------------------------------------------------------------- #
# Splitting                                                                   #
# --------------------------------------------------------------------------- #
def chronological_split(
    X: pd.DataFrame, y: pd.Series, train_frac: float = TRAIN_FRACTION, embargo: int = EMBARGO
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Strict chronological split: first ``train_frac`` of rows train, the rest test.

    ``embargo`` rows right after the cut are discarded so no training label's
    forward window overlaps a test label's window (purged split).
    """
    split = int(len(X) * train_frac)
    X_train, y_train = X.iloc[:split], y.iloc[:split]
    X_test, y_test = X.iloc[split + embargo :], y.iloc[split + embargo :]
    return X_train, X_test, y_train, y_test


# --------------------------------------------------------------------------- #
# Evaluation                                                                  #
# --------------------------------------------------------------------------- #
def json_safe(obj: Any) -> Any:
    """Recursively convert numpy types and NaN/inf to plain JSON-safe values (NaN -> None)."""
    if isinstance(obj, dict):
        return {str(k): json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [json_safe(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return json_safe(obj.tolist())
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (float, np.floating)):
        return None if (math.isnan(obj) or math.isinf(obj)) else float(obj)
    return obj


def classification_metrics(
    y_true: pd.Series, probs: np.ndarray, upper: float = CONF_UPPER, lower: float = CONF_LOWER
) -> Dict[str, Any]:
    """
    Metrics for P(Up) scores on one test window.

    Threshold 0.5 -> accuracy / precision / recall / F1 (positive class = Up/Flat) and a
    confusion matrix laid out [[TN, FP], [FN, TP]] with Down=0, Up=1.  ROC-AUC is None
    when undefined (single class, or constant scores as for the baselines).
    Also keeps the original high-confidence trade statistics.
    """
    y = np.asarray(y_true)
    pred = (probs >= 0.5).astype(int)
    auc: Optional[float] = None
    if len(np.unique(y)) == 2 and len(np.unique(probs)) > 1:
        auc = float(roc_auc_score(y, probs))
    long_mask, short_mask = probs >= upper, probs <= lower
    trade_mask = long_mask | short_mask
    m: Dict[str, Any] = {
        "standard_accuracy": float(accuracy_score(y, pred)),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred, zero_division=0)),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "roc_auc": auc,
        "confusion_matrix": confusion_matrix(y, pred, labels=[0, 1]).tolist(),
        "n_test_days": int(len(y)),
        "naive_always_up_accuracy": float(y.mean()),
        "n_trades": int(trade_mask.sum()),
        "pct_days_traded": float(trade_mask.mean()),
        "high_confidence_accuracy": float("nan"),
    }
    if trade_mask.any():
        m["high_confidence_accuracy"] = float(accuracy_score(y[trade_mask], long_mask[trade_mask].astype(int)))
    return m


def evaluate_classifier(
    model, X_test: pd.DataFrame, y_test: pd.Series,
    upper: float = CONF_UPPER, lower: float = CONF_LOWER,
) -> Dict[str, Any]:
    """
    Original entry point, now returning the extended metric set (same original keys).

    A "trade" is taken only when the model is decisive:
        P(Up) >= upper  -> go long   (predict 1)
        P(Up) <= lower  -> go short  (predict 0)
        otherwise       -> stay in cash (no prediction scored)
    """
    return classification_metrics(y_test, model.predict_proba(X_test)[:, 1], upper, lower)


def baseline_metrics(y_train: pd.Series, y_test: pd.Series) -> Dict[str, Dict[str, Any]]:
    """Always-Up baseline and train-majority-class baseline, scored on the test window."""
    maj = int(DummyClassifier(strategy="most_frequent").fit(np.zeros((len(y_train), 1)), y_train).predict(np.zeros((1, 1)))[0])
    out = {}
    for name, cls in (("always_up", 1), ("majority_class_train", maj)):
        probs = np.full(len(y_test), 1.0 if cls == 1 else 0.0)
        m = classification_metrics(y_test, probs)
        m["predicts_class"] = cls
        out[name] = m
    return out


# --------------------------------------------------------------------------- #
# Training                                                                    #
# --------------------------------------------------------------------------- #
def train_model(
    X_train: pd.DataFrame, y_train: pd.Series, n_iter: Optional[int] = None, n_splits: int = 5
) -> RandomizedSearchCV:
    """
    Tune an XGBClassifier with RandomizedSearchCV on TRAINING data only.

    ``TimeSeriesSplit(n_splits=5)`` grows the training window fold by fold and
    always validates on the block that comes *after* it (rolling-origin
    evaluation), mimicking how the model would be used live.
    """
    tscv = TimeSeriesSplit(n_splits=n_splits, gap=EMBARGO)
    base = xgb.XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        tree_method="hist",
        random_state=RANDOM_STATE,
        n_jobs=1,  # parallelism handled by the search (n_jobs=-1) below
    )
    search = RandomizedSearchCV(
        estimator=base,
        param_distributions=XGB_PARAM_DIST,
        n_iter=n_iter or N_SEARCH_ITER,
        cv=tscv,
        scoring="accuracy",
        n_jobs=-1,
        random_state=RANDOM_STATE,
        refit=True,  # refit best params on the full training window
    )
    search.fit(X_train, y_train)
    return search


def fit_model_zoo(
    X_train: pd.DataFrame, y_train: pd.Series, n_iter: Optional[int] = None, n_splits: int = 5
) -> Dict[str, Dict[str, Any]]:
    """
    Fit every candidate on TRAINING data only.  Returns {name: {model, cv_accuracy, params}}.

    ``cv_accuracy`` is the mean rolling-origin CV accuracy on the training window
    (for the tuned models: the best candidate's CV score, which is slightly
    optimistic because it is the maximum over the searched settings).
    """
    n_iter = n_iter or N_SEARCH_ITER
    tscv = TimeSeriesSplit(n_splits=n_splits, gap=EMBARGO)
    zoo: Dict[str, Dict[str, Any]] = {}

    logit = Pipeline([("scaler", StandardScaler()), ("logit", LogisticRegression(max_iter=1000))])
    cv = cross_val_score(logit, X_train, y_train, cv=tscv, scoring="accuracy")
    zoo["logistic_regression"] = {"model": logit.fit(X_train, y_train), "cv_accuracy": float(cv.mean()), "params": {}}

    ridge = Pipeline([("scaler", StandardScaler()), ("ridge", RidgeScoreClassifier())])
    g = GridSearchCV(ridge, RIDGE_PARAM_GRID, cv=tscv, scoring="accuracy", n_jobs=1).fit(X_train, y_train)
    zoo["ridge_classifier"] = {"model": g.best_estimator_, "cv_accuracy": float(g.best_score_), "params": g.best_params_}

    print(f"Tuning XGBoost strictly on training data (TimeSeriesSplit, {n_splits} folds)...")
    xs = train_model(X_train, y_train, n_iter=n_iter, n_splits=n_splits)
    zoo["xgboost"] = {"model": xs.best_estimator_, "cv_accuracy": float(xs.best_score_), "params": xs.best_params_}

    print("Tuning LightGBM (small search, training data only)...")
    lgbm = lgb.LGBMClassifier(objective="binary", random_state=RANDOM_STATE, n_jobs=1, verbose=-1,
                              subsample_freq=1)
    ls = RandomizedSearchCV(lgbm, LGBM_PARAM_DIST, n_iter=min(n_iter, 8), cv=tscv, scoring="accuracy",
                            n_jobs=-1, random_state=RANDOM_STATE, refit=True).fit(X_train, y_train)
    zoo["lightgbm"] = {"model": ls.best_estimator_, "cv_accuracy": float(ls.best_score_), "params": ls.best_params_}
    return zoo


def validation_gate(selected: Dict[str, Any], baselines: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    """
    Promotion criteria (applied to the model already chosen by TRAIN cross-validation):
      1. test accuracy strictly greater than BOTH baselines (always-up and train-majority);
      2. test ROC-AUC defined and > 0.5.
    Also reports a naive z-score of the edge over the best baseline.  Labels overlap
    (3-day windows), so the effective sample is smaller than n and the z-score is
    optimistic: a 'pass' means "measured edge over baseline", not statistical proof.
    """
    best_base = max(b["standard_accuracy"] for b in baselines.values())
    n = selected["n_test_days"]
    edge = selected["standard_accuracy"] - best_base
    z = edge / math.sqrt(best_base * (1 - best_base) / n) if n and 0 < best_base < 1 else float("nan")
    auc = selected["roc_auc"]
    checks = {
        "accuracy_beats_all_baselines": bool(edge > 0),
        "roc_auc_above_0.5": bool(auc is not None and auc > 0.5),
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "best_baseline_accuracy": best_base,
        "edge_over_best_baseline": edge,
        "edge_z_score_naive": z,
        "edge_significant_naive_5pct": bool(not math.isnan(z) and z > 1.645),  # informational, not a gate
        "note": ("Overlapping multi-day labels make the effective sample smaller than n_test_days; treat small edges as unproven."
                 if HORIZON > 1 else
                 "Single-session direction is close to a coin flip; a small edge on this many sessions is weak evidence."),
    }


def run_training(
    rebuild_features: bool = False,
    db_path: Optional[str | Path] = None,
    model_dir: Optional[str | Path] = None,
    n_search_iter: Optional[int] = None,
    n_splits: int = 5,
) -> Dict[str, object]:
    """Full training run; returns the metadata that was written to the registry."""
    db_path = Path(db_path) if db_path else Path(FEATURE_STORE_PATH)
    model_dir = Path(model_dir) if model_dir else MODEL_DIR
    if rebuild_features or not db_path.exists():
        logger.info("Building feature store first ...")
        run_feature_pipeline(db_path)

    X, y = load_feature_matrix(db_path)
    X_train, X_test, y_train, y_test = chronological_split(X, y)
    logger.info(
        "Train: %d rows (%s -> %s) | Test: %d rows (%s -> %s)",
        len(X_train), X_train.index.min().date(), X_train.index.max().date(),
        len(X_test), X_test.index.min().date(), X_test.index.max().date(),
    )

    # ---- Baselines --------------------------------------------------------- #
    baselines = baseline_metrics(y_train, y_test)
    print(f"Naive Baseline ('Always Up') Accuracy: {baselines['always_up']['standard_accuracy'] * 100:.2f}%")
    print(f"Majority-class (train) Baseline Accuracy: {baselines['majority_class_train']['standard_accuracy'] * 100:.2f}%")

    # ---- Model zoo (training data only) ------------------------------------ #
    zoo = fit_model_zoo(X_train, y_train, n_iter=n_search_iter, n_splits=n_splits)
    comparison: Dict[str, Dict[str, Any]] = {}
    for name in MODEL_ORDER:
        m = classification_metrics(y_test, zoo[name]["model"].predict_proba(X_test)[:, 1])
        m.update(cv_accuracy=zoo[name]["cv_accuracy"], params=zoo[name]["params"])
        comparison[name] = m
        print(f"{name:<20} CV {m['cv_accuracy']*100:5.2f}% | test acc {m['standard_accuracy']*100:5.2f}% "
              f"| F1 {m['f1']:.3f} | AUC {m['roc_auc'] if m['roc_auc'] is None else round(m['roc_auc'], 3)}")

    # ---- Selection: TRAIN cross-validation only ----------------------------- #
    selected_name = max(MODEL_ORDER, key=lambda n_: zoo[n_]["cv_accuracy"])
    selected_model = zoo[selected_name]["model"]
    sel = comparison[selected_name]
    gate = validation_gate(sel, baselines)
    rationale = (
        f"{selected_name} had the highest mean rolling-origin CV accuracy on the training window "
        f"({zoo[selected_name]['cv_accuracy']:.4f}); CV ranking: "
        + ", ".join(f"{n_}={zoo[n_]['cv_accuracy']:.4f}" for n_ in sorted(MODEL_ORDER, key=lambda k: -zoo[k]['cv_accuracy']))
        + f". Test set was not used for selection. Gate {'PASSED' if gate['passed'] else 'FAILED'}: test accuracy "
        f"{sel['standard_accuracy']:.4f} vs best baseline {gate['best_baseline_accuracy']:.4f}, ROC-AUC {sel['roc_auc']}."
    )

    print("\n--- Final Strategy Performance (out-of-sample, selected model: %s) ---" % selected_name)
    print(f"Best CV Accuracy (train folds):         {zoo[selected_name]['cv_accuracy'] * 100:.2f}%")
    print(f"Standard Accuracy (Trading Every Day):  {sel['standard_accuracy'] * 100:.2f}%")
    if sel["n_trades"] > 0:
        print(f"High-Confidence Trade Accuracy:         {sel['high_confidence_accuracy'] * 100:.2f}%")
        print(f"Percentage of Days Traded:              {sel['pct_days_traded'] * 100:.2f}%")
    else:
        print("No predictions met the confidence threshold.")
    print(f"Validation gate: {'PASSED' if gate['passed'] else 'FAILED'}  (edge over best baseline {gate['edge_over_best_baseline']*100:+.2f} pp)")

    # ---- Legacy flat metrics.json keys (kept for backwards compatibility) --- #
    legacy = {k: sel[k] for k in ("standard_accuracy", "naive_always_up_accuracy", "n_test_days", "n_trades",
                                  "pct_days_traded", "high_confidence_accuracy")}
    legacy.update(
        logistic_regression_accuracy=comparison["logistic_regression"]["standard_accuracy"],
        cv_best_accuracy=zoo[selected_name]["cv_accuracy"],
        best_params=zoo[selected_name]["params"],
        horizon_days=HORIZON,
        selected_model=selected_name,
        train_start=str(X_train.index.min().date()), train_end=str(X_train.index.max().date()),
        test_start=str(X_test.index.min().date()), test_end=str(X_test.index.max().date()),
    )

    pre = FeaturePreprocessor(FEATURE_COLUMNS, FEATURE_SCHEMA_HASH).fit(X_train)
    metadata = json_safe({
        "model_name": selected_name,
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "feature_schema": {"columns": FEATURE_COLUMNS, "hash": FEATURE_SCHEMA_HASH},
        "target": {"column": TARGET_COL, "horizon_days": HORIZON, "definition": TARGET_DEFINITION},
        "data": {
            "train_start": legacy["train_start"], "train_end": legacy["train_end"],
            "test_start": legacy["test_start"], "test_end": legacy["test_end"],
            "n_train": len(X_train), "n_test": len(X_test), "embargo_rows": EMBARGO,
            "train_fraction": TRAIN_FRACTION, "final_fit": "train window only (deployed model is the validated one)",
        },
        "thresholds": {"conf_upper": CONF_UPPER, "conf_lower": CONF_LOWER},
        "metrics": sel,
        "comparison": comparison,
        "baselines": baselines,
        "selection": {"criterion": "max mean TimeSeriesSplit CV accuracy on training window", "rationale": rationale,
                      "cv_scores": {n_: zoo[n_]["cv_accuracy"] for n_ in MODEL_ORDER}},
        "validation": gate,
        "preprocessing": "FeaturePreprocessor (schema/order/finite checks); scaling, if any, is inside the model Pipeline.",
        "explainability": "TreeSHAP" if selected_name in ("xgboost", "lightgbm") else "not supported for this model type",
        "libraries": {"sklearn": sklearn.__version__, "xgboost": xgb.__version__, "lightgbm": lgb.__version__,
                      "pandas": pd.__version__, "numpy": np.__version__},
        "legacy_metrics": legacy,
    })

    # ---- Registry: store the version, promote ONLY if the gate passed -------- #
    registry = ModelRegistry(model_dir)
    version = registry.register(selected_model, pre, metadata)
    if gate["passed"]:
        registry.promote(version)
        probs = selected_model.predict_proba(X_test)[:, 1]
        now = datetime.now(timezone.utc).replace(tzinfo=None)
        upsert_predictions(
            pd.DataFrame({
                "date": X_test.index, "model_version": version, "split": "test", "prob_up": probs,
                "sentiment": np.where(probs >= CONF_UPPER, "BULL", np.where(probs <= CONF_LOWER, "BEAR", "NEUTRAL")),
                "actual": y_test.to_numpy(), "generated_at": now,
            }),
            db_path, replace_split="test",
        )
        print(f"\nRegistered and APPROVED {version} ({selected_name}) -> {registry.root}")
        print(f"Mirrored to {model_dir / 'best_model.joblib'} and preprocessor.joblib")
    else:
        print(f"\nRegistered {version} as REJECTED (failed the validation gate). "
              f"Approved model unchanged: {registry.approved_version()}")
        print("It will NOT be served unless you set ALLOW_UNVALIDATED_MODEL=1 in .env (experimental demo mode, clearly labelled).")
    metadata["version"] = version
    return metadata


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    run_training(rebuild_features=False)
