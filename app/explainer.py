"""
app/explainer.py  --  LAYER 3/4: TreeSHAP Explainability Engine
================================================================

Explains *why* the model gave today's P(Up).

Why TreeSHAP?
-------------
SHAP values split a prediction into additive per-feature contributions:

    log-odds(Up) = base_value + sum_i shap_i

* ``base_value``  : the model's average output over the training data
                    (its "prior" before looking at any feature).
* ``shap_i``      : how much feature *i* pushed the call towards Up (+) or Down (-).

TreeSHAP computes these *exactly* and quickly for tree ensembles such as
XGBoost.  For a trading desk this is the difference between "the model says
BULL" and "the model says BULL because Brent sits far above its 200-day SMA
and the 14-day DXY correlation is strongly negative".

The output of :func:`explain_prediction` is shaped for a waterfall chart:
a starting bar (base value), one step per top feature, an "other features"
step, and a final total.  Values are in log-odds (the space where SHAP is
exactly additive); the probability equivalent is supplied for display.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from src.config import PROJECT_ROOT  # noqa: F401  (ensures project root is importable)
from src.features import FEATURE_COLUMNS

logger = logging.getLogger(__name__)

_EXPLAINER_CACHE: Dict[int, tuple] = {}

SHAP_NOTE = (
    "SHAP values show how much each feature contributed to this model's output for this row "
    "(relative to the model's average output). They describe the model's behaviour, not market "
    "causality, and do not imply that changing a feature would change the market."
)


def tree_estimator(model):
    """Return the underlying tree model (XGBoost / LightGBM) or None if SHAP TreeExplainer can't be used."""
    est = model.steps[-1][1] if hasattr(model, "steps") else model
    return est if type(est).__module__.split(".")[0] in ("xgboost", "lightgbm") else None


def _sigmoid(x: float) -> float:
    """Map log-odds to probability."""
    return float(1.0 / (1.0 + np.exp(-x)))


def _get_shap_explainer(model):
    """Build (and cache) a shap.TreeExplainer for the given model."""
    key = id(model)
    if key not in _EXPLAINER_CACHE or _EXPLAINER_CACHE[key][0] is not model:  # keep a ref so id() stays unique
        import shap  # imported lazily so the API can start even if shap is missing

        _EXPLAINER_CACHE[key] = (model, shap.TreeExplainer(model))
    return _EXPLAINER_CACHE[key][1]


def compute_shap_values(model, X: pd.DataFrame) -> tuple[np.ndarray, float]:
    """
    Return (shap_matrix[n_rows, n_features], base_value) in log-odds space.

    Primary path: the ``shap`` library's TreeExplainer.
    Fallback: XGBoost's built-in ``pred_contribs`` -- the same TreeSHAP
    algorithm, with no extra dependency -- used if the installed shap and
    xgboost versions are incompatible.
    """
    try:
        explainer = _get_shap_explainer(model)
        values = explainer.shap_values(X)
        values = np.asarray(values[1] if isinstance(values, list) else values)
        base = np.asarray(explainer.expected_value).reshape(-1)
        return values, float(base[-1])
    except Exception as exc:  # pragma: no cover - defensive fallback
        if type(model).__module__.split(".")[0] != "xgboost":
            raise
        logger.warning("shap.TreeExplainer failed (%s); using XGBoost native TreeSHAP.", exc)
        import xgboost as xgb

        contribs = model.get_booster().predict(xgb.DMatrix(X), pred_contribs=True)
        return contribs[:, :-1], float(contribs[0, -1])


# --------------------------------------------------------------------------- #
# Plain-English layer (for non-technical readers)                             #
# --------------------------------------------------------------------------- #
# technical name -> (friendly label, one-sentence explanation, function turning today's value into words)
def _pct(v: float) -> float:
    return (float(np.exp(v)) - 1.0) * 100.0


def _vs_avg(v: float, what: str) -> str:
    d = (v - 1.0) * 100.0
    return f"Brent is {abs(d):.1f}% {'above' if d >= 0 else 'below'} {what}"


def _link(v: float) -> str:
    kind = "in opposite directions" if v < -0.3 else "in the same direction" if v > 0.3 else "with little connection"
    return f"over the last 14 sessions they mostly moved {kind}"


_DAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
_MONTHS = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October",
           "November", "December"]

FEATURE_INFO: Dict[str, tuple] = {
    "spread_brent_wti": ("Brent vs. US crude (WTI) price gap",
                         "How much more Brent costs than US crude. It shifts with shipping costs and regional supply.",
                         lambda v: f"Brent is ${abs(v):.2f} {'above' if v >= 0 else 'below'} WTI"),
    "spread_brent_dxy": ("Oil price vs. the US dollar",
                         "Whether oil looks expensive or cheap compared with the dollar, each measured against its own past-year average.",
                         lambda v: f"oil looks {'expensive' if v >= 0 else 'cheap'} relative to the dollar compared with the past year"),
    "brent_log_return": ("Brent's latest daily move", "How much Brent's price changed in the last completed session.",
                         lambda v: f"Brent moved {_pct(v):+.2f}% in the last session"),
    "wti_log_return": ("US crude's (WTI) latest daily move", "How much WTI's price changed in the last completed session.",
                       lambda v: f"WTI moved {_pct(v):+.2f}% in the last session"),
    "dxy_log_return": ("US dollar's latest daily move", "How much the US Dollar Index changed in the last completed session.",
                       lambda v: f"the dollar index moved {_pct(v):+.2f}% in the last session"),
    "rolling_corr_brent_dxy_14d": ("Oil-dollar link (last 2 weeks)",
                                   "How closely Brent and the US dollar have moved together recently. Oil and the dollar often move in opposite directions.",
                                   lambda v: _link(v)),
    "rolling_corr_brent_wti_14d": ("Brent-US crude link (last 2 weeks)",
                                   "How closely Brent and WTI have moved together recently. A weaker link can signal unusual market stress.",
                                   lambda v: _link(v)),
    "brent_volatility_7d": ("Price swings (last week)", "How jumpy Brent's daily moves have been over the last 7 sessions.",
                            lambda v: f"Brent's typical daily swing over the last 7 sessions is about {v * 100:.2f}%"),
    "brent_volatility_30d": ("Price swings (last month)", "How jumpy Brent's daily moves have been over the last 30 sessions.",
                             lambda v: f"Brent's typical daily swing over the last 30 sessions is about {v * 100:.2f}%"),
    "high_low_spread_ratio": ("Size of the last day's trading range",
                              "The gap between the highest and lowest price in the last session. A wide range means a nervous market.",
                              lambda v: f"the last session's high-to-low range was {v * 100:.1f}% of the price"),
    "distance_ema20": ("Brent vs. its 1-month trend", "Compares Brent with its recent (about 20-day) average, a short-term trend gauge.",
                       lambda v: _vs_avg(v, "its ~20-day average")),
    "distance_sma50": ("Brent vs. its 3-month average", "Compares Brent with its average over the last 50 sessions.",
                       lambda v: _vs_avg(v, "its 50-day average")),
    "distance_sma200": ("Brent vs. its 1-year average", "Compares Brent with its average over the last 200 sessions, a long-term trend gauge.",
                        lambda v: _vs_avg(v, "its 200-day average")),
    "brent_return_lag1": ("Brent's move one session earlier", "Brent's price change in the session before the latest one (momentum).",
                          lambda v: f"Brent moved {_pct(v):+.2f}% then"),
    "brent_return_lag2": ("Brent's move two sessions earlier", "Brent's price change two sessions before the latest one (momentum).",
                          lambda v: f"Brent moved {_pct(v):+.2f}% then"),
    "brent_return_lag3": ("Brent's move three sessions earlier", "Brent's price change three sessions before the latest one (momentum).",
                          lambda v: f"Brent moved {_pct(v):+.2f}% then"),
    "wti_return_lag1": ("US crude's move one session earlier", "WTI's price change in the session before the latest one.",
                        lambda v: f"WTI moved {_pct(v):+.2f}% then"),
    "dxy_return_lag1": ("US dollar's move one session earlier", "The dollar index's change in the session before the latest one.",
                        lambda v: f"the dollar index moved {_pct(v):+.2f}% then"),
    "day_of_week": ("Day of the week", "Some weekdays have tended to behave differently from others.",
                    lambda v: f"the last completed session was a {_DAYS[int(round(v)) % 7]}"),
    "month_of_year": ("Time of year", "Some months have tended to behave differently from others (seasonal effects).",
                      lambda v: f"the last completed session was in {_MONTHS[(int(round(v)) - 1) % 12]}"),
}

TOP_FACTORS = 6


def _strength(share: float) -> str:
    return "strong" if share >= 0.20 else "moderate" if share >= 0.08 else "slight"


def plain_explanation(shap_by_feature: Dict[str, float], values: Dict[str, float], base_prob: float,
                      final_prob: float, top_k: int = TOP_FACTORS) -> Dict[str, object]:
    """
    Re-express the model's real SHAP values in everyday language.  Nothing is estimated or invented:

    * direction  = the sign of the feature's SHAP value (positive pushes toward Up/Flat);
    * strength   = the feature's share of the total push, |shap_i| / sum_j |shap_j|
                   (strong >= 20%, moderate >= 8%, otherwise slight);
    * base/final = the model's average chance of an up day and its chance for this session.
    """
    total = float(sum(abs(v) for v in shap_by_feature.values())) or 1.0
    order = sorted(shap_by_feature, key=lambda f: abs(shap_by_feature[f]), reverse=True)
    factors = []
    for f in order[:top_k]:
        sv = float(shap_by_feature[f])
        if sv == 0.0 or f not in FEATURE_INFO:
            continue
        label, desc, describe = FEATURE_INFO[f]
        share = abs(sv) / total
        factors.append({"feature": f, "label": label, "description": desc, "value_text": describe(float(values[f])),
                        "direction": "up" if sv > 0 else "down", "strength": _strength(share), "share": share, "shap": sv})
    rest = order[top_k:]
    rest_net = float(sum(shap_by_feature[f] for f in rest))
    other = {"count": len(rest), "net_direction": "up" if rest_net > 0 else "down" if rest_net < 0 else "neutral",
             "share": float(sum(abs(shap_by_feature[f]) for f in rest) / total)}

    ups = [x for x in factors if x["direction"] == "up"]
    downs = [x for x in factors if x["direction"] == "down"]

    def names(items):
        ls = [x["label"] for x in items[:3]]
        return ls[0] if len(ls) == 1 else ", ".join(ls[:-1]) + " and " + ls[-1]

    move = final_prob - base_prob
    sentence = [f"The model's usual starting point is a {base_prob:.0%} chance of an up-or-flat session. "
                + (f"Today's conditions {'raise' if move > 0 else 'lower'} it to {final_prob:.0%}." if abs(move) >= 0.01
                   else f"Today's conditions leave it almost unchanged at {final_prob:.0%}.")]
    if ups:
        sentence.append(f"Pushing it up: {names(ups)}.")
    if downs:
        sentence.append(f"Pulling it down: {names(downs)}.")
    if not ups and not downs:
        sentence.append("No single factor stands out.")
    return {"base_prob": float(base_prob), "final_prob": float(final_prob), "factors": factors, "other": other,
            "summary": " ".join(sentence)}


def explain_prediction(
    model, context: pd.DataFrame, top_k: int = 10, preprocessor: Optional[object] = None
) -> Dict[str, object]:
    """
    Explain the prediction for the LATEST row of ``context`` (local) and summarise the
    whole lookback slice (global mean |SHAP|).

    Parameters
    ----------
    model        : fitted model (XGBoost / LightGBM, optionally inside a Pipeline).
    context      : DataFrame of feature rows; the last row is the one explained.
    preprocessor : the registry's fitted FeaturePreprocessor, so the SHAP input has exactly
                   the column order the model was trained on.

    Returns a dict with ``supported`` (bool).  When True: base/final value (log-odds) and
    probability, waterfall ``steps``, local ``feature_importance`` and
    ``global_importance`` (mean |SHAP| over the lookback slice).  When False (e.g. linear
    models) only a ``message`` is returned - never an error.
    """
    base_info = {"note": SHAP_NOTE}
    est = tree_estimator(model)
    if est is None:
        return {"supported": False,
                "message": f"TreeSHAP is not available for {type(model.steps[-1][1] if hasattr(model, 'steps') else model).__name__}; "
                           "explanations are only provided for tree models (XGBoost, LightGBM).",
                **base_info}
    X_all = (preprocessor.transform(context) if preprocessor is not None
             else context[FEATURE_COLUMNS].astype(float))
    shap_all, base_value = compute_shap_values(est, X_all)
    X_last = X_all.tail(1)
    shap_row = shap_all[-1]
    global_imp = pd.Series(np.abs(shap_all).mean(axis=0), index=X_all.columns).sort_values(ascending=False)

    contrib = pd.Series(shap_row, index=X_all.columns)
    order = contrib.abs().sort_values(ascending=False).index.tolist()
    top, rest = order[:top_k], order[top_k:]

    # Waterfall steps run from the smallest to the largest effect so the biggest
    # drivers sit next to the final total, the usual SHAP layout.
    steps: List[Dict[str, object]] = []
    running = base_value
    if rest:
        other = float(contrib[rest].sum())
        steps.append(
            {
                "label": f"Other {len(rest)} features",
                "feature": "other",
                "feature_value": None,
                "shap": other,
                "start": running,
                "end": running + other,
            }
        )
        running += other
    for feat in reversed(top):
        val = float(contrib[feat])
        steps.append(
            {
                "label": f"{feat} = {float(X_last[feat].iloc[0]):.4g}",
                "feature": feat,
                "feature_value": float(X_last[feat].iloc[0]),
                "shap": val,
                "start": running,
                "end": running + val,
            }
        )
        running += val

    final_value = base_value + float(contrib.sum())
    return {
        "supported": True,
        "note": SHAP_NOTE,
        "global_importance": [{"feature": f, "label": FEATURE_INFO.get(f, (f,))[0], "mean_abs_shap": float(v)}
                              for f, v in global_imp.items()],
        "n_rows_global": int(len(X_all)),
        "as_of": str(X_last.index[0].date()) if hasattr(X_last.index[0], "date") else str(X_last.index[0]),
        "base_value": float(base_value),
        "final_value": float(final_value),
        "base_prob": _sigmoid(base_value),
        "final_prob": _sigmoid(final_value),
        "steps": steps,
        "feature_importance": [
            {"feature": f, "label": FEATURE_INFO.get(f, (f,))[0], "shap": float(contrib[f]), "abs_shap": float(abs(contrib[f])),
             "value": float(X_last[f].iloc[0])}
            for f in order
        ],
        "plain": plain_explanation({f: float(contrib[f]) for f in order}, {f: float(X_last[f].iloc[0]) for f in order},
                                   _sigmoid(base_value), _sigmoid(final_value)),
    }
