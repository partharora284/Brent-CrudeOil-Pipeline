import re

import numpy as np

from app.explainer import FEATURE_INFO, plain_explanation
from src.features import FEATURE_COLUMNS

JARGON = re.compile(r"\b(log-odds|shap|rolling|correlation|feature|lgbm|ema|sma|z-score)\b", re.I)


def _example():
    rng = np.random.default_rng(1)
    shap = {f: float(v) for f, v in zip(FEATURE_COLUMNS, rng.normal(0, 0.2, len(FEATURE_COLUMNS)))}
    shap["distance_sma200"], shap["dxy_log_return"] = 0.6, -0.5
    vals = {f: 1.0 for f in FEATURE_COLUMNS}
    vals.update(distance_sma200=1.04, dxy_log_return=0.004, day_of_week=4, month_of_year=10, brent_log_return=-0.012)
    return shap, vals


def test_every_model_input_has_a_friendly_label_and_description():
    assert set(FEATURE_INFO) == set(FEATURE_COLUMNS)
    for f, (label, desc, describe) in FEATURE_INFO.items():
        assert label and desc and describe(1.0)


def test_direction_matches_shap_sign_and_strengths_are_ranked():
    shap, vals = _example()
    p = plain_explanation(shap, vals, base_prob=0.52, final_prob=0.58)
    assert len(p["factors"]) == 6
    for f in p["factors"]:
        assert (f["direction"] == "up") == (shap[f["feature"]] > 0)
        assert f["share"] > 0
    shares = [f["share"] for f in p["factors"]]
    assert shares == sorted(shares, reverse=True)
    assert p["other"]["count"] == len(FEATURE_COLUMNS) - 6


def test_user_facing_text_has_no_technical_jargon():
    shap, vals = _example()
    p = plain_explanation(shap, vals, base_prob=0.52, final_prob=0.58)
    texts = [p["summary"]] + [f[k] for f in p["factors"] for k in ("label", "description", "value_text")]
    for t in texts:
        assert not JARGON.search(t), t


def test_summary_is_faithful_to_probabilities_and_factors():
    shap, vals = _example()
    p = plain_explanation(shap, vals, base_prob=0.52, final_prob=0.58)
    assert "52%" in p["summary"] and "58%" in p["summary"] and "raise" in p["summary"]
    top_up = next(f for f in p["factors"] if f["direction"] == "up")
    top_down = next(f for f in p["factors"] if f["direction"] == "down")
    assert top_up["label"] in p["summary"] and top_down["label"] in p["summary"]
    lower = plain_explanation(shap, vals, base_prob=0.58, final_prob=0.52)
    assert "lower" in lower["summary"]
    same = plain_explanation(shap, vals, base_prob=0.52, final_prob=0.5204)
    assert "almost unchanged" in same["summary"]
