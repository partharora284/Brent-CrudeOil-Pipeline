"""Dashboard checks: removed UI elements stay removed, and the page renders from a mocked API."""

from pathlib import Path

import pytest

DASH = Path(__file__).resolve().parents[1] / "app" / "dashboard.py"
REMOVED = ["Run live inference", "API URL", "file_uploader", "st.sidebar", "Data is stale", "Data fresh",
           "Model available", "Model version", "Generated {", "trained {", "waterfall_chart"]


def test_removed_elements_are_gone_from_the_dashboard_source():
    src = DASH.read_text(encoding="utf-8")
    for text in REMOVED:
        assert text not in src, text
    assert "Settings" not in src and "version" not in src.lower().replace("versions", "")


def _payload(status="current", message=None, experimental=False):
    pt = lambda d, s, p, tag, out: {"date": d, "session_date": s, "prob_up": p, "sentiment": tag, "realised_outcome": out, "in_training_period": False}
    factors = [
        {"feature": "distance_sma200", "label": "Brent vs. its 1-year average", "description": "Compares Brent with its long-term average.",
         "value_text": "Brent is 4.0% above its 200-day average", "direction": "up", "strength": "strong", "share": 0.31, "shap": 0.4},
        {"feature": "dxy_log_return", "label": "US dollar's latest daily move", "description": "Dollar change.",
         "value_text": "the dollar index moved +0.40% in the last session", "direction": "down", "strength": "moderate", "share": 0.12, "shap": -0.15},
    ]
    return {
        "as_of": "2026-10-22", "generated_at": "x", "horizon_days": 1, "prediction": "Up/Flat", "prob_up": 0.58,
        "probabilities": {"down": 0.42, "up": 0.58}, "sentiment": "BULL", "avg_prob_7d": 0.55, "n_rows_used": 60,
        "realised_outcome": None if status == "current" else "Up/Flat",
        "session": {"status": status, "session_date": "2026-10-23", "input_date": "2026-10-22", "is_current_session": status == "current",
                    "message": message, "latest_data_date": "2026-10-22", "next_session": "2026-10-26"},
        "experimental": experimental,
        "experimental_notice": ("Experimental model \u2014 this model did not pass the required validation and is not established as reliable."
                                if experimental else None),
        "model": {"name": "lightgbm", "version": "v0009", "trained_at": "2026-10-24"},
        "signal_history": [pt("2026-10-21", "2026-10-22", 0.5, "NEUTRAL", "Up/Flat"), pt("2026-10-22", "2026-10-23", 0.58, "BULL", None)],
        "explanation": {"supported": True, "matches_prediction": True,
                        "plain": {"base_prob": 0.52, "final_prob": 0.58, "factors": factors,
                                  "other": {"count": 16, "net_direction": "up", "share": 0.2}, "summary": "Plain summary."},
                        "feature_importance": [{"feature": "distance_sma200", "label": "Brent vs. its 1-year average", "shap": 0.4, "abs_shap": 0.4, "value": 1.04}],
                        "global_importance": [{"feature": "distance_sma200", "label": "Brent vs. its 1-year average", "mean_abs_shap": 0.3}]},
    }


class FakeResp:
    def __init__(self, status, body):
        self.status_code, self._body, self.text = status, body, str(body)

    def json(self):
        return self._body


def _run(monkeypatch, status_code, body):
    st_testing = pytest.importorskip("streamlit.testing.v1")
    import requests
    import streamlit as st

    st.cache_data.clear()
    monkeypatch.setattr(requests, "get", lambda url, **kw: FakeResp(status_code, body) if url.endswith("/predict/latest") else FakeResp(404, {}))
    return st_testing.AppTest.from_file(str(DASH), default_timeout=60).run()


def _all_text(at):
    parts = [e.value for e in list(at.markdown) + list(at.caption) + list(at.info) + list(at.warning) + list(at.error)]
    parts += [f"{m.label} {m.value}" for m in at.metric]
    return "\n".join(parts)


def test_dashboard_renders_current_session_without_settings_or_status_cards(monkeypatch):
    at = _run(monkeypatch, 200, _payload())
    assert not at.exception
    assert not at.text_input and not at.button                   # no Settings controls
    text = _all_text(at)
    for gone in ("Run live inference", "Model available", "Data is stale", "Generated", "v0009"):
        assert gone not in text
    assert "Today's session" in text and "Market data up to" in text and "Plain summary." in text
    assert "Pushing the prediction UP" in text and "Pulling the prediction DOWN" in text


def test_dashboard_shows_simple_message_when_data_is_outdated(monkeypatch):
    msg = "Market data has not been updated since Mon 12 Oct 2026, so a prediction for today isn't available."
    at = _run(monkeypatch, 200, _payload("outdated", msg))
    assert not at.exception and any(msg in w.value for w in at.warning)
    assert "Latest session shown" in _all_text(at) and "What actually happened" in _all_text(at)


def test_dashboard_shows_the_reason_when_no_prediction_is_possible(monkeypatch):
    at = _run(monkeypatch, 503, {"detail": "There isn't enough up-to-date market data for a reliable prediction right now: WTI crude."})
    assert not at.exception and any("London Gas Oil" in w.value for w in at.warning)


def test_experimental_warning_is_shown_when_the_model_is_experimental(monkeypatch):
    at = _run(monkeypatch, 200, _payload(experimental=True))
    assert not at.exception
    assert any("did not pass the required validation" in e.value for e in at.error)
    assert "Plain summary." in _all_text(at)  # the full prediction/SHAP view still renders


def test_no_experimental_warning_for_a_validated_model(monkeypatch):
    at = _run(monkeypatch, 200, _payload())
    assert not any("Experimental" in e.value for e in at.error)


def test_an_api_error_is_shown_once_without_duplication(monkeypatch):
    msg = "No validated prediction model is available yet. Run `python -m src.train`, then restart the API."
    at = _run(monkeypatch, 503, {"detail": msg})
    shown = [w.value for w in at.warning]
    assert shown == [msg] and not at.error
