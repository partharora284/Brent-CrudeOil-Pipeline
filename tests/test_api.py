import shutil

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from src.features import FEATURE_COLUMNS, load_recent_context
from src.registry import ModelRegistry
from src.sessions import next_session
from src.store import persist_dataframe


@pytest.fixture()
def client(trained_env, tmp_path, monkeypatch):
    models, db = tmp_path / "models", tmp_path / "s.duckdb"
    shutil.copytree(trained_env["models"], models)
    shutil.copy2(trained_env["db"], db)
    reg = ModelRegistry(models)
    if reg.approved_version() is None:  # synthetic random data may fail the gate; promote only to test API plumbing
        reg.promote(trained_env["meta"]["version"], force=True)
    monkeypatch.setenv("MODEL_DIR", str(models))
    monkeypatch.setenv("FEATURE_STORE_PATH", str(db))
    from app.api import app

    with TestClient(app) as c:
        yield c


def rows(db, n=10):
    ctx = load_recent_context(n, db)
    out = []
    for idx, r in ctx.iterrows():
        d = {"date": str(idx.date())}; d.update({c: float(r[c]) for c in FEATURE_COLUMNS}); out.append(d)
    return out


def test_health(client):
    j = client.get("/health").json()
    assert j["status"] == "ok" and j["model_loaded"] and j["model_version"] and "data_stale" in j


def test_predict_latest_shape_and_no_future_leak(client):
    j = client.get("/predict/latest").json()
    assert {"prediction", "prob_up", "probabilities", "sentiment", "generated_at", "model", "signal_history", "explanation"} <= set(j)
    assert len(j["signal_history"]) == 7 and "forecast_7d" not in j
    assert j["probabilities"]["up"] == pytest.approx(j["prob_up"]) and j["model"]["version"]
    cut = client.get("/predict/latest", params={"as_of": j["signal_history"][2]["date"]}).json()
    assert cut["as_of"] <= j["signal_history"][2]["date"]  # nothing after as_of was used


def test_predict_post_ok_and_order_independent(client, trained_env):
    recs = rows(trained_env["db"])
    a = client.post("/predict", json={"records": recs})
    assert a.status_code == 200
    b = client.post("/predict", json={"records": recs[::-1]})  # dates re-sorted server-side
    assert b.json()["prob_up"] == pytest.approx(a.json()["prob_up"])
    if a.json()["explanation"]["supported"]:
        assert a.json()["explanation"]["matches_prediction"]


@pytest.mark.parametrize("mutate", ["missing", "string", "too_few", "dup_dates"])
def test_predict_rejects_invalid_input(client, trained_env, mutate):
    recs = rows(trained_env["db"])
    if mutate == "missing":
        del recs[3][FEATURE_COLUMNS[0]]
    elif mutate == "string":
        recs[3][FEATURE_COLUMNS[1]] = "abc"
    elif mutate == "too_few":
        recs = recs[:3]
    else:
        recs[1]["date"] = recs[0]["date"]
    assert client.post("/predict", json={"records": recs}).status_code == 422


def test_503_when_no_model(tmp_path, monkeypatch):
    monkeypatch.setenv("MODEL_DIR", str(tmp_path / "empty"))
    monkeypatch.setenv("FEATURE_STORE_PATH", str(tmp_path / "none.duckdb"))
    from app.api import app

    with TestClient(app) as c:
        assert c.get("/health").json()["status"] == "degraded"
        assert c.get("/predict/latest").status_code == 503


def _last_date(client):
    return pd.Timestamp(client.get("/health").json()["data_latest_date"])


def test_old_data_is_reported_as_outdated_and_the_latest_session_is_shown(client):
    j = client.get("/predict/latest").json()  # synthetic data ends in 2021 -> far older than today
    s = j["session"]
    assert s["status"] == "outdated" and not s["is_current_session"] and "not been updated" in s["message"]
    assert s["session_date"] == s["latest_data_date"]            # the latest AVAILABLE session, never "today"
    assert j["as_of"] < s["session_date"]                         # its inputs come from the session BEFORE it
    assert j["data_stale"] is True and j["signal_history"][-1]["session_date"] == s["session_date"]
    assert j["realised_outcome"] in ("Up/Flat", "Down")           # that session has settled, so the result is known


def test_current_session_uses_only_data_up_to_the_previous_session(client, monkeypatch):
    import app.api as api

    last = _last_date(client)
    target = next_session(last)
    monkeypatch.setattr(api, "today_in_market", lambda *a, **k: target)
    j = client.get("/predict/latest").json()
    assert j["session"]["status"] == "current" and j["session"]["is_current_session"] and j["session"]["message"] is None
    assert j["session"]["session_date"] == str(target.date()) and j["as_of"] == str(last.date())
    assert j["realised_outcome"] is None and j["signal_history"][-1]["session_date"] == str(target.date())
    assert j["data_stale"] is False


def test_closed_market_shows_latest_completed_session(client, monkeypatch):
    import app.api as api

    last = _last_date(client)
    monkeypatch.setattr(api, "today_in_market", lambda *a, **k: last)  # "today" is not yet the next session -> closed
    s = client.get("/predict/latest").json()["session"]
    assert s["status"] == "closed" and "closed" in s["message"].lower() and s["session_date"] == str(last.date())


def test_stale_input_gives_no_prediction_and_a_plain_message(client, trained_env, tmp_path):
    import app.api as api

    last = _last_date(client)
    status = pd.DataFrame([{"asset": a, "last_observation": str((last - pd.Timedelta(days=12 if a == "wti" else 0)).date()),
                            "provider_rows_used": 0, "provider_status": "not_used", "provider_message": "", "checked_at": pd.Timestamp.now()}
                           for a in ("brent", "wti", "dxy")])
    persist_dataframe(status, "data_status", api.app.state.db_path, index=False)
    r = client.get("/predict/latest")
    assert r.status_code == 503 and "WTI crude" in r.json()["detail"]
    assert "Traceback" not in r.text


@pytest.fixture()
def unapproved_env(trained_env, tmp_path, monkeypatch):
    """A registry whose only model is REJECTED (nothing approved), as after a failed validation gate."""
    import json

    models, db = tmp_path / "models", tmp_path / "s.duckdb"
    shutil.copytree(trained_env["models"], models)
    shutil.copy2(trained_env["db"], db)
    idx_path = models / "registry" / "registry.json"
    idx = json.loads(idx_path.read_text())
    idx["approved"] = None
    for v in idx["versions"].values():
        v.update(status="rejected", validation_passed=False)
    idx_path.write_text(json.dumps(idx))
    monkeypatch.setenv("MODEL_DIR", str(models))
    monkeypatch.setenv("FEATURE_STORE_PATH", str(db))
    return monkeypatch


def test_strict_mode_gives_one_clear_message_when_no_model_is_approved(unapproved_env):
    unapproved_env.setenv("ALLOW_UNVALIDATED_MODEL", "0")
    from app.api import app

    with TestClient(app) as c:
        r = c.get("/predict/latest")
        assert r.status_code == 503
        assert r.json()["detail"].count("python -m src.train") == 1 and "ALLOW_UNVALIDATED_MODEL" in r.json()["detail"]
        assert c.get("/health").json()["model_loaded"] is False


def test_experimental_mode_serves_rejected_model_and_labels_it_everywhere(unapproved_env):
    unapproved_env.setenv("ALLOW_UNVALIDATED_MODEL", "1")
    from app.api import app

    with TestClient(app) as c:
        j = c.get("/predict/latest").json()
        assert j["experimental"] is True and j["experimental_notice"].startswith("Experimental model")
        assert "not established as reliable" in j["experimental_notice"]
        assert j["explanation"]  # SHAP/plain explanation still produced
        h = c.get("/health").json()
        assert h["model_loaded"] and h["model_experimental"] is True
        assert c.get("/model").json()["experimental"] is True
