"""
app/api.py  --  LAYER 4: FastAPI Application Server & Online Inference Engine
==============================================================================

Endpoints
---------
GET  /health          liveness, model availability, data freshness
GET  /model           approved model metadata: version, schema, target, metrics, baselines
GET  /model/versions  every registered version and its status
POST /model/reload    re-read the approved model from the registry (after promote/rollback)
POST /predict         body = lookback slice of FEATURE rows (JSON, validated by Pydantic)
GET  /predict/latest  prediction for the CURRENT trading session from the latest feature-store data
                      (optional ?as_of=YYYY-MM-DD replays using only rows up to that date)

Response (both predict endpoints)
---------------------------------
prediction ("Up/Flat" | "Down" at 0.5), prob_up / probabilities, sentiment (BULL/BEAR/NEUTRAL),
as_of (the last completed session whose data was used), ``session`` (which session the call is
about and whether it is the CURRENT one - see src/sessions.py), generated_at, model, data_latest_date,
``signal_history`` (the model's call for each of the last 7 sessions, with the realised outcome once
known) and ``explanation`` (TreeSHAP + a plain-English version, when supported by the model).

What the prediction means
-------------------------
The model answers one question: "will Brent's settlement on session S be at or above the previous
settlement?".  It is made BEFORE S settles from data up to the last COMPLETED session (horizon = 1).
It does not use intraday prices.  If today is not a trading day, or the data is not up to date, the
response shows the latest available session instead and ``session.message`` says so; if an input
is stale there is no prediction at all (HTTP 503 with a plain-English reason).

Why the request carries FEATURE rows and not raw prices
-------------------------------------------------------
SMA-200 needs 200 sessions and the Brent-DXY z-score 252, so a 60-session raw price
slice cannot reproduce the features; the slice is made of already-engineered rows.

Experimental mode
-----------------
``ALLOW_UNVALIDATED_MODEL=1`` (default 0) lets the API serve the newest REJECTED model when none is approved,
but only if it is complete, loadable and trained for the current ``target_1d_class``.  Every response then
carries ``experimental: true`` and ``experimental_notice``; /health and /model say so too.

Run::

    uvicorn app.api:app --port 8000
"""

from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager
from datetime import date as date_type
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, ConfigDict, Field, create_model

from app.explainer import explain_prediction
from src.config import FEATURE_STORE_PATH, MODEL_DIR, resolve_path
from src.features import FEATURE_COLUMNS, HORIZON, LOOKBACK_DAYS, load_labels, load_recent_context
from src.registry import LoadedModel, ModelRegistry, RegistryError
from src.sessions import assess_session, next_session, today_in_market
from src.store import read_table, upsert_predictions

logger = logging.getLogger("brent_api")

EXPERIMENTAL_NOTICE = ("Experimental model — this model did not pass the required validation and is not established as reliable.")


def allow_unvalidated() -> bool:
    """Read at (re)load time so the setting can be changed in .env and picked up on restart."""
    return os.getenv("ALLOW_UNVALIDATED_MODEL", "0").strip().lower() in ("1", "true", "yes", "on")

CONF_UPPER = float(os.getenv("CONF_UPPER", "0.55"))
CONF_LOWER = float(os.getenv("CONF_LOWER", "0.45"))
STALE_DAYS = int(os.getenv("STALE_DAYS", "4"))  # calendar days; covers a weekend + one holiday
MIN_ROWS = 7           # need at least 7 sessions for the 7-session signal history
HISTORY_SESSIONS = 7
MAX_ROWS = 500


# --------------------------------------------------------------------------- #
# Schemas                                                                     #
# --------------------------------------------------------------------------- #
# One float field per model feature (finite numbers only), generated from the
# single source of truth FEATURE_COLUMNS so request and model schema cannot drift.
FeatureRow = create_model(
    "FeatureRow",
    __config__=ConfigDict(extra="ignore"),
    date=(Optional[date_type], Field(None, description="Session date YYYY-MM-DD (recommended; rows are sorted by it)")),
    **{c: (float, Field(..., allow_inf_nan=False)) for c in FEATURE_COLUMNS},
)


class PredictRequest(BaseModel):
    """A lookback slice: list of feature rows, oldest first, newest last."""

    records: List[FeatureRow] = Field(  # type: ignore[valid-type]
        ...,
        min_length=MIN_ROWS,
        max_length=MAX_ROWS,
        description=(
            f"Feature rows (oldest -> newest). Each needs the {len(FEATURE_COLUMNS)} model features "
            f"and may include a 'date'. Up to the last {LOOKBACK_DAYS} rows are used."
        ),
    )


class SessionInfo(BaseModel):
    status: str = Field(..., description="current | closed | outdated | replay")
    session_date: str = Field(..., description="The trading session this prediction is about")
    input_date: str = Field(..., description="Last completed session whose data the model used")
    is_current_session: bool
    message: Optional[str] = Field(None, description="Plain-English note shown to the user when status is not 'current'")
    latest_data_date: str
    next_session: Optional[str] = None


class SignalPoint(BaseModel):
    date: str = Field(..., description="Session whose data the signal was made from (the signal is issued after it closes)")
    session_date: Optional[str] = Field(None, description="The session the signal is about (the next one after `date`)")
    prob_up: float
    sentiment: str
    realised_outcome: Optional[str] = Field(None, description="'Up/Flat' or 'Down' once the horizon has elapsed, else null")
    in_training_period: Optional[bool] = Field(None, description="True if this session is inside the model's training window")


class ModelRef(BaseModel):
    name: str
    version: str
    trained_at: str


class PredictResponse(BaseModel):
    as_of: str
    generated_at: str
    horizon_days: int
    prediction: str
    prob_up: float
    probabilities: Dict[str, float]
    sentiment: str
    avg_prob_7d: float
    n_rows_used: int
    experimental: bool = Field(False, description="True when the model did NOT pass validation (ALLOW_UNVALIDATED_MODEL=1)")
    experimental_notice: Optional[str] = None
    session: Optional[SessionInfo] = None
    realised_outcome: Optional[str] = Field(None, description="Actual result of the predicted session, once it has settled")
    model: ModelRef
    data_latest_date: Optional[str] = None
    data_age_days: Optional[int] = None
    data_stale: Optional[bool] = None
    signal_history: List[SignalPoint]
    explanation: Dict[str, Any]


# --------------------------------------------------------------------------- #
# Online inference engine                                                     #
# --------------------------------------------------------------------------- #
def sentiment_tag(prob_up: float) -> str:
    """
    Map P(Up) to a trading-desk tag using the same thresholds as training.

    BULL  : P(Up) >= 0.55   (model is confident of Up/Flat)
    BEAR  : P(Up) <= 0.45   (model is confident of Down)
    NEUTRAL: in between     (no-trade / cash zone)
    """
    if prob_up >= CONF_UPPER:
        return "BULL"
    if prob_up <= CONF_LOWER:
        return "BEAR"
    return "NEUTRAL"


def _freshness(latest: Optional[str]) -> Dict[str, Any]:
    if not latest:
        return {"data_latest_date": None, "data_age_days": None, "data_stale": None}
    age = (pd.Timestamp.today().normalize() - pd.Timestamp(latest)).days
    return {"data_latest_date": latest, "data_age_days": int(age), "data_stale": bool(age > STALE_DAYS)}


def run_inference(
    loaded: LoadedModel,
    context: pd.DataFrame,
    labels: Optional[pd.Series] = None,
    data_latest: Optional[str] = None,
    session: Optional[Dict[str, Any]] = None,
    session_dates: Optional[Dict[str, str]] = None,
) -> PredictResponse:
    """Score every row of the lookback slice with the approved model and package the call for the NEWEST row."""
    context = context.tail(LOOKBACK_DAYS)
    try:
        probs = loaded.predict_proba_up(context)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))

    train_end = loaded.metadata.get("data", {}).get("train_end")
    tail = context.tail(HISTORY_SESSIONS)
    tail_probs = probs[-HISTORY_SESSIONS:]
    history = []
    for idx, p in zip(tail.index, tail_probs):
        outcome = None
        if labels is not None and str(idx) in labels.index:
            outcome = "Up/Flat" if int(labels[str(idx)]) == 1 else "Down"
        in_train = bool(str(idx) <= train_end) if (train_end and str(idx)[:2] == "20") else None
        history.append(SignalPoint(date=str(idx), session_date=(session_dates or {}).get(str(idx)), prob_up=float(p),
                                   sentiment=sentiment_tag(float(p)), realised_outcome=outcome, in_training_period=in_train))

    prob_up = float(probs[-1])
    try:
        explanation = explain_prediction(loaded.model, context, preprocessor=loaded.preprocessor)
        if explanation.get("supported"):
            explanation["matches_prediction"] = bool(abs(explanation["final_prob"] - prob_up) < 1e-4)
    except Exception as exc:  # never let an explanation failure hide the prediction
        logger.exception("SHAP explanation failed")
        explanation = {"supported": False, "message": f"Explanation failed: {type(exc).__name__}: {exc}"}

    meta = loaded.metadata
    fresh = _freshness(data_latest)
    if session is not None:
        fresh["data_stale"] = session["status"] == "outdated"
    return PredictResponse(
        as_of=str(context.index[-1]),
        generated_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        horizon_days=HORIZON,
        prediction="Up/Flat" if prob_up >= 0.5 else "Down",
        prob_up=prob_up,
        probabilities={"down": 1.0 - prob_up, "up": prob_up},
        sentiment=sentiment_tag(prob_up),
        avg_prob_7d=float(np.mean(tail_probs)),
        n_rows_used=len(context),
        experimental=loaded.experimental,
        experimental_notice=EXPERIMENTAL_NOTICE if loaded.experimental else None,
        session=SessionInfo(**session) if session is not None else None,
        realised_outcome=history[-1].realised_outcome if history else None,
        model=ModelRef(name=meta.get("model_name", "?"), version=loaded.version, trained_at=meta.get("trained_at", "?")),
        signal_history=history,
        explanation=explanation,
        **fresh,
    )


def records_to_frame(records: List[Any]) -> pd.DataFrame:
    """Convert validated rows into an ordered feature frame (duplicates/ordering checked)."""
    df = pd.DataFrame([r.model_dump() for r in records])
    if df["date"].notna().all():
        if df["date"].duplicated().any():
            raise HTTPException(status_code=422, detail="Duplicate dates in records.")
        df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
        df = df.sort_values("date").set_index("date")
    elif df["date"].notna().any():
        raise HTTPException(status_code=422, detail="Provide 'date' on every record or on none.")
    else:
        df.index = [f"t-{len(df) - 1 - i}" for i in range(len(df))]
    return df[FEATURE_COLUMNS]


# --------------------------------------------------------------------------- #
# App                                                                         #
# --------------------------------------------------------------------------- #
def _load_approved(app: FastAPI) -> None:
    app.state.loaded = None
    app.state.load_error = None
    try:
        app.state.loaded = app.state.registry.load(allow_unvalidated=allow_unvalidated())
        logger.warning("Loaded EXPERIMENTAL (unvalidated) model %s", app.state.loaded.version) if app.state.loaded.experimental \
            else logger.info("Loaded approved model %s", app.state.loaded.version)
    except RegistryError as exc:
        app.state.load_error = str(exc)
        logger.warning("No usable model: %s", exc)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Resolve config and load the APPROVED registry model once at start-up."""
    app.state.db_path = resolve_path(os.getenv("FEATURE_STORE_PATH", str(FEATURE_STORE_PATH)))
    app.state.registry = ModelRegistry(resolve_path(os.getenv("MODEL_DIR", str(MODEL_DIR))))
    _load_approved(app)
    yield


app = FastAPI(
    title="Brent Crude Direction API",
    description="Online inference + TreeSHAP explanations: will Brent settle the CURRENT trading session at or above the previous settlement?",
    version="3.0.0",
    lifespan=lifespan,
)


def _require_model() -> LoadedModel:
    if app.state.loaded is None:
        err = app.state.load_error or ""
        if err.startswith("No approved model in the registry"):
            msg = ("No validated prediction model is available yet. Run `python -m src.train`, then restart the API. "
                   "(To demonstrate a rejected model, set ALLOW_UNVALIDATED_MODEL=1 in .env.)")
        else:
            msg = f"The prediction model could not be loaded: {err}"
        raise HTTPException(status_code=503, detail=msg)
    return app.state.loaded


def _latest_data_date() -> Optional[str]:
    try:
        return str(load_recent_context(1, app.state.db_path).index[-1].date())
    except (FileNotFoundError, RuntimeError, IndexError):
        return None


def _asset_last_observations() -> Optional[Dict[str, pd.Timestamp]]:
    """Last REAL observation date of each input, as recorded by the ingestion job (None if unknown)."""
    try:
        df = read_table("data_status", app.state.db_path)
        return {r.asset: pd.Timestamp(r.last_observation) for r in df.itertuples()}
    except (FileNotFoundError, RuntimeError) as exc:
        logger.warning("data_status unavailable, skipping per-input staleness check: %s", exc)
        return None


@app.get("/health")
def health() -> Dict[str, Any]:
    """Liveness + model availability + data freshness."""
    latest = _latest_data_date()
    loaded = app.state.loaded
    fresh = _freshness(latest)
    ok = loaded is not None and latest is not None
    return {
        "status": "ok" if ok else "degraded",
        "model_loaded": loaded is not None,
        "model_experimental": bool(loaded and loaded.experimental),
        "model_version": loaded.version if loaded else None,
        "model_error": app.state.load_error,
        "feature_store_available": latest is not None,
        "horizon_days": HORIZON,
        "n_features": len(FEATURE_COLUMNS),
        "session_status": (assess_session(pd.Timestamp(latest), today_in_market(), _asset_last_observations())["status"]
                           if latest else None),
        **fresh,
    }


@app.get("/model")
def model_info() -> Dict[str, Any]:
    """Approved model metadata (schema, target, metrics, comparison, selection rationale)."""
    loaded = _require_model()
    return {**loaded.metadata, "experimental": loaded.experimental,
            "experimental_notice": EXPERIMENTAL_NOTICE if loaded.experimental else None}


@app.get("/model/versions")
def model_versions() -> Dict[str, Any]:
    reg: ModelRegistry = app.state.registry
    return {"approved": reg.approved_version(), "versions": reg.list_versions()}


@app.post("/model/reload")
def model_reload() -> Dict[str, Any]:
    """Re-load the approved model (use after `python -m src.registry promote|rollback`). Local-use endpoint."""
    _load_approved(app)
    if app.state.loaded is None:
        raise HTTPException(status_code=503, detail=app.state.load_error)
    return {"loaded_version": app.state.loaded.version, "experimental": app.state.loaded.experimental}


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest) -> PredictResponse:
    """Run inference on a caller-supplied lookback slice of feature rows."""
    loaded = _require_model()
    return run_inference(loaded, records_to_frame(req.records), labels=None, data_latest=_latest_data_date())


@app.get("/predict/latest", response_model=PredictResponse)
def predict_latest(as_of: Optional[date_type] = Query(None, description="Replay: only use feature rows up to this date")) -> PredictResponse:
    """
    Prediction for the CURRENT trading session (see src/sessions.py for the exact rules).

    current  -> today's session, from the last completed session's data.
    closed / outdated -> the latest available session (the one before it supplies the inputs);
                         ``session.message`` explains why.
    stale input -> 503 with a plain-English reason (no prediction is produced).
    """
    loaded = _require_model()
    if HORIZON != 1:
        raise HTTPException(status_code=503, detail="Current-session predictions need FORECAST_HORIZON=1 and a model trained for it. "
                                                    "Set FORECAST_HORIZON=1, then run `python -m src.features` and `python -m src.train`.")
    try:
        full = load_recent_context(LOOKBACK_DAYS + 2, app.state.db_path, as_of=as_of)
        labels = load_labels(app.state.db_path)
    except (FileNotFoundError, RuntimeError, KeyError) as exc:  # missing / locked / old-schema store
        raise HTTPException(status_code=503, detail=f"Market data is not available: {exc}")
    if len(full) < MIN_ROWS + 1:
        raise HTTPException(status_code=503, detail="There is not enough market history for a reliable prediction.")

    latest = full.index[-1]
    if as_of is None:
        info = assess_session(latest, today_in_market(), _asset_last_observations())
        if info["status"] == "insufficient":
            raise HTTPException(status_code=503, detail=info["message"])
    else:  # replay of a past date: every row after as_of is excluded above
        nxt = next_session(latest)
        info = {"status": "replay", "session_date": str(nxt.date()), "input_date": str(latest.date()), "uses_previous_row": False,
                "is_current_session": False, "message": f"Replay: using only data up to {latest.date()}.",
                "latest_data_date": str(latest.date()), "next_session": str(nxt.date())}

    rows = full.iloc[:-1] if info["uses_previous_row"] else full   # last row = the data the model is allowed to use
    info["input_date"] = str(rows.index[-1].date())
    context = rows.tail(LOOKBACK_DAYS).copy()
    # session each signal is about = the next row in the store (or the target session for the newest signal)
    following = {d: n for d, n in zip(full.index[:-1], full.index[1:])}
    session_dates = {d.strftime("%Y-%m-%d"): (following[d].strftime("%Y-%m-%d") if d in following else info["session_date"])
                     for d in context.index}
    context.index = context.index.strftime("%Y-%m-%d")
    labels.index = labels.index.strftime("%Y-%m-%d")
    resp = run_inference(loaded, context, labels=labels, data_latest=str(latest.date()), session=info, session_dates=session_dates)
    if info["status"] in ("current", "replay"):
        try:  # persist the signal; failure to write must not fail the request
            upsert_predictions(
                pd.DataFrame([{"date": resp.as_of, "model_version": loaded.version, "split": "live" if as_of is None else "replay",
                               "prob_up": resp.prob_up, "sentiment": resp.sentiment, "actual": None,
                               "generated_at": resp.generated_at.replace("+00:00", "")}]),
                app.state.db_path,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("Could not persist prediction: %s", exc)
    return resp


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "app.api:app",
        host=os.getenv("API_HOST", "127.0.0.1"),
        port=int(os.getenv("API_PORT", "8000")),
        reload=False,
    )
