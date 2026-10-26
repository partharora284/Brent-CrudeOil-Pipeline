"""
app/dashboard.py  --  LAYER 5: Interactive Web UI (Streamlit)
==============================================================

A thin client: it never loads the model itself.  It calls the FastAPI backend (URL from the
``API_URL`` environment variable / .env, default http://127.0.0.1:8000) and shows:

1. The prediction for the CURRENT trading session: will Brent settle at or above the previous
   settlement?  (Made before the session settles from data up to the last completed session;
   it does not use intraday prices.)  If markets are closed or the data is not up to date, the
   latest available session is shown with a plain message saying so.
2. The Bull / Bear / Neutral tag and the probability.
3. The model's recent calls (last 7 sessions) with the actual result once known.
4. A plain-English explanation of the main factors pushing the call up or down (from the
   model's real SHAP values).
5. Model evaluation vs. baselines on the held-out test period.

If the API runs in experimental mode (ALLOW_UNVALIDATED_MODEL=1 in .env), a warning stays on screen
the whole time: the model did not pass validation and must not be treated as reliable.

Run (the API must already be running)::

    streamlit run app/dashboard.py
"""

from __future__ import annotations

import os
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(PROJECT_ROOT / ".env")

API_URL = os.getenv("API_URL", "http://127.0.0.1:8000").rstrip("/")
CONF_UPPER = float(os.getenv("CONF_UPPER", "0.55"))
CONF_LOWER = float(os.getenv("CONF_LOWER", "0.45"))
REFRESH_SECONDS = int(os.getenv("DASHBOARD_REFRESH_SECONDS", "300"))  # a fetched prediction is reused this long

EXPERIMENTAL_NOTICE = "Experimental model \u2014 this model did not pass the required validation and is not established as reliable."
GREEN, RED, GREY, NAVY = "#1b9e5a", "#d64545", "#7f8c8d", "#34495e"
TAG_STYLE = {"BULL": (GREEN, "BULL"), "BEAR": (RED, "BEAR"), "NEUTRAL": (GREY, "NEUTRAL")}
STRENGTH_TEXT = {"strong": "strong effect", "moderate": "moderate effect", "slight": "slight effect"}

st.set_page_config(page_title="Brent Crude Signal", page_icon="🛢️", layout="wide")


# --------------------------------------------------------------------------- #
# API client                                                                  #
# --------------------------------------------------------------------------- #
class ApiProblem(Exception):
    """The API answered, but with a message meant for the user (e.g. data unavailable)."""


@st.cache_data(ttl=REFRESH_SECONDS, show_spinner=False)
def fetch_prediction(base_url: str) -> dict:
    resp = requests.get(f"{base_url}/predict/latest", timeout=60)
    if resp.status_code != 200:
        try:
            detail = resp.json().get("detail", resp.text)
        except ValueError:
            detail = resp.text
        raise ApiProblem(str(detail))
    return resp.json()


@st.cache_data(ttl=REFRESH_SECONDS, show_spinner=False)
def fetch_model_info(base_url: str) -> dict | None:
    """Best-effort: the evaluation section is simply hidden if this is unavailable."""
    try:
        r = requests.get(f"{base_url}/model", timeout=15)
        return r.json() if r.status_code == 200 else None
    except Exception:  # noqa: BLE001
        return None


def nice_date(s: str) -> str:
    return pd.Timestamp(s).strftime("%a %d %b %Y")


# --------------------------------------------------------------------------- #
# Chart builders                                                              #
# --------------------------------------------------------------------------- #
def signal_history_chart(points: list[dict]) -> go.Figure:
    """The model's call for each of the last 7 sessions, with the BULL / BEAR zones shaded."""
    df = pd.DataFrame(points)
    day = df["session_date"] if "session_date" in df else df["date"]
    x = day.fillna(df["date"]).map(lambda d: pd.Timestamp(d).strftime("%a %d %b"))
    colors = [TAG_STYLE[s][0] for s in df["sentiment"]]
    fig = go.Figure()
    fig.add_hrect(y0=CONF_UPPER, y1=1, fillcolor=GREEN, opacity=0.08, line_width=0)
    fig.add_hrect(y0=0, y1=CONF_LOWER, fillcolor=RED, opacity=0.08, line_width=0)
    fig.add_hline(y=0.5, line_dash="dot", line_color="#999")
    fig.add_trace(go.Scatter(
        x=x, y=df["prob_up"], mode="lines+markers", line=dict(color=NAVY, width=2),
        marker=dict(size=11, color=colors, line=dict(width=1, color="white")),
        hovertemplate="%{x}<br>Chance of up/flat: %{y:.1%}<extra></extra>"))
    fig.update_yaxes(range=[0, 1], tickformat=".0%", title="Chance of an up/flat session")
    fig.update_xaxes(type="category")
    fig.update_layout(height=340, margin=dict(l=10, r=10, t=10, b=10), showlegend=False)
    return fig


def factors_chart(factors: list[dict]) -> go.Figure:
    """Diverging bars: green bars push the prediction UP (right), red bars push it DOWN (left)."""
    f = sorted(factors, key=lambda x: x["share"])  # the biggest effect ends up at the top
    xs = [x["share"] if x["direction"] == "up" else -x["share"] for x in f]
    labels = [("▲ " if x["direction"] == "up" else "▼ ") + x["label"] for x in f]
    fig = go.Figure(go.Bar(
        x=xs, y=labels, orientation="h",
        marker_color=[GREEN if v > 0 else RED for v in xs],
        text=[STRENGTH_TEXT[x["strength"]] for x in f], textposition="outside", cliponaxis=False,
        customdata=[[x["description"], x["value_text"]] for x in f],
        hovertemplate="<b>%{y}</b><br>Right now: %{customdata[1]}<br><i>%{customdata[0]}</i><extra></extra>"))
    lim = max(abs(v) for v in xs) * 1.6
    fig.update_xaxes(range=[-lim, lim], showticklabels=False, zeroline=True, zerolinecolor="#555", zerolinewidth=2,
                     showgrid=False, title="◀ pushes the prediction DOWN          pushes the prediction UP ▶")
    fig.update_yaxes(automargin=True, tickfont=dict(size=14))
    fig.update_layout(height=90 + 62 * len(f), margin=dict(l=10, r=10, t=10, b=10), showlegend=False,
                      bargap=0.35, font=dict(size=13))
    return fig


def importance_chart(explanation: dict, top: int = 10) -> go.Figure:
    """Typical importance over the recent lookback (technical-details section)."""
    imp = pd.DataFrame(explanation["global_importance"]).head(top).iloc[::-1]
    names = imp["label"] if "label" in imp else imp["feature"]
    fig = go.Figure(go.Bar(x=imp["mean_abs_shap"], y=names, orientation="h", marker_color=NAVY))
    fig.update_layout(height=120 + 30 * len(imp), margin=dict(l=10, r=10, t=10, b=10),
                      xaxis_title="Average size of the factor's effect over the last 60 sessions (model units)")
    fig.update_yaxes(automargin=True)
    return fig


def metrics_table(meta: dict) -> pd.DataFrame:
    """Held-out test comparison of every model and the baselines."""
    rows = []
    for name, m in meta["baselines"].items():
        rows.append({"model": f"baseline: {name}", "accuracy": m["standard_accuracy"], "precision": m["precision"],
                     "recall": m["recall"], "F1": m["f1"], "ROC-AUC": None, "train CV acc": None})
    for name, m in meta["comparison"].items():
        rows.append({"model": name + ("  ← selected" if name == meta["model_name"] else ""),
                     "accuracy": m["standard_accuracy"], "precision": m["precision"], "recall": m["recall"],
                     "F1": m["f1"], "ROC-AUC": m["roc_auc"], "train CV acc": m["cv_accuracy"]})
    return pd.DataFrame(rows)


def render_explanation(ex: dict) -> None:
    """Plain-English 'why' section; the technical numbers live in an optional expander."""
    if not ex.get("supported") or not ex.get("plain"):
        st.info("A factor-by-factor explanation isn't available for this model type.")
        return
    plain = ex["plain"]
    st.info(plain["summary"])
    factors = plain["factors"]
    if factors:
        st.plotly_chart(factors_chart(factors), width="stretch")
        st.caption("Longer bar = bigger effect on this prediction. Hover over a bar for details.")
        up = [f for f in factors if f["direction"] == "up"]
        down = [f for f in factors if f["direction"] == "down"]
        c_up, c_down = st.columns(2)
        for col, title, items in ((c_up, "▲ Pushing the prediction UP", up), (c_down, "▼ Pulling the prediction DOWN", down)):
            with col:
                st.markdown(f"**{title}**")
                if not items:
                    st.caption("Nothing major in this direction.")
                for f in items:
                    st.markdown(f"**{f['label']}** · {STRENGTH_TEXT[f['strength']]}  \n"
                                f"Right now: {f['value_text']}.  \n"
                                f"<span style='color:#666'>{f['description']}</span>", unsafe_allow_html=True)
        other = plain["other"]
        if other["count"] and other["net_direction"] != "neutral":
            st.caption(f"The other {other['count']} factors are small and, taken together, lean "
                       f"{'up' if other['net_direction'] == 'up' else 'down'}.")
    if ex.get("matches_prediction") is False:
        st.warning("The explanation does not exactly match the displayed prediction; treat it with caution.")
    st.caption("⚠ This describes how the model behaves, not why the oil market moves. It is not a guarantee.")

    with st.expander("Technical details (optional)"):
        st.markdown(
            "The model works on an internal score called *log-odds*: factors add to or subtract from a starting "
            "score, and the total is converted into the probability shown above. Positive numbers push toward "
            "**up/flat**, negative numbers toward **down**. These are SHAP values, computed exactly for this model.")
        tbl = pd.DataFrame(ex["feature_importance"])
        tbl = tbl.rename(columns={"label": "factor", "feature": "model input name", "shap": "effect (log-odds)", "value": "current value"})
        st.dataframe(tbl[["factor", "model input name", "current value", "effect (log-odds)"]], hide_index=True, width="stretch")
        st.markdown("**Which factors matter most on average (last 60 sessions)**")
        st.plotly_chart(importance_chart(ex), width="stretch")


# --------------------------------------------------------------------------- #
# Layout                                                                      #
# --------------------------------------------------------------------------- #
st.title("🛢️ Brent Crude – Session Outlook")

try:
    with st.spinner("Loading the latest prediction..."):
        res = fetch_prediction(API_URL)
except requests.exceptions.ConnectionError:
    st.error("The prediction service isn't running. Start it with `uvicorn app.api:app --port 8000`, then reload this page.")
    st.stop()
except ApiProblem as exc:  # the API already wrote a complete, user-friendly sentence
    st.warning(str(exc))
    st.stop()
except Exception as exc:  # noqa: BLE001
    st.error(f"The prediction couldn't be loaded ({type(exc).__name__}). Please try again in a moment.")
    st.stop()


sess = res.get("session") or {}
is_current = bool(sess.get("is_current_session"))
session_date = sess.get("session_date") or res["as_of"]
color, label = TAG_STYLE[res["sentiment"]]

if is_current:
    st.caption("Will Brent settle **today's session at or above the previous settlement**? This is generated before the "
               "session ends, using market data up to the last completed session. It does not use live intraday prices.")
else:
    st.caption("Will Brent settle a session at or above the previous settlement? Each call is made before the session, "
               "using market data up to the session before it. It does not use live intraday prices.")
if sess.get("message"):
    (st.warning if sess.get("status") == "outdated" else st.info)(sess["message"])
st.caption(f"📅 Market data up to: **{nice_date(sess.get('latest_data_date') or res['as_of'])}** (last completed session)")

# ---- Row 1: headline tiles ------------------------------------------------ #
c1, c2, c3, c4 = st.columns(4)
with c1:
    st.markdown("**Signal (experimental)**" if res.get("experimental") else "**Signal**")
    st.markdown(
        f"<div style='background:{color};color:white;padding:14px;border-radius:10px;"
        f"text-align:center;font-size:30px;font-weight:700'>{label}</div>", unsafe_allow_html=True)
c2.metric("Today's session" if is_current else "Latest session shown", nice_date(session_date),
          help="The trading session this prediction is about.")
c3.metric("Chance of finishing up or flat", f"{res['prob_up']:.1%}",
          help=f"Chance of finishing down: {res['probabilities']['down']:.1%}. Call at 50%: {res['prediction']}.")
if res.get("realised_outcome"):
    c4.metric("What actually happened", res["realised_outcome"], help="The real result of this session versus the previous settlement.")
else:
    c4.metric("Average chance, last 7 sessions", f"{res['avg_prob_7d']:.1%}")

# ---- Row 2: recent calls + explanation ------------------------------------ #
left, right = st.columns([1, 1.25])
with left:
    st.subheader("Recent calls (last 7 sessions)")
    st.plotly_chart(signal_history_chart(res["signal_history"]), width="stretch")
    st.caption(
        f"Each dot is the model's chance that Brent would finish that session up or flat, issued before the session. "
        f"Green zone ≥ {CONF_UPPER:.0%} = BULL, red zone ≤ {CONF_LOWER:.0%} = BEAR, in between = NEUTRAL (no clear lean).")
    hist = pd.DataFrame(res["signal_history"])
    if "session_date" not in hist:
        hist["session_date"] = None
    hist["session"] = hist["session_date"].fillna(hist["date"]).map(nice_date)
    hist["chance up/flat"] = hist["prob_up"].map("{:.1%}".format)
    hist["actual result"] = hist["realised_outcome"].fillna("not settled yet")
    st.dataframe(hist[["session", "chance up/flat", "sentiment", "actual result"]], hide_index=True, width="stretch")
with right:
    st.subheader("Why did the model say this?")
    render_explanation(res["explanation"])

# ---- Row 3: evaluation ------------------------------------------------------ #
st.subheader("Model evaluation vs. baselines (held-out test period)")
meta = fetch_model_info(API_URL)
if meta:
    d = meta["data"]
    st.caption(f"Test window {d['test_start']} → {d['test_end']} ({d['n_test']} sessions); trained on "
               f"{d['train_start']} → {d['train_end']} with a {d['embargo_rows']}-row embargo. "
               f"Target: {meta['target']['definition']}")
    st.dataframe(
        metrics_table(meta).style.format({c: "{:.3f}" for c in ["accuracy", "precision", "recall", "F1", "ROC-AUC", "train CV acc"]}, na_rep="–"),
        hide_index=True, width="stretch")
    g = meta["validation"]
    (st.success if g["passed"] else st.error)(
        f"Validation gate {'PASSED' if g['passed'] else 'FAILED'}: edge over the best baseline "
        f"{g['edge_over_best_baseline']*100:+.2f} pp (naive z = {g['edge_z_score_naive']:.2f}). {g['note']}")
    cm = meta["metrics"]["confusion_matrix"]
    cc1, cc2 = st.columns([1, 2])
    cc1.markdown("**Confusion matrix (selected model)**")
    cc1.dataframe(pd.DataFrame(cm, index=["actual Down", "actual Up/Flat"], columns=["pred Down", "pred Up/Flat"]), width="stretch")
    cc2.markdown("**Selection rationale**")
    cc2.write(meta["selection"]["rationale"])
else:
    st.info("Model evaluation details are unavailable.")

st.caption("Research tool only - not investment advice.")
