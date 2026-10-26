# Brent Crude Current-Session Direction Pipeline

Predicts whether Brent's settlement **for the current trading session** will be **at or above (Up/Flat = 1) or below
(Down = 0) the previous session's settlement**, serves the call through FastAPI with TreeSHAP explanations, and shows it
in a Streamlit dashboard with a plain-English explanation.

```
Yahoo Finance (live, optional) + CSV history -> src/providers.py + src/ingestion.py -> DuckDB: raw_*, master_aligned, data_status
                          -> src/features.py   -> DuckDB: feature_matrix, inference_features
                          -> src/train.py      -> models/registry/ (versioned, gated)  + DuckDB: predictions
                          -> src/sessions.py   -> which session is predicted? is the data good enough?
                          -> app/api.py        -> FastAPI (approved model only, SHAP + plain-English factors)
                          -> app/dashboard.py  -> Streamlit
```

## What the prediction means (read this)

* **Question answered:** "Will Brent settle today's session at or above the previous settlement?"
* **When it is generated:** before that session settles, from data up to the **last completed session**. Features are
  the same 22 daily features as before; the target is now **1 session ahead** (`target_1d_class`), the model was changed
  from 3 sessions ahead and **must be retrained** (old 3-day model files are not compatible and are refused).
* **What it is not:** it does not use intraday prices and does **not** predict the close of a half-finished session from
  partial data. A provider's bar for today is discarded as unfinished (a bar counts only if its date is before today in `MARKET_TZ`).
* **If today is not a trading day** (weekend / holiday), the dashboard shows the **latest completed session** (its call was made
  from the session before it, with the actual result) and says so. **If data is out of date** it shows the latest available session
  with a message. **If an input (WTI, DXY) is stale** it shows no prediction, only a message. Rules: `src/sessions.py`.
* Holidays: weekends, 25 Dec, 1 Jan and Good Friday are built in; add others via `MARKET_HOLIDAYS`. This is an approximation, not
  the official ICE calendar; on an unlisted holiday the dashboard may say data is outdated until the next session.

## Setup (Windows PowerShell)

```powershell
cd brent_crude_pipeline
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1          # if blocked: Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
python -m pip install --upgrade pip
pip install -r requirements.txt
Copy-Item .env.example .env           # then edit .env if needed (keys go here, never in code)
```

> **Upgrading from the SQLite version?** An old `.env` with `FEATURE_STORE_PATH=data/feature_store.db` is detected and
> rejected with a clear message. Change it to `data/feature_store.duckdb` (or delete the line) and rebuild.

## First run: fetch data, train, serve

The old feature store and 3-day model were removed from this ZIP because they are incompatible with the new target.
Rebuild them once:

```powershell
python -m src.scheduled_ingest   # live download (Yahoo) + CSV history -> validate -> features. Safe to re-run any time.
python -m src.train              # retrain for the 1-session target; promotes ONLY if the validation gate passes
python -m src.registry list      # inspect versions / approved model
uvicorn app.api:app --port 8000          # terminal 1
streamlit run app/dashboard.py           # terminal 2 (activate .venv there too)
pytest                                   # tests (synthetic data, temp dirs; never touch real models/data)
```

Check the three live feeds first (stores nothing; prints OK/FAILED per input and the exact reason):

```powershell
python -m src.providers
```

### If the model fails the validation gate

`python -m src.train` registers the model as **rejected** and nothing is served (the dashboard says no validated model
is available). To demonstrate the full dashboard anyway, add this to `.env` and restart `uvicorn`:

```
ALLOW_UNVALIDATED_MODEL=1
```

The API then loads the newest rejected model **only if** its files are complete, its checksums match and it was trained for
`target_1d_class` with the current features. The old 3-day model is never loaded. Every response carries `experimental: true`, and the
dashboard shows a permanent banner: *"Experimental model — this model did not pass the required validation and is not
established as reliable."* Set it back to `0` for strict behaviour. An approved model always takes priority over an experimental one.

Version folders: new versions are numbered from the registry **and** the folders on disk, so a leftover folder from an interrupted
run (such as `v0002`) can no longer cause `FileExistsError`; a failed registration removes its half-written folder.

Keep it fresh automatically (pick one):

```powershell
python -m src.scheduled_ingest --loop --every-minutes 60     # terminal 3, repeats until Ctrl+C
# or Windows Task Scheduler, see "Data ingestion & scheduling" below
```

The API recomputes the prediction from the newest stored data on every request and the dashboard re-asks it every
5 minutes (`DASHBOARD_REFRESH_SECONDS`), so a refreshed data set appears without restarting anything. The dashboard has no
settings: the API address comes from `API_URL` in `.env`. Retrain (`python -m src.train`) occasionally, for example monthly.

Open http://127.0.0.1:8000/docs for the API explorer.

## API

| Endpoint | Purpose |
|---|---|
| `GET /health` | model loaded?, version, feature-store available?, latest data date, staleness |
| `GET /model` | approved model metadata: schema, target, metrics, baselines, selection rationale |
| `GET /model/versions` | all registered versions and status |
| `POST /model/reload` | reload the approved model after `promote` / `rollback` |
| `POST /predict` | body `{"records":[{date, <22 features>}, ... >=7 rows]}`; Pydantic-validated (422 on missing/non-numeric/NaN/duplicate-date) |
| `GET /predict/latest[?as_of=YYYY-MM-DD]` | prediction for the current session (see above); `as_of` replays using only rows up to that date; **503 + plain message** when an input is stale |

Responses include `prediction`, `prob_up`, `probabilities`, `sentiment` (BULL ≥ 0.55, BEAR ≤ 0.45, else NEUTRAL),
`session{status, session_date, input_date, is_current_session, message}`, `realised_outcome`, `generated_at`,
`model{...}` (API only; the dashboard no longer shows it), `signal_history` (last 7 sessions, each with its `session_date`
and the realised outcome once known) and `explanation` (raw SHAP **plus** `plain`: friendly labels, direction, strength, summary).

## Model comparison, selection and registry

`python -m src.train` fits, on the training window only: Logistic Regression, Ridge classifier (scores are *not*
calibrated probabilities), XGBoost and LightGBM (`RandomizedSearchCV`, `TimeSeriesSplit(gap=3)`), and scores all of them plus
two baselines (always-Up, train-majority) on the **same** chronological test window (accuracy, precision, recall, F1,
confusion matrix, ROC-AUC). The test window is skipped by selection: the model is chosen by training-CV accuracy.
A **validation gate** (test accuracy strictly above both baselines and ROC-AUC > 0.5) decides promotion; a failing
model is stored as `rejected`, cannot be promoted, and the previous approved model keeps serving.

```powershell
python -m src.registry list
python -m src.registry promote v0002        # refuses rejected / corrupt / schema-incompatible versions
python -m src.registry rollback             # or: rollback --to v0001
# then: Invoke-RestMethod -Method Post http://127.0.0.1:8000/model/reload   (or restart the API)
```

Each version stores `model.joblib`, a fitted `preprocessor.joblib` (`FeaturePreprocessor`: schema/order/finite checks),
and `metadata.json` (name, version, trained_at, feature schema + hash, target definition, metrics, rationale, gate,
SHA-256 of artefacts). Loading re-verifies hashes, the feature-schema fingerprint and runs a smoke prediction.
The approved model is mirrored to `models/best_model.joblib` / `preprocessor.joblib`. `models/legacy/` keeps the
original pre-registry artefacts for reference only (not loaded by anything).

**Honest reading of the result:** the gate means "measured edge on the held-out window", not "proven skill". The numbers for the retrained 1-session model are printed by `python -m src.train` and shown on the dashboard; none were measured when this README was written.

## Features & target

20 features (Gasoil removed; the other features are unchanged): Brent-WTI spread, Brent-DXY z-score gap, log returns (3 assets), 14d correlations,
7d/30d volatility, high-low range, EMA20/SMA50/SMA200 distance ratios, return lags, day-of-week, month. All three inputs (Brent, WTI, DXY) are
required. Target `target_1d_class` = Brent's next-session settlement >= this session's; the newest row stays unlabelled (NaN),
never "Down". Features use only past data (tested by recomputing on truncated history). **Term spreads are not implemented**
(needs multi-contract futures data).

Honest expectation: next-session direction is close to a coin flip. The validation gate promotes a model only if it beats the
baselines on the held-out window; read the numbers it prints and the dashboard's evaluation table, and do not assume skill.

## Data ingestion & scheduling

* **History** = the Brent, WTI and DXY CSVs in `data/raw/` (never modified; the London Gas Oil CSV is no longer used and can stay in the folder). **Live rows** = completed daily bars from Yahoo Finance
  (`BZ=F` Brent, `CL=F` WTI, `DX-Y.NYB` DXY), stored in `raw_provider_yahoo_<asset>` and appended **only after**
  the CSV's last date. No API key is needed.
* **Validation per asset (`src/providers.py`)**: drops missing/non-positive closes, today's unfinished bar, High<Low, Brent rows
  without High/Low, duplicate dates and single-day jumps (>25%, DXY >5%). It then compares the download with the CSV on the
  overlapping dates; if prices differ by more than 5% (median) the series is **rejected** (wrong symbol/contract) and nothing from it is used.
* **Missing data**: nothing is invented. Only settlement prices of WTI/DXY are forward-filled (past values, for days Brent traded and they
  did not). If an input stays older than Brent by more than `MAX_ASSET_LAG_DAYS` (3) the API refuses to predict. Per-asset results are in table `data_status`.
* **Brent calendar** stays the master calendar, so holidays create no fake zero-return rows.
* **Provider failure**: logged per asset, the job continues with stored data, exit code 2; the dashboard shows an "outdated" message.
* Layers in DuckDB: `raw_*` -> `master_aligned` -> `feature_matrix` / `inference_features` -> `predictions`; plus `data_status`.
* The optional FRED hook (`DATA_PROVIDER=fred`) still exists but only supplies Brent/WTI closes, which is **not** enough for the model.

```powershell
python -m src.scheduled_ingest                      # exit 0 ok, 2 finished but live data failed for >=1 input, 1 job failed
# Windows Task Scheduler (daily 18:30) - adjust the path:
schtasks /Create /SC DAILY /ST 18:30 /TN "BrentIngestion" /TR "powershell.exe -NoProfile -ExecutionPolicy Bypass -File C:\path\to\brent_crude_pipeline\scripts\run_ingestion.ps1"
schtasks /Run /TN "BrentIngestion"          # test it
```

**Provider limitations:** Yahoo's endpoint is free but unofficial: it can rate-limit, change format or drop a symbol without notice.
Its Brent front-month series can differ slightly from the Investing.com history around contract rolls (a small level jump can appear where
the two sources join). London Gasoil was removed as a model input because its live feed was unreliable; the model now needs only Brent, WTI and DXY. If one of those fails, `python -m src.providers` shows which and why.
A paid data feed would be more reliable.

## Troubleshooting

* `is an old SQLite feature store` -> set `FEATURE_STORE_PATH=data/feature_store.duckdb` in `.env`, run `python -m src.features`.
* `Feature store ... is locked by another process` -> DuckDB allows one writer; wait for the ingestion job to finish.
* Dashboard says *Cannot reach the API* -> start `uvicorn` first, check `API_URL`.
* API returns 503 *No approved model* -> `python -m src.train`; if the model was rejected, see `python -m src.registry list`; then `POST /model/reload`.
* 503 *schema mismatch* -> features changed since the model was trained; retrain.
* Dashboard says *data has not been updated* -> run `python -m src.scheduled_ingest` and read which input failed in its output / `logs/ingestion.log`.
* Dashboard says *not enough up-to-date market data* -> that input (named in the message) has not updated; check its symbol in `.env`.
* `FileExistsError ... versions\\v000N` -> fixed in this release; update the code, re-run `python -m src.train`.
* API says `FORECAST_HORIZON=1` needed -> an old `.env` may still contain `FORECAST_HORIZON=3`; set it to 1, rebuild features and retrain.
* `target_3d_class` / KeyError / 503 *schema mismatch* -> old store or model; run `python -m src.scheduled_ingest` then `python -m src.train`.
* Activation blocked in PowerShell -> `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass`.

*Research tool only - not investment advice.*


## Model-improvement experiments (`python -m src.experiments`)

A separate, read-only script (it never changes the feature schema, registry or dashboard):

```powershell
python -m src.experiments audit         # duplicates, gaps, NaNs, look-ahead check, split layout
python -m src.experiments baseline      # current features/models on the project holdout (metrics only)
python -m src.experiments walkforward   # 7 extra Brent/WTI/DXY features x logistic / XGBoost / LightGBM, training region only
python -m src.experiments final --features extended --model lightgbm --param 0   # ONE pre-chosen candidate, holdout, run at most once
```

Walk-forward = 6 chronological expanding-window blocks inside the training region (1-row embargo). A candidate qualifies only if it beats
"always up" and has ROC-AUC above 0.5 on average and in at least 4 of 6 blocks. Only a qualifying candidate may be evaluated on the
holdout with `final`, and it must then also pass the project's normal validation gate. Result with the data in this repo: no candidate
qualified, so the production model and features were left unchanged.
