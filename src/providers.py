"""
src/providers.py  --  Optional live market-data providers (incremental download)
================================================================================

The CSV workflow in ``src/ingestion.py`` is the primary, always-available source.
This module adds a small, pluggable interface for live feeds:

    class MyProvider(MarketDataProvider): ...   # implement ``fetch``
    PROVIDERS["myprovider"] = MyProvider

Credentials come from environment variables only (never from source code).

Enable with ``DATA_PROVIDER=fred`` and ``FRED_API_KEY=...`` in ``.env``.

YAHOO FINANCE (``DATA_PROVIDER=yahoo``, the default; no API key) is the provider that CAN feed the
model: it returns Open/High/Low/Close for all three inputs (Brent, WTI, DXY).  It is
an unofficial, free endpoint, so every row goes through ``validate_provider_frame`` and every asset is
cross-checked against the existing CSV history (``overlap_difference``) before it is used.  Anything
that fails is reported per asset and is NEVER replaced by an invented or substituted value.

IMPORTANT LIMITATION: FRED publishes daily CLOSES only (no Open/High/Low, no ICE
DXY), while the feature set needs Brent High/Low.
Downloaded rows are therefore stored in ``raw_provider_fred_<asset>`` and used to
report data freshness; they are deliberately NOT merged into ``master_aligned``
(that would require inventing High/Low values).  A provider that returns full
OHLC rows for all three assets can be merged by extending ``ingest_provider``.
"""

from __future__ import annotations

import logging
import os
import time
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Type
from urllib.parse import quote

import pandas as pd

from src.store import _connect, append_rows, list_tables, persist_dataframe, read_table

logger = logging.getLogger(__name__)


class ProviderUnavailable(RuntimeError):
    """Provider not configured (missing credentials) or unreachable."""


class MarketDataProvider(ABC):
    name: str = "base"
    assets: tuple = ()
    has_ohlc: bool = False

    @abstractmethod
    def is_configured(self) -> bool: ...

    @abstractmethod
    def fetch(self, asset: str, start: Optional[pd.Timestamp]) -> pd.DataFrame:
        """Return rows with columns ``date`` (datetime) and ``price`` (float) newer than ``start``."""


class FredProvider(MarketDataProvider):
    name = "fred"
    assets = ("brent", "wti")
    SERIES = {"brent": "DCOILBRENTEU", "wti": "DCOILWTICO"}
    URL = "https://api.stlouisfed.org/fred/series/observations"

    def is_configured(self) -> bool:
        return bool(os.getenv("FRED_API_KEY"))

    def fetch(self, asset: str, start: Optional[pd.Timestamp]) -> pd.DataFrame:
        import requests

        key = os.getenv("FRED_API_KEY")
        if not key:
            raise ProviderUnavailable("FRED_API_KEY is not set.")
        params = {"series_id": self.SERIES[asset], "api_key": key, "file_type": "json"}
        if start is not None:
            params["observation_start"] = (start + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
        try:
            r = requests.get(self.URL, params=params, timeout=30)
            r.raise_for_status()
        except Exception as exc:  # network down, bad key, rate limit ...
            raise ProviderUnavailable(f"FRED request failed: {type(exc).__name__}") from exc
        obs = pd.DataFrame(r.json().get("observations", []))
        if obs.empty:
            return pd.DataFrame(columns=["date", "price"])
        out = pd.DataFrame({"date": pd.to_datetime(obs["date"]), "price": pd.to_numeric(obs["value"], errors="coerce")})
        return out.dropna().reset_index(drop=True)


PROVIDERS: Dict[str, Type[MarketDataProvider]] = {"fred": FredProvider}


def ingest_provider(name: str, db_path: Optional[Path] = None) -> Dict[str, Optional[str]]:
    """Incrementally download new rows into ``raw_provider_<name>_<asset>``; return latest date per asset."""
    if name not in PROVIDERS:
        raise ProviderUnavailable(f"Unknown provider '{name}'. Available: {sorted(PROVIDERS)}")
    prov = PROVIDERS[name]()
    if not prov.is_configured():
        raise ProviderUnavailable(f"Provider '{name}' is not configured (missing credentials in environment).")
    from src.config import FEATURE_STORE_PATH

    db_path = Path(db_path) if db_path else FEATURE_STORE_PATH
    latest: Dict[str, Optional[str]] = {}
    for asset in prov.assets:
        table = f"raw_provider_{name}_{asset}"
        start = None
        if db_path.exists():
            con = _connect(db_path, read_only=True)
            try:
                if table in list_tables(con):
                    start = pd.Timestamp(con.execute(f'SELECT max(date) FROM "{table}"').fetchone()[0])
            finally:
                con.close()
        new = prov.fetch(asset, start)
        if len(new):
            new["downloaded_at"] = datetime.now(timezone.utc).replace(tzinfo=None)
            append_rows(new, table, db_path)
        logger.info("[%s/%s] %d new rows", name, asset, len(new))
        latest[asset] = str(new["date"].max().date()) if len(new) else (str(start.date()) if start is not None else None)
    return latest


# =========================================================================== #
# Yahoo Finance: full OHLC for all three model inputs                          #
# =========================================================================== #
YAHOO_URL = "https://query1.finance.yahoo.com/v8/finance/chart/{symbol}"
YAHOO_DEFAULT_SYMBOLS = {"brent": "BZ=F", "wti": "CL=F", "dxy": "DX-Y.NYB"}
YAHOO_ENV_NAMES = {"brent": "YAHOO_BRENT_SYMBOL", "wti": "YAHOO_WTI_SYMBOL",
                   "dxy": "YAHOO_DXY_SYMBOL"}
# A single-day move bigger than this is treated as a bad tick and the row is rejected.
MAX_DAILY_MOVE = {"brent": 0.25, "wti": 0.25, "dxy": 0.05}
OVERLAP_DAYS = 10  # re-download this many days before the CSV's last date so the two sources can be compared


def yahoo_symbols() -> Dict[str, str]:
    return {a: os.getenv(YAHOO_ENV_NAMES[a], d) for a, d in YAHOO_DEFAULT_SYMBOLS.items()}


def parse_yahoo_chart(payload: dict) -> pd.DataFrame:
    """Turn a Yahoo ``chart`` JSON response into rows ``date, price, open, high, low`` (may contain NaN)."""
    chart = (payload or {}).get("chart") or {}
    if chart.get("error"):
        raise ProviderUnavailable(f"Yahoo returned an error: {chart['error'].get('description', chart['error'])}")
    results = chart.get("result") or []
    if not results:
        return pd.DataFrame(columns=["date", "price", "open", "high", "low"])
    res = results[0]
    stamps = res.get("timestamp") or []
    quote_ = ((res.get("indicators") or {}).get("quote") or [{}])[0]
    if not stamps or not quote_:
        return pd.DataFrame(columns=["date", "price", "open", "high", "low"])
    tz = (res.get("meta") or {}).get("exchangeTimezoneName") or "UTC"
    try:
        dates = pd.to_datetime(stamps, unit="s", utc=True).tz_convert(tz).tz_localize(None).normalize()
    except Exception:  # unknown tz name
        dates = pd.to_datetime(stamps, unit="s", utc=True).tz_localize(None).normalize()
    out = pd.DataFrame({"date": dates})
    for col, key in (("price", "close"), ("open", "open"), ("high", "high"), ("low", "low")):
        vals = quote_.get(key) or [None] * len(stamps)
        out[col] = pd.to_numeric(pd.Series(vals), errors="coerce").to_numpy()
    return out


class YahooProvider:
    name = "yahoo"
    assets = ("brent", "wti", "dxy")
    has_ohlc = True

    def __init__(self, retries: int = 3, pause: float = 1.5):
        self.retries, self.pause = retries, pause

    def is_configured(self) -> bool:  # no key needed
        return True

    def fetch_ohlc(self, asset: str, start: Optional[pd.Timestamp]) -> pd.DataFrame:
        import requests

        symbol = yahoo_symbols()[asset]
        p1 = int((pd.Timestamp(start) if start is not None else pd.Timestamp.now() - pd.Timedelta(days=60)).timestamp())
        params = {"period1": p1, "period2": int(time.time()) + 86400, "interval": "1d", "events": "history"}
        headers = {"User-Agent": "Mozilla/5.0 (compatible; brent-pipeline/3.0)"}
        last: Optional[str] = None
        for attempt in range(self.retries):
            try:
                r = requests.get(YAHOO_URL.format(symbol=quote(symbol, safe="")), params=params, headers=headers, timeout=30)
                if r.status_code == 200:
                    return parse_yahoo_chart(r.json())
                last = f"HTTP {r.status_code}"
                if r.status_code not in (429, 500, 502, 503, 504):
                    break  # not worth retrying (e.g. 404 = unknown symbol)
            except ProviderUnavailable:
                raise
            except Exception as exc:  # network down, timeout, bad JSON
                last = type(exc).__name__
            time.sleep(self.pause * (attempt + 1))
        raise ProviderUnavailable(f"Yahoo request for '{symbol}' failed ({last})")


def validate_provider_frame(df: pd.DataFrame, asset: str, today: Optional[pd.Timestamp] = None) -> Tuple[pd.DataFrame, List[str]]:
    """
    Keep only trustworthy, COMPLETED daily rows.  Returns (clean rows, human-readable issues).

    Dropped: missing/non-positive close, today's or future bars (a session that has not finished),
    High < Low, Brent rows without High/Low (needed by the model), duplicate dates (last kept) and
    single-day jumps above ``MAX_DAILY_MOVE`` (bad ticks).
    """
    from src.sessions import today_in_market

    today = today_in_market() if today is None else pd.Timestamp(today).normalize()
    issues: List[str] = []
    d = df.copy()
    d["date"] = pd.to_datetime(d["date"]).dt.normalize()

    def drop(mask, why):
        nonlocal d
        n = int(mask.sum())
        if n:
            issues.append(f"{n} row(s) dropped: {why}")
            d = d[~mask]

    drop(d["price"].isna() | (d["price"] <= 0), "missing or non-positive close")
    drop(d["date"] >= today, "bar for today/future is not a completed session")
    drop((d["high"] < d["low"]).fillna(False), "High < Low")
    if asset == "brent":
        drop(d["high"].isna() | d["low"].isna(), "Brent needs High and Low")
    d = d.sort_values("date")
    n_dup = int(d["date"].duplicated().sum())
    if n_dup:
        issues.append(f"{n_dup} duplicate date(s) removed (last kept)")
        d = d.drop_duplicates("date", keep="last")
    move = d["price"].pct_change().abs()
    drop((move > MAX_DAILY_MOVE.get(asset, 0.25)).fillna(False), f"daily move above {MAX_DAILY_MOVE.get(asset, 0.25):.0%} (bad tick?)")
    return d.reset_index(drop=True), issues


def overlap_difference(csv_df: pd.DataFrame, prov_df: pd.DataFrame, prefix: str) -> Optional[float]:
    """Median |provider/CSV - 1| of the closing price on dates present in both (None if < 3 common dates)."""
    p = prov_df.set_index("date")["price"]
    common = csv_df.index.intersection(p.index)
    if len(common) < 3:
        return None
    return float((p.loc[common] / csv_df.loc[common, f"{prefix}_price"] - 1).abs().median())


def _provider_table(asset: str) -> str:
    return f"raw_provider_yahoo_{asset}"


def refresh_yahoo(
    db_path: Optional[Path] = None,
    raw_dir: Optional[Path] = None,
    today: Optional[pd.Timestamp] = None,
    provider: Optional[YahooProvider] = None,
    dry_run: bool = False,
) -> Dict[str, dict]:
    """
    Download recent completed sessions for all three inputs and store the validated rows in
    ``raw_provider_yahoo_<asset>`` (upsert by date).  NEVER raises for a provider problem: the outcome
    is returned per asset as ``{"ok", "message", "rows_new", "latest"}`` so the pipeline can carry on
    with the data it already has and the dashboard can say so honestly.
    ``dry_run=True`` checks everything but stores nothing (used by ``python -m src.providers``).
    """
    from src.config import FEATURE_STORE_PATH, RAW_DATA_DIR
    from src.ingestion import ASSET_FILES, clean_financial_csv, find_csv

    db_path = Path(db_path) if db_path else FEATURE_STORE_PATH
    raw_dir = Path(raw_dir) if raw_dir else RAW_DATA_DIR
    provider = provider or YahooProvider()
    max_diff = float(os.getenv("PROVIDER_MAX_OVERLAP_DIFF", "0.05"))
    status: Dict[str, dict] = {}
    for asset in provider.assets:
        try:
            csv_df = clean_financial_csv(find_csv(raw_dir, ASSET_FILES[asset]), asset)
            start = csv_df.index.max() - pd.Timedelta(days=OVERLAP_DAYS)
            clean, issues = validate_provider_frame(provider.fetch_ohlc(asset, start), asset, today)
            if clean.empty:
                raise ProviderUnavailable("no completed sessions returned" + (f" ({'; '.join(issues)})" if issues else ""))
            diff = overlap_difference(csv_df, clean, asset)
            if diff is None:
                raise ProviderUnavailable("could not be compared with the existing history (too few overlapping dates), so it was not used")
            if diff > max_diff:
                raise ProviderUnavailable(f"prices differ from the existing history by {diff:.1%} (limit {max_diff:.0%}); "
                                          f"wrong symbol '{yahoo_symbols()[asset]}'? Not used")
            if dry_run:
                status[asset] = {"ok": True, "message": "; ".join(issues), "rows_new": 0, "latest": str(clean["date"].max().date()),
                                 "symbol": yahoo_symbols()[asset], "overlap_diff": diff}
                continue
            stored = None
            try:
                stored = read_table(_provider_table(asset), db_path).reset_index()
            except FileNotFoundError:
                pass
            clean["downloaded_at"] = datetime.now(timezone.utc).replace(tzinfo=None)
            merged = clean if stored is None else pd.concat([stored, clean], ignore_index=True)
            merged = merged.drop_duplicates("date", keep="last").sort_values("date")
            n_new = len(set(clean["date"]) - (set(stored["date"]) if stored is not None else set()))
            persist_dataframe(merged.set_index("date"), _provider_table(asset), db_path)
            status[asset] = {"ok": True, "message": "; ".join(issues), "rows_new": n_new, "latest": str(clean["date"].max().date())}
        except ProviderUnavailable as exc:
            status[asset] = {"ok": False, "message": str(exc), "rows_new": 0, "latest": None, "symbol": yahoo_symbols()[asset]}
        except Exception as exc:  # noqa: BLE001  (a provider problem must never crash the job)
            logger.exception("Unexpected error refreshing %s", asset)
            status[asset] = {"ok": False, "message": f"unexpected error: {type(exc).__name__}", "rows_new": 0, "latest": None}
        logger.info("[yahoo/%s] %s", asset, status[asset])
    return status


def load_provider_frames(db_path: Optional[Path] = None) -> Dict[str, pd.DataFrame]:
    """Stored, validated Yahoo rows per asset (index = date; columns price/open/high/low). Empty if none."""
    out: Dict[str, pd.DataFrame] = {}
    for asset in YahooProvider.assets:
        try:
            out[asset] = read_table(_provider_table(asset), db_path)[["price", "open", "high", "low"]]
        except FileNotFoundError:
            continue
    return out


def main() -> int:
    """``python -m src.providers``: test every Yahoo symbol against the CSV history; stores nothing."""
    import sys

    logging.basicConfig(level=logging.WARNING, format="%(levelname)s | %(message)s")
    st = refresh_yahoo(dry_run=True)
    print(f"{'input':<8}{'symbol':<12}{'result':<8}{'latest completed session':<26}details")
    for asset, r in st.items():
        sym = r.get("symbol") or yahoo_symbols()[asset]
        detail = r["message"] or (f"prices match CSV history within {r['overlap_diff']:.2%} (median)" if r.get("overlap_diff") is not None else "")
        print(f"{asset:<8}{sym:<12}{'OK' if r['ok'] else 'FAILED':<8}{(r['latest'] or '-'):<26}{detail}")
    bad = [a for a, r in st.items() if not r["ok"]]
    print("\nAll three feeds work." if not bad else f"\nNOT working: {', '.join(bad)}. The model needs all three inputs; fix the symbol in .env (YAHOO_*_SYMBOL).")
    return 1 if bad else 0


if __name__ == "__main__":
    import sys

    sys.exit(main())
