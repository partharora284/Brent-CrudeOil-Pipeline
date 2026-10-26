"""
src/scheduled_ingest.py  --  One-shot (or looping) data refresh for Windows Task Scheduler / cron
=================================================================================================

    python -m src.scheduled_ingest [--no-features] [--provider yahoo|fred|none] [--loop --every-minutes 60]

1. Live download of the latest COMPLETED sessions for Brent, WTI and DXY
   (``--provider``, default from DATA_PROVIDER, default ``yahoo``).  Every row is validated and
   cross-checked against the CSV history (src/providers.py).  A provider failure is logged and
   shown to the user; it never stops the job and never produces invented data.
2. CSV + stored live rows -> validate -> aligned master table + ``data_status``.
3. Feature engineering -> feature_matrix + inference_features.  The API predicts from these tables
   on every request, so the prediction updates as soon as this job has run.

Exit code: 0 = fully successful, 2 = finished but the live provider failed for >= 1 input
(data may be outdated), 1 = the job itself failed.  Output is appended to logs/ingestion.log.
The model is NOT retrained here; retrain explicitly with ``python -m src.train``.

``--loop`` keeps the process running and repeats every ``--every-minutes`` (Ctrl+C to stop), as an
alternative to Task Scheduler.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from logging.handlers import RotatingFileHandler

from src.config import PROJECT_ROOT


def run_once(provider: str, no_features: bool) -> int:
    log = logging.getLogger("scheduled_ingest")
    try:
        status = None
        provider_failed = False
        if provider == "yahoo":
            from src.providers import refresh_yahoo

            status = refresh_yahoo()
            for asset, st in status.items():
                if st["ok"]:
                    log.info("Live data OK  | %-6s latest completed session %s (%d new rows)", asset, st["latest"], st["rows_new"])
                else:
                    provider_failed = True
                    log.warning("Live data FAILED | %-6s %s", asset, st["message"])
        elif provider == "fred":
            from src.providers import ProviderUnavailable, ingest_provider

            try:
                log.info("Provider 'fred' latest dates: %s", ingest_provider("fred"))
            except ProviderUnavailable as exc:
                log.warning("Provider download skipped (%s). Continuing with stored data.", exc)
        from src.features import run_feature_pipeline
        from src.ingestion import run_ingestion

        if no_features:
            df = run_ingestion(save=True, provider_status=status)
            log.info("Ingestion complete: %d rows through %s", len(df), df.index.max().date())
        else:
            run_feature_pipeline(provider_status=status)
            log.info("Ingestion + feature pipeline complete.")
        return 2 if provider_failed else 0
    except Exception:
        log.exception("Scheduled ingestion FAILED")
        return 1


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1] if __doc__ else "")
    ap.add_argument("--provider", default=os.getenv("DATA_PROVIDER", "yahoo"), help="yahoo (default), fred or none")
    ap.add_argument("--no-features", action="store_true", help="only refresh raw/cleaned tables")
    ap.add_argument("--loop", action="store_true", help="repeat forever instead of running once")
    ap.add_argument("--every-minutes", type=float, default=60.0, help="pause between runs with --loop")
    args = ap.parse_args(argv)
    provider = (args.provider or "none").strip().lower()
    if provider in ("", "none"):
        provider = "none"

    log_dir = PROJECT_ROOT / "logs"
    log_dir.mkdir(exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s | %(message)s",
        handlers=[RotatingFileHandler(log_dir / "ingestion.log", maxBytes=1_000_000, backupCount=3, encoding="utf-8"),
                  logging.StreamHandler(sys.stdout)],
    )
    if not args.loop:
        return run_once(provider, args.no_features)
    try:
        while True:
            rc = run_once(provider, args.no_features)
            logging.getLogger("scheduled_ingest").info("Run finished with exit code %d; next run in %.0f min.", rc, args.every_minutes)
            time.sleep(max(args.every_minutes, 1.0) * 60)
    except KeyboardInterrupt:
        return 0


if __name__ == "__main__":
    sys.exit(main())
