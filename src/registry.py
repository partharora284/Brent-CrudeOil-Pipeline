"""
src/registry.py  --  Lightweight file-based model registry
===========================================================

Layout (under ``MODEL_DIR/registry``)::

    registry.json                 index: approved version, approval history, per-version status
    versions/v0001/model.joblib
                  /preprocessor.joblib
                  /metadata.json   name, version, trained_at, feature schema, target,
                                   metrics, selection rationale, validation gate, file hashes

* ``register`` stores a new immutable version.  It is marked ``approved`` ONLY by an
  explicit ``promote``; training promotes automatically only if the validation gate passed.
* A version that failed the gate is stored as ``rejected`` and cannot be promoted
  (unless ``force=True``, which is logged and never used by the training script).
* ``load`` re-verifies file hashes, the feature-schema fingerprint, the target definition
  (must be the current ``target_{H}d_class``) and runs a smoke prediction, so a corrupt, stale
  or incompatible artefact is refused.
* ``load(allow_unvalidated=True)`` (API setting ``ALLOW_UNVALIDATED_MODEL=1``) may fall back to the
  NEWEST *rejected* version when nothing is approved.  It goes through exactly the same checks, and the
  returned ``LoadedModel.experimental`` is True so the API/dashboard can say so.  Old-target models
  and corrupt artefacts are never loaded this way.
* Version numbers come from BOTH the index and the folders on disk, so an orphan folder left by an
  interrupted run (e.g. ``v0002``) can never collide with a new registration.
* The currently approved model + preprocessor are mirrored to ``MODEL_DIR/best_model.joblib``
  and ``preprocessor.joblib`` (the filenames used in the architecture diagram).

CLI::

    python -m src.registry list
    python -m src.registry promote v0002
    python -m src.registry rollback [--to v0001]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import shutil
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import joblib
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class RegistryError(RuntimeError):
    """Missing, corrupt, unapproved or schema-incompatible model artefacts."""


@dataclass
class LoadedModel:
    version: str
    model: Any
    preprocessor: Any
    metadata: Dict[str, Any]
    experimental: bool = False  # True = loaded although it did NOT pass the validation gate

    def predict_proba_up(self, X: pd.DataFrame) -> np.ndarray:
        return self.model.predict_proba(self.preprocessor.transform(X))[:, 1]


def _sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _atomic_write(path: Path, text: str) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


class ModelRegistry:
    def __init__(self, model_dir: str | Path):
        self.model_dir = Path(model_dir)
        self.root = self.model_dir / "registry"
        self.index_path = self.root / "registry.json"

    # ------------------------------------------------------------------ index
    def _read(self) -> Dict[str, Any]:
        if not self.index_path.exists():
            return {"approved": None, "approved_history": [], "versions": {}}
        return json.loads(self.index_path.read_text(encoding="utf-8"))

    def _write(self, idx: Dict[str, Any]) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        _atomic_write(self.index_path, json.dumps(idx, indent=2))

    def approved_version(self) -> Optional[str]:
        return self._read()["approved"]

    def list_versions(self) -> List[Dict[str, Any]]:
        idx = self._read()
        return [{"version": v, "approved": v == idx["approved"], **info} for v, info in sorted(idx["versions"].items())]

    def metadata(self, version: str) -> Dict[str, Any]:
        p = self.root / "versions" / version / "metadata.json"
        if not p.exists():
            raise RegistryError(f"Unknown model version '{version}'.")
        return json.loads(p.read_text(encoding="utf-8"))

    # --------------------------------------------------------------- register
    def register(self, model: Any, preprocessor: Any, metadata: Dict[str, Any]) -> str:
        """Store a new immutable version; returns its id. Does NOT approve it."""
        idx = self._read()
        vroot = self.root / "versions"
        on_disk = [int(p.name[1:]) for p in vroot.glob("v[0-9]*") if p.name[1:].isdigit()] if vroot.exists() else []
        n = max([int(v[1:]) for v in idx["versions"]] + on_disk + [0]) + 1
        version = f"v{n:04d}"
        vdir = vroot / version
        vdir.mkdir(parents=True, exist_ok=False)
        try:
            joblib.dump(model, vdir / "model.joblib")
            joblib.dump(preprocessor, vdir / "preprocessor.joblib")
            passed = bool(metadata.get("validation", {}).get("passed", False))
            meta = dict(metadata)
            meta.update(
                version=version,
                registered_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
                artifacts={"model.joblib": _sha256(vdir / "model.joblib"),
                           "preprocessor.joblib": _sha256(vdir / "preprocessor.joblib")},
            )
            _atomic_write(vdir / "metadata.json", json.dumps(meta, indent=2, default=str))
        except Exception:  # never leave a half-written version folder behind
            shutil.rmtree(vdir, ignore_errors=True)
            raise
        idx["versions"][version] = {
            "status": "candidate" if passed else "rejected",
            "model_name": meta.get("model_name"),
            "trained_at": meta.get("trained_at"),
            "validation_passed": passed,
        }
        self._write(idx)
        logger.info("Registered %s (%s) status=%s", version, meta.get("model_name"), idx["versions"][version]["status"])
        return version

    # ---------------------------------------------------------- promote / back
    def promote(self, version: str, force: bool = False) -> None:
        """Make ``version`` the approved model (after load-time compatibility checks)."""
        idx = self._read()
        if version not in idx["versions"]:
            raise RegistryError(f"Unknown model version '{version}'.")
        if not idx["versions"][version]["validation_passed"] and not force:
            raise RegistryError(f"{version} failed the validation gate and cannot be promoted.")
        self._load_dir(version)  # raises if corrupt / incompatible
        prev = idx["approved"]
        if prev and prev != version:
            idx["versions"][prev]["status"] = "retired"
        idx["approved"] = version
        idx["versions"][version]["status"] = "approved"
        if not idx["approved_history"] or idx["approved_history"][-1] != version:
            idx["approved_history"].append(version)
        self._write(idx)
        self._sync_current(version)
        logger.info("Approved model is now %s%s", version, " (FORCED)" if force else "")

    def rollback(self, to: Optional[str] = None) -> str:
        """Re-approve the previously approved version (or ``to``). Returns the new approved id."""
        idx = self._read()
        if to is None:
            hist = [v for v in idx["approved_history"] if v != idx["approved"]]
            if not hist:
                raise RegistryError("No previous approved version to roll back to.")
            to = hist[-1]
        if to not in idx["versions"]:
            raise RegistryError(f"Unknown model version '{to}'.")
        self._load_dir(to)
        cur = idx["approved"]
        if cur and cur != to:
            idx["versions"][cur]["status"] = "retired"
        idx["approved"] = to
        idx["versions"][to]["status"] = "approved"
        idx["approved_history"] = [v for v in idx["approved_history"] if v != cur] or [to]
        if idx["approved_history"][-1] != to:
            idx["approved_history"].append(to)
        self._write(idx)
        self._sync_current(to)
        logger.info("Rolled back %s -> %s", cur, to)
        return to

    def _sync_current(self, version: str) -> None:
        """Mirror the approved artefacts to MODEL_DIR/best_model.joblib etc. (diagram filenames)."""
        vdir = self.root / "versions" / version
        shutil.copy2(vdir / "model.joblib", self.model_dir / "best_model.joblib")
        shutil.copy2(vdir / "preprocessor.joblib", self.model_dir / "preprocessor.joblib")
        meta = json.loads((vdir / "metadata.json").read_text(encoding="utf-8"))
        _atomic_write(self.model_dir / "metrics.json", json.dumps(meta.get("legacy_metrics", meta.get("metrics", {})), indent=2))

    # ------------------------------------------------------------------- load
    def load(self, version: Optional[str] = None, allow_unvalidated: bool = False) -> LoadedModel:
        """
        Load the approved (or a named) version after full compatibility validation.

        With ``allow_unvalidated=True`` and NO approved version, fall back to the newest ``rejected``
        version that passes every load-time check (hashes, schema, target, smoke prediction).
        The result has ``experimental=True``.  Anything that fails a check is skipped, never forced.
        """
        version = version or self.approved_version()
        if version:
            return self._load_dir(version)
        if not allow_unvalidated:
            raise RegistryError("No approved model in the registry. Run `python -m src.train`.")
        problems = []
        rejected = [v for v, info in self._read()["versions"].items() if info.get("status") == "rejected"]
        for v in sorted(rejected, reverse=True):  # newest first
            try:
                loaded = self._load_dir(v)
            except RegistryError as exc:
                problems.append(f"{v}: {exc}")
                continue
            loaded.experimental = True
            return loaded
        raise RegistryError("No approved model, and no rejected model is complete and compatible with the current "
                            "daily target. Run `python -m src.train`." + (f" Skipped: {'; '.join(problems)}" if problems else ""))

    def _load_dir(self, version: str) -> LoadedModel:
        from src.features import FEATURE_COLUMNS, FEATURE_SCHEMA_HASH, HORIZON, TARGET_COL

        vdir = self.root / "versions" / version
        meta_path = vdir / "metadata.json"
        if not meta_path.exists():
            raise RegistryError(f"Artefacts for {version} not found under {vdir}.")
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if not meta.get("artifacts") or any(not (vdir / f).exists() for f in ("model.joblib", "preprocessor.joblib")):
            raise RegistryError(f"{version} is incomplete (model/preprocessor file or recorded checksums missing).")
        for fname, digest in meta.get("artifacts", {}).items():
            if not (vdir / fname).exists() or _sha256(vdir / fname) != digest:
                raise RegistryError(f"{version}/{fname} is missing or does not match its recorded hash (corrupt artefact).")
        tgt = meta.get("target", {})
        if tgt.get("column") != TARGET_COL or tgt.get("horizon_days") != HORIZON:
            raise RegistryError(f"{version} was trained for target '{tgt.get('column')}' (horizon {tgt.get('horizon_days')}), "
                                f"but the system now uses '{TARGET_COL}' (horizon {HORIZON}). It is not compatible; retrain.")
        if meta.get("feature_schema", {}).get("hash") != FEATURE_SCHEMA_HASH:
            raise RegistryError(
                f"{version} was trained on feature schema {meta.get('feature_schema', {}).get('hash')} but the code "
                f"now defines {FEATURE_SCHEMA_HASH}. Retrain with `python -m src.train`."
            )
        model = joblib.load(vdir / "model.joblib")
        pre = joblib.load(vdir / "preprocessor.joblib")
        if not hasattr(pre, "transform") or getattr(pre, "feature_columns", None) != FEATURE_COLUMNS:
            raise RegistryError(f"{version} preprocessor is not a fitted transformer with the current feature order.")
        try:  # smoke test: the pair must actually score a frame end-to-end
            probe = pd.DataFrame([{c: 1.0 for c in FEATURE_COLUMNS}] * 2)
            p = model.predict_proba(pre.transform(probe))
            assert p.shape == (2, 2) and np.isfinite(p).all()
        except Exception as exc:
            raise RegistryError(f"{version} failed the load-time smoke prediction: {exc}") from exc
        return LoadedModel(version=version, model=model, preprocessor=pre, metadata=meta)


def main(argv=None) -> int:
    from src.config import MODEL_DIR

    ap = argparse.ArgumentParser(description="Model registry admin")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("list")
    sp = sub.add_parser("promote"); sp.add_argument("version")
    sr = sub.add_parser("rollback"); sr.add_argument("--to", default=None)
    args = ap.parse_args(argv)
    reg = ModelRegistry(MODEL_DIR)
    try:
        if args.cmd == "list":
            for v in reg.list_versions():
                print(f"{'*' if v['approved'] else ' '} {v['version']}  {v['status']:<9} {v['model_name']}  trained {v['trained_at']}")
        elif args.cmd == "promote":
            reg.promote(args.version); print(f"Approved: {args.version}")
        else:
            print(f"Approved after rollback: {reg.rollback(args.to)}")
        return 0
    except RegistryError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
