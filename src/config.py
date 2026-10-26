"""Shared configuration: project root, .env loading and path resolution."""

from __future__ import annotations

import os
import sys
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:  # allow `python src/xyz.py`
    sys.path.insert(0, str(PROJECT_ROOT))

load_dotenv(PROJECT_ROOT / ".env")


def resolve_path(path_str: str) -> Path:
    """Resolve a (possibly relative) path from .env against the project root."""
    p = Path(path_str)
    return p if p.is_absolute() else PROJECT_ROOT / p


RAW_DATA_DIR: Path = resolve_path(os.getenv("RAW_DATA_DIR", "data/raw"))
FEATURE_STORE_PATH: Path = resolve_path(os.getenv("FEATURE_STORE_PATH", "data/feature_store.duckdb"))
MODEL_DIR: Path = resolve_path(os.getenv("MODEL_DIR", "models"))
REPORT_DIR: Path = resolve_path(os.getenv("REPORT_DIR", "data/quality"))
