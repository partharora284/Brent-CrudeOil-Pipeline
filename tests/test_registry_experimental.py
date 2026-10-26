"""Registry: version numbering, strict target/artifact checks and the opt-in experimental loader."""

import json

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression

from src.features import FEATURE_COLUMNS, FEATURE_SCHEMA_HASH, HORIZON, TARGET_COL
from src.preprocess import FeaturePreprocessor
from src.registry import ModelRegistry, RegistryError


def _artifacts():
    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.normal(size=(80, len(FEATURE_COLUMNS))), columns=FEATURE_COLUMNS)
    y = (rng.random(80) > 0.5).astype(int)
    return LogisticRegression(max_iter=200).fit(X, y), FeaturePreprocessor(FEATURE_COLUMNS, FEATURE_SCHEMA_HASH).fit(X)


def _meta(passed, column=TARGET_COL, horizon=HORIZON, schema=FEATURE_SCHEMA_HASH):
    return {"model_name": "logistic_regression", "trained_at": "2026-10-25T00:00:00+00:00",
            "feature_schema": {"columns": FEATURE_COLUMNS, "hash": schema},
            "target": {"column": column, "horizon_days": horizon, "definition": "x"},
            "validation": {"passed": passed}}


def test_orphan_folder_does_not_collide_with_new_version(tmp_path):
    reg = ModelRegistry(tmp_path)
    (tmp_path / "registry" / "versions" / "v0002").mkdir(parents=True)  # left over from an interrupted run
    m, p = _artifacts()
    assert reg.register(m, p, _meta(False)) == "v0003"
    assert reg.register(m, p, _meta(False)) == "v0004"


def test_strict_mode_never_loads_a_rejected_model(tmp_path):
    reg = ModelRegistry(tmp_path)
    reg.register(*_artifacts(), _meta(False))
    with pytest.raises(RegistryError, match="No approved model"):
        reg.load()
    with pytest.raises(RegistryError):
        reg.promote("v0001")  # rejected models still cannot be approved


def test_experimental_mode_loads_newest_rejected_and_flags_it(tmp_path):
    reg = ModelRegistry(tmp_path)
    reg.register(*_artifacts(), _meta(False))
    v2 = reg.register(*_artifacts(), _meta(False))
    got = reg.load(allow_unvalidated=True)
    assert got.version == v2 and got.experimental is True
    assert got.predict_proba_up(pd.DataFrame([{c: 0.1 for c in FEATURE_COLUMNS}])).shape == (1,)


def test_approved_model_wins_over_experimental_and_is_not_flagged(tmp_path):
    reg = ModelRegistry(tmp_path)
    reg.register(*_artifacts(), _meta(False))
    ok = reg.register(*_artifacts(), _meta(True))
    reg.promote(ok)
    got = reg.load(allow_unvalidated=True)
    assert got.version == ok and got.experimental is False


def test_old_three_day_model_is_never_used_as_a_substitute(tmp_path):
    reg = ModelRegistry(tmp_path)
    reg.register(*_artifacts(), _meta(False, column="target_3d_class", horizon=3, schema="d87ac0bb5394aef6"))
    with pytest.raises(RegistryError, match="not compatible|No approved|no rejected"):
        reg.load(allow_unvalidated=True)
    # same schema hash but a wrong target label is still refused
    reg.register(*_artifacts(), _meta(False, column="target_3d_class", horizon=3))
    with pytest.raises(RegistryError):
        reg.load("v0002")


def test_experimental_mode_skips_corrupt_and_incomplete_versions(tmp_path):
    reg = ModelRegistry(tmp_path)
    good = reg.register(*_artifacts(), _meta(False))
    bad = reg.register(*_artifacts(), _meta(False))
    (tmp_path / "registry" / "versions" / bad / "model.joblib").write_bytes(b"corrupt")
    assert reg.load(allow_unvalidated=True).version == good  # newest is corrupt -> falls back to the intact one
    (tmp_path / "registry" / "versions" / good / "preprocessor.joblib").unlink()
    with pytest.raises(RegistryError):
        reg.load(allow_unvalidated=True)


def test_metadata_without_checksums_is_refused(tmp_path):
    reg = ModelRegistry(tmp_path)
    v = reg.register(*_artifacts(), _meta(False))
    mp = tmp_path / "registry" / "versions" / v / "metadata.json"
    meta = json.loads(mp.read_text()); meta["artifacts"] = {}
    mp.write_text(json.dumps(meta))
    with pytest.raises(RegistryError, match="incomplete"):
        reg.load(v)


def test_failed_registration_leaves_no_half_written_folder(tmp_path, monkeypatch):
    import src.registry as r

    def boom(*a, **k):
        raise OSError("disk full")

    monkeypatch.setattr(r.joblib, "dump", boom)
    with pytest.raises(OSError):
        ModelRegistry(tmp_path).register(*_artifacts(), _meta(False))
    assert not list((tmp_path / "registry" / "versions").glob("v*"))
