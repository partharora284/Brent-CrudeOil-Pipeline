import json

import joblib
import pytest

from src.features import FEATURE_COLUMNS, load_feature_matrix
from src.registry import ModelRegistry, RegistryError


def test_training_registers_a_version_consistent_with_gate(trained_env):
    meta, reg = trained_env["meta"], ModelRegistry(trained_env["models"])
    v = meta["version"]
    assert reg.metadata(v)["model_name"] == meta["model_name"]
    status = {x["version"]: x for x in reg.list_versions()}[v]["status"]
    if meta["validation"]["passed"]:
        assert reg.approved_version() == v and status == "approved"
    else:  # an unvalidated / failing model must NOT be auto-promoted
        assert reg.approved_version() is None and status == "rejected"


def test_metadata_contains_required_fields(trained_env):
    m = ModelRegistry(trained_env["models"]).metadata(trained_env["meta"]["version"])
    for k in ("model_name", "version", "trained_at", "feature_schema", "target", "metrics", "selection", "validation",
              "comparison", "baselines"):
        assert k in m
    assert m["feature_schema"]["columns"] == FEATURE_COLUMNS
    for name in ("logistic_regression", "ridge_classifier", "xgboost", "lightgbm"):
        c = m["comparison"][name]
        assert {"standard_accuracy", "precision", "recall", "f1", "confusion_matrix"} <= set(c)
    assert {"always_up", "majority_class_train"} <= set(m["baselines"])
    json.dumps(m)  # strictly JSON-serialisable


def test_rejected_model_cannot_be_promoted_and_load_roundtrip(trained_env):
    reg = ModelRegistry(trained_env["models"])
    v = trained_env["meta"]["version"]
    if not trained_env["meta"]["validation"]["passed"]:
        with pytest.raises(RegistryError):
            reg.promote(v)
        reg.promote(v, force=True)  # only to exercise the loader below
    loaded = reg.load()
    X, _ = load_feature_matrix(trained_env["db"])
    p = loaded.predict_proba_up(X.tail(5).sample(frac=1, random_state=1)[FEATURE_COLUMNS[::-1]])  # shuffled cols
    assert p.shape == (5,) and ((p >= 0) & (p <= 1)).all()


def test_preprocessor_is_fitted_not_schema_only(trained_env):
    pre = joblib.load(trained_env["models"] / "registry" / "versions" / trained_env["meta"]["version"] / "preprocessor.joblib")
    assert hasattr(pre, "transform") and pre.fitted_ and pre.feature_columns == FEATURE_COLUMNS
    with pytest.raises(ValueError):
        pre.transform(load_feature_matrix(trained_env["db"])[0].drop(columns=[FEATURE_COLUMNS[0]]))


def test_corrupt_artifact_is_refused_and_rollback_works(trained_env, tmp_path):
    import shutil

    models = tmp_path / "m"
    shutil.copytree(trained_env["models"], models)
    reg = ModelRegistry(models)
    v1 = trained_env["meta"]["version"]
    if reg.approved_version() is None:
        reg.promote(v1, force=True)
    # second version = same artefacts re-registered, then promote and roll back
    l = reg.load()
    v2 = reg.register(l.model, l.preprocessor, {**l.metadata, "validation": {"passed": True}})
    reg.promote(v2)
    assert reg.approved_version() == v2
    assert reg.rollback() == v1 and reg.approved_version() == v1
    (models / "registry" / "versions" / v1 / "model.joblib").write_bytes(b"corrupt")
    with pytest.raises(RegistryError):
        reg.load(v1)
