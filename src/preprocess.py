"""
src/preprocess.py  --  Fitted preprocessing step shared by training and inference
=================================================================================

Tree models need no scaling, but the *validation, selection and ordering* of
features is still a preprocessing step that must be identical at train and
inference time.  ``FeaturePreprocessor`` is a fitted object (not just a column
list): ``fit`` learns the schema from the training frame and ``transform``
enforces it (missing columns -> error, order restored, float dtype, finite
values).  Scaling for linear models lives inside their sklearn ``Pipeline``.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd


class FeaturePreprocessor:
    """Schema enforcement + ordering. Serialised as ``preprocessor.joblib``."""

    def __init__(self, feature_columns: Sequence[str], schema_hash: str = ""):
        self.feature_columns = list(feature_columns)
        self.schema_hash = schema_hash
        self.fitted_ = False

    def fit(self, X: pd.DataFrame) -> "FeaturePreprocessor":
        missing = [c for c in self.feature_columns if c not in X.columns]
        if missing:
            raise ValueError(f"Training frame is missing features: {missing}")
        sub = X[self.feature_columns].astype(float)
        self.n_train_rows_ = int(len(sub))
        # Training ranges are stored for diagnostics (not used to clip predictions).
        self.train_min_ = sub.min().to_dict()
        self.train_max_ = sub.max().to_dict()
        self.fitted_ = True
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if not self.fitted_:
            raise RuntimeError("FeaturePreprocessor is not fitted.")
        missing = [c for c in self.feature_columns if c not in X.columns]
        if missing:
            raise ValueError(f"Missing feature columns: {missing}")
        out = X[self.feature_columns].apply(pd.to_numeric, errors="coerce").astype(float)
        if not np.isfinite(out.to_numpy()).all():
            raise ValueError("Feature values must be finite numbers (no nulls/NaN/inf).")
        return out

    def fit_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return self.fit(X).transform(X)
