"""Model factories: the same recipe is used everywhere so results are comparable."""

from __future__ import annotations

from typing import Any

import lightgbm as lgb
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def make_model(kind: str, params: dict[str, Any] | None = None, seed: int = 42):
    """Create an untrained model.

    * ``"linear"`` - linear regression on standardised descriptors
    * ``"rf"`` - random forest (many decision trees, each sees a random part of the data; predictions are averaged)
    * ``"lgbm"`` - LightGBM gradient boosting (trees built one after another, each fixing the previous ones' errors)
    """
    params = dict(params or {})
    if kind == "linear":
        return make_pipeline(StandardScaler(), LinearRegression())
    if kind == "rf":
        return RandomForestRegressor(random_state=seed, n_jobs=4, **params)
    if kind == "lgbm":
        return lgb.LGBMRegressor(
            random_state=seed,
            n_jobs=4,
            subsample_freq=1,
            deterministic=True,
            force_col_wise=True,
            verbose=-1,
            **params,
        )
    raise ValueError(f"unknown model kind {kind!r}")
