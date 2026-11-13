"""Metrics: how wrong are the predictions, on average?

* **RMSE** (root mean squared error) - typical size of an error, in log units.
  Big mistakes count extra. RMSE = 1.0 means "typically off by a factor of 10".
* **MAE** (mean absolute error) - the plain average size of the errors.
* **R²** - the share of the variation in solubility the model explains
  (1.0 = perfect, 0 = no better than always guessing the average).
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def metrics(y_true, y_pred) -> dict[str, float]:
    """RMSE, MAE and R² rounded to 3 decimals."""
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    return {
        "rmse": round(float(np.sqrt(mean_squared_error(y_true, y_pred))), 3),
        "mae": round(float(mean_absolute_error(y_true, y_pred)), 3),
        "r2": round(float(r2_score(y_true, y_pred)), 3),
    }
