"""
cost_calculator.py
------------------
Cost-sensitive evaluation helpers.

Cost matrix (illustrative assumptions, NOT real company figures):
    False Negative (churner missed)      = FN cost  (default $100)
    False Positive (unneeded offer)      = FP cost  (default $10)
    True Positive / True Negative        = $0

    Estimated Total Cost = FN x FN_cost + FP x FP_cost
"""

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix

DEFAULT_FN_COST = 100.0
DEFAULT_FP_COST = 10.0

# Candidate decision thresholds: 0.01, 0.02, ..., 0.99
THRESHOLD_GRID = np.round(np.arange(0.01, 1.00, 0.01), 2)


def confusion_counts(y_true, y_pred):
    """Return (tn, fp, fn, tp) as plain Python ints."""
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return int(tn), int(fp), int(fn), int(tp)


def compute_cost(fn: int, fp: int, n_customers: int,
                 fn_cost: float = DEFAULT_FN_COST,
                 fp_cost: float = DEFAULT_FP_COST) -> dict:
    """Total cost, cost per customer, and the breakdown by error type."""
    fn_total = fn * fn_cost
    fp_total = fp * fp_cost
    total = fn_total + fp_total
    return {
        "fn_cost_total": float(fn_total),
        "fp_cost_total": float(fp_total),
        "total_cost": float(total),
        "cost_per_customer": float(total / n_customers) if n_customers else 0.0,
    }


def threshold_curve(y_true, proba, fn_cost: float = DEFAULT_FN_COST,
                    fp_cost: float = DEFAULT_FP_COST,
                    thresholds=THRESHOLD_GRID) -> pd.DataFrame:
    """Estimated cost at every candidate threshold (one row per threshold)."""
    y_true = np.asarray(y_true)
    proba = np.asarray(proba)
    rows = []
    for t in thresholds:
        y_pred = (proba >= t).astype(int)
        _, fp, fn, _ = confusion_counts(y_true, y_pred)
        rows.append({
            "threshold": float(t),
            "fn": fn,
            "fp": fp,
            "fn_cost_total": fn * fn_cost,
            "fp_cost_total": fp * fp_cost,
            "total_cost": fn * fn_cost + fp * fp_cost,
        })
    return pd.DataFrame(rows)


def select_best_threshold(curve: pd.DataFrame) -> float:
    """
    Threshold with the lowest total cost.
    If several thresholds tie, pick the one closest to 0.5 (the most conventional).
    """
    lowest = curve["total_cost"].min()
    tied = curve[np.isclose(curve["total_cost"], lowest)]
    best_row = (tied["threshold"] - 0.5).abs().idxmin()
    return float(curve.loc[best_row, "threshold"])