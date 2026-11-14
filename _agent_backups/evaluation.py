"""
evaluation.py
-------------
Stages 4 and 5:
  - Choose each model's decision threshold on the VALIDATION set (lowest cost).
  - Apply that frozen threshold ONCE to the untouched TEST set.
  - Compute accuracy, precision, recall, F1, ROC-AUC, confusion matrix, cost.
  - Rank models by test cost, and report best by F1 and by ROC-AUC.
  - Save results for the dashboard.

Run with:  python -m src.evaluation
"""

import json
from pathlib import Path
import sys
from datetime import datetime

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import joblib
import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score, roc_auc_score,
)

from src.cost_calculator import (
    DEFAULT_FN_COST, DEFAULT_FP_COST, THRESHOLD_GRID,
    compute_cost, confusion_counts, select_best_threshold, threshold_curve,
)
from src.preprocessing import (
    FEATURE_COLUMNS, RANDOM_STATE, clean_data, load_data, split_data,
)

MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_PATH = MODELS_DIR / "evaluation_results.json"
PREDICTIONS_PATH = MODELS_DIR / "predictions.pkl"
METADATA_PATH = MODELS_DIR / "metadata.json"


def evaluate_at_threshold(y_true, proba, threshold, fn_cost, fp_cost) -> dict:
    """All metrics for one set of probabilities at one fixed threshold."""
    y_true = np.asarray(y_true)
    y_pred = (np.asarray(proba) >= threshold).astype(int)
    tn, fp, fn, tp = confusion_counts(y_true, y_pred)
    result = {
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, proba)),  # threshold-independent
        "tn": tn, "fp": fp, "fn": fn, "tp": tp,
    }
    result.update(compute_cost(fn, fp, n_customers=len(y_true),
                               fn_cost=fn_cost, fp_cost=fp_cost))
    return result


def main() -> None:
    # --- Check that training has been done ---
    if not METADATA_PATH.exists():
        print("models/metadata.json not found. Automatically training models first...")
        from src.train_models import main as train_main
        train_main()
    with open(METADATA_PATH) as f:
        metadata = json.load(f)

    # --- Rebuild the exact same splits (fixed random_state) ---
    try:
        raw_df = load_data()
    except (FileNotFoundError, ValueError) as err:
        print(f"ERROR: {err}")
        raise SystemExit(1)
    X, y = clean_data(raw_df)
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y)

    fn_cost, fp_cost = DEFAULT_FN_COST, DEFAULT_FP_COST
    print(f"Cost assumptions (illustrative): FN = ${fn_cost:.0f}, FP = ${fp_cost:.0f}, TP = TN = $0")
    print(f"Rule of thumb: for well-calibrated probabilities the cost-optimal threshold is "
          f"about FP/(FP+FN) = {fp_cost / (fp_cost + fn_cost):.3f}\n")

    results = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "cost_assumptions": {
            "fn_cost": fn_cost, "fp_cost": fp_cost, "tp_cost": 0.0, "tn_cost": 0.0,
            "note": "Illustrative assumptions only. The dataset has no real discount cost, "
                    "customer lifetime value, or evidence that an offer prevents churn.",
        },
        "threshold_grid": [float(t) for t in THRESHOLD_GRID],
        "models": {},
    }
    predictions = {
        "y_val": y_val.to_numpy(), "y_test": y_test.to_numpy(),
        "val_proba": {}, "test_proba": {},
    }

    for name, info in metadata["models"].items():
        model_path = MODELS_DIR / info["file"]
        if not model_path.exists():
            print(f"ERROR: {model_path} not found. Run:  python -m src.train_models")
            raise SystemExit(1)
        pipeline = joblib.load(model_path)

        val_proba = pipeline.predict_proba(X_val)[:, 1]
        test_proba = pipeline.predict_proba(X_test)[:, 1]
        predictions["val_proba"][name] = val_proba
        predictions["test_proba"][name] = test_proba

        # 1) Choose threshold using VALIDATION data only
        val_curve = threshold_curve(y_val, val_proba, fn_cost, fp_cost)
        best_threshold = select_best_threshold(val_curve)

        # 2) Apply that frozen threshold to the TEST data (no tuning on test)
        val_metrics = evaluate_at_threshold(y_val, val_proba, best_threshold, fn_cost, fp_cost)
        test_metrics = evaluate_at_threshold(y_test, test_proba, best_threshold, fn_cost, fp_cost)
        # For comparison only: what the default 0.5 threshold would have cost
        test_at_half = evaluate_at_threshold(y_test, test_proba, 0.5, fn_cost, fp_cost)
        val_at_half = evaluate_at_threshold(y_val, val_proba, 0.5, fn_cost, fp_cost)

        # 3) Feature importance on VALIDATION data (works on the raw columns via the pipeline)
        perm = permutation_importance(
            pipeline, X_val, y_val, scoring="roc_auc", n_repeats=5,
            random_state=RANDOM_STATE, n_jobs=1,
        )
        importance = {
            col: {"mean": float(m), "std": float(s)}
            for col, m, s in zip(FEATURE_COLUMNS, perm.importances_mean, perm.importances_std)
        }

        results["models"][name] = {
            "threshold": best_threshold,
            "validation": val_metrics,
            "validation_at_0.5": val_at_half,
            "test": test_metrics,
            "test_at_0.5": test_at_half,
            "val_cost_curve": [float(c) for c in val_curve["total_cost"]],
            "permutation_importance": importance,
        }

        # Safety check: the stored cost must equal FN*cost + FP*cost
        expected = test_metrics["fn"] * fn_cost + test_metrics["fp"] * fp_cost
        assert abs(test_metrics["total_cost"] - expected) < 1e-9, "Cost calculation mismatch!"

        print(f"{name}: validation-selected threshold = {best_threshold:.2f} "
              f"(validation cost ${val_metrics['total_cost']:,.0f} vs ${val_at_half['total_cost']:,.0f} at 0.50)")

    # --- Rank models ---
    names = list(results["models"])
    by_cost = sorted(names, key=lambda n: results["models"][n]["test"]["total_cost"])
    results["best_by"] = {
        "cost": by_cost[0],
        "f1": max(names, key=lambda n: results["models"][n]["test"]["f1"]),
        "roc_auc": max(names, key=lambda n: results["models"][n]["test"]["roc_auc"]),
    }
    results["ranking_by_test_cost"] = by_cost

    # --- Save for the dashboard ---
    with open(RESULTS_PATH, "w") as f:
        json.dump(results, f, indent=2)
    joblib.dump(predictions, PREDICTIONS_PATH)

    # --- Print a summary table (TEST set, validation-selected thresholds) ---
    rows = []
    for n in by_cost:
        t = results["models"][n]["test"]
        rows.append({
            "Model": n, "Thresh": t["threshold"],
            "Acc": t["accuracy"], "Prec": t["precision"], "Rec": t["recall"],
            "F1": t["f1"], "AUC": t["roc_auc"],
            "FN": t["fn"], "FP": t["fp"],
            "Test cost ($)": t["total_cost"], "Cost/cust ($)": t["cost_per_customer"],
        })
    table = pd.DataFrame(rows)
    print("\n" + "=" * 100)
    print("TEST SET RESULTS (threshold chosen on validation set), ranked by estimated cost")
    print("=" * 100)
    print(table.to_string(index=False, float_format=lambda v: f"{v:,.3f}"))

    print("\nSame models at the default 0.50 threshold (for comparison only):")
    rows_half = []
    for n in by_cost:
        t = results["models"][n]["test_at_0.5"]
        rows_half.append({"Model": n, "Recall": t["recall"], "F1": t["f1"],
                          "FN": t["fn"], "FP": t["fp"], "Test cost ($)": t["total_cost"]})
    print(pd.DataFrame(rows_half).to_string(index=False, float_format=lambda v: f"{v:,.3f}"))

    print("\nBest model by lowest estimated test cost :", results["best_by"]["cost"])
    print("Best model by F1-score (test)            :", results["best_by"]["f1"])
    print("Best model by ROC-AUC (test)             :", results["best_by"]["roc_auc"])
    print(f"\nSaved: {RESULTS_PATH.name} and {PREDICTIONS_PATH.name} in the models folder.")
    print("Estimated costs are illustrative assumptions, not real profit or proven savings.")


if __name__ == "__main__":
    main()