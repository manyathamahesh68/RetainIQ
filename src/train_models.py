"""
train_models.py
---------------
Trains four models (Logistic Regression, Random Forest, Gradient Boosting,
Neural Network). Each is a full Pipeline: preprocessing + classifier.

- Hyperparameters are tuned with cross-validation on the TRAINING set only.
- The validation set is used here only to print a sanity-check ROC-AUC.
- The test set is NOT touched in this file.
Run with:  python -m src.train_models
"""

import json
from pathlib import Path
import sys
import time
import warnings
from datetime import datetime

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import joblib
import sklearn
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline

from src.preprocessing import (
    FEATURE_COLUMNS, RANDOM_STATE,
    build_preprocessor, clean_data, load_data, split_data,
)

MODELS_DIR = PROJECT_ROOT / "models"

# name -> (file name, estimator, search space, number of random combinations to try)
MODEL_CONFIGS = {
    "Logistic Regression": (
        "logistic_regression.pkl",
        LogisticRegression(max_iter=2000, random_state=RANDOM_STATE),
        {"model__C": [0.01, 0.1, 1, 10]},
        4,
    ),
    "Random Forest": (
        "random_forest.pkl",
        RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1),
        {
            "model__n_estimators": [200, 300, 500],
            "model__max_depth": [None, 6, 10, 15],
            "model__min_samples_leaf": [1, 3, 5, 10],
            "model__max_features": ["sqrt", 0.5],
        },
        12,
    ),
    "Gradient Boosting": (
        "gradient_boosting.pkl",
        GradientBoostingClassifier(random_state=RANDOM_STATE),
        {
            "model__n_estimators": [100, 200, 300],
            "model__learning_rate": [0.03, 0.05, 0.1],
            "model__max_depth": [2, 3, 4],
            "model__subsample": [0.8, 1.0],
        },
        10,
    ),
    "Neural Network": (
        "neural_network.pkl",
        MLPClassifier(early_stopping=True, max_iter=500, random_state=RANDOM_STATE),
        {
            "model__hidden_layer_sizes": [(16,), (32,), (32, 16), (64, 32)],
            "model__alpha": [1e-4, 1e-3, 1e-2],
            "model__learning_rate_init": [1e-3, 3e-3],
        },
        8,
    ),
}


def main() -> None:
    # --- Load and split (same split every time because of the fixed random_state) ---
    try:
        raw_df = load_data()
    except (FileNotFoundError, ValueError) as err:
        print(f"ERROR: {err}")
        raise SystemExit(1)

    X, y = clean_data(raw_df)
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y)
    print(f"Training rows: {len(X_train)} | Validation rows: {len(X_val)} | Test rows: {len(X_test)} (untouched)\n")

    MODELS_DIR.mkdir(exist_ok=True)
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)
    metadata = {
        "trained_at": datetime.now().isoformat(timespec="seconds"),
        "sklearn_version": sklearn.__version__,
        "random_state": RANDOM_STATE,
        "feature_columns": FEATURE_COLUMNS,
        "split_sizes": {"train": len(X_train), "validation": len(X_val), "test": len(X_test)},
        "models": {},
    }

    for name, (file_name, estimator, search_space, n_iter) in MODEL_CONFIGS.items():
        print(f"--- Training: {name} ---")
        start = time.time()

        pipeline = Pipeline([("preprocessor", build_preprocessor()), ("model", estimator)])
        search = RandomizedSearchCV(
            pipeline,
            param_distributions=search_space,
            n_iter=n_iter,
            scoring="roc_auc",   # threshold-free metric, computed on training folds only
            cv=cv,
            random_state=RANDOM_STATE,
            refit=True,          # refit the best setting on the full training set
            n_jobs=1,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # hides harmless convergence warnings
            search.fit(X_train, y_train)

        best_pipeline = search.best_estimator_
        val_auc = roc_auc_score(y_val, best_pipeline.predict_proba(X_val)[:, 1])
        elapsed = time.time() - start

        joblib.dump(best_pipeline, MODELS_DIR / file_name)

        # Convert params to plain strings so they can be written to JSON
        best_params = {k.replace("model__", ""): str(v) for k, v in search.best_params_.items()}
        metadata["models"][name] = {
            "file": file_name,
            "best_params": best_params,
            "cv_roc_auc_train": round(float(search.best_score_), 4),
            "validation_roc_auc": round(float(val_auc), 4),
        }
        print(f"  Best params        : {best_params}")
        print(f"  CV ROC-AUC (train) : {search.best_score_:.4f}")
        print(f"  Validation ROC-AUC : {val_auc:.4f}")
        print(f"  Saved to           : models/{file_name}  ({elapsed:.0f} s)\n")

    with open(MODELS_DIR / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
    print("All four models trained and saved. Metadata written to models/metadata.json")
    print("Test set was not used. Next step: threshold optimisation on the validation set.")


if __name__ == "__main__":
    main()