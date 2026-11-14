"""Inference helpers for the saved RetainIQ model pipelines."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import joblib
import numpy as np
import pandas as pd

from src.preprocessing import (
    CATEGORICAL_FEATURES,
    FEATURE_COLUMNS,
    ID_COLUMN,
    NUMERIC_FEATURES,
    TARGET_COLUMN,
)


PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = PROJECT_ROOT / "models"
MODEL_NAMES = (
    "Logistic Regression",
    "Random Forest",
    "Gradient Boosting",
    "Neural Network",
)


class PredictionModel(Protocol):
    classes_: np.ndarray

    def predict_proba(self, features: pd.DataFrame) -> np.ndarray: ...


@dataclass(frozen=True)
class PredictionAssets:
    models: Mapping[str, PredictionModel]
    thresholds: Mapping[str, float]


@dataclass(frozen=True)
class PredictionBatch:
    results: pd.DataFrame
    missing_value_counts: Mapping[str, int]


class PredictionInputError(ValueError):
    """Raised when customer rows do not match the saved model input schema."""


def _read_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as file:
        value = json.load(file)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {path}.")
    return value


def load_prediction_assets(models_dir: Path = MODELS_DIR) -> PredictionAssets:
    """Load the four saved pipelines and their frozen evaluation thresholds."""
    models_dir = Path(models_dir)
    metadata = _read_json(models_dir / "metadata.json")
    evaluation = _read_json(models_dir / "evaluation_results.json")
    model_metadata = metadata.get("models")
    evaluated_models = evaluation.get("models")
    if not isinstance(model_metadata, dict):
        raise ValueError("Model metadata does not contain a valid 'models' object.")
    if not isinstance(evaluated_models, dict):
        raise ValueError("Evaluation results do not contain a valid 'models' object.")

    models: dict[str, PredictionModel] = {}
    thresholds: dict[str, float] = {}
    for name in MODEL_NAMES:
        info = model_metadata.get(name)
        if not isinstance(info, dict):
            raise ValueError(f"Model metadata is missing {name}.")
        filename = info.get("file")
        if not isinstance(filename, str) or not filename or Path(filename).name != filename:
            raise ValueError(f"Model metadata has an invalid artifact filename for {name}.")

        model_path = models_dir / filename
        if not model_path.is_file():
            raise FileNotFoundError(f"Saved model for {name} was not found: {model_path}")
        model = joblib.load(model_path)
        if not callable(getattr(model, "predict_proba", None)):
            raise ValueError(f"Saved model for {name} does not support probability predictions.")
        classes = getattr(model, "classes_", None)
        if classes is None or 1 not in classes:
            raise ValueError(f"Saved model for {name} does not contain the churn class (1).")

        saved_features = getattr(model, "feature_names_in_", None)
        if saved_features is not None and list(saved_features) != FEATURE_COLUMNS:
            raise ValueError(f"Saved model for {name} does not match the current feature schema.")

        result = evaluated_models.get(name)
        threshold = result.get("threshold") if isinstance(result, dict) else None
        if isinstance(threshold, bool) or not isinstance(threshold, (int, float)):
            raise ValueError(f"Saved evaluation threshold for {name} is missing or invalid.")
        threshold = float(threshold)
        if not np.isfinite(threshold) or not 0.0 <= threshold <= 1.0:
            raise ValueError(f"Saved evaluation threshold for {name} must be between 0 and 1.")

        models[name] = model
        thresholds[name] = threshold

    return PredictionAssets(models=models, thresholds=thresholds)


def _validate_customer_rows(customer_rows: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int]]:
    if not isinstance(customer_rows, pd.DataFrame):
        raise PredictionInputError("Customer input must be a pandas DataFrame.")
    if customer_rows.empty:
        raise PredictionInputError("Customer input must contain at least one row.")
    if customer_rows.columns.duplicated().any():
        raise PredictionInputError("Customer input contains duplicate column names.")

    missing_columns = [column for column in FEATURE_COLUMNS if column not in customer_rows.columns]
    if missing_columns:
        raise PredictionInputError(f"Missing required feature columns: {missing_columns}")

    allowed_columns = set(FEATURE_COLUMNS) | {ID_COLUMN}
    unexpected_columns = [column for column in customer_rows.columns if column not in allowed_columns]
    if unexpected_columns:
        if TARGET_COLUMN in unexpected_columns:
            raise PredictionInputError(
                f"The target column '{TARGET_COLUMN}' must not be included in prediction input."
            )
        raise PredictionInputError(f"Unexpected customer input columns: {unexpected_columns}")

    if ID_COLUMN in customer_rows.columns:
        customer_ids = customer_rows[ID_COLUMN]
        invalid_ids = customer_ids.isna() | customer_ids.map(
            lambda value: isinstance(value, str) and not value.strip()
        )
        if invalid_ids.any():
            raise PredictionInputError(
                f"'{ID_COLUMN}' must not contain missing or blank values."
            )

    features = customer_rows.loc[:, FEATURE_COLUMNS].copy()
    missing_value_counts: dict[str, int] = {}

    for column in NUMERIC_FEATURES:
        values = features[column]
        blank_values = values.map(lambda value: isinstance(value, str) and not value.strip())
        normalized = values.mask(blank_values, np.nan)
        converted = pd.to_numeric(normalized, errors="coerce")
        invalid_values = normalized.notna() & converted.isna()
        if invalid_values.any():
            raise PredictionInputError(
                f"Feature '{column}' contains non-numeric values at input rows "
                f"{list(customer_rows.index[invalid_values][:5])}."
            )
        non_finite = converted.notna() & ~np.isfinite(converted)
        if non_finite.any():
            raise PredictionInputError(
                f"Feature '{column}' contains non-finite numeric values at input rows "
                f"{list(customer_rows.index[non_finite][:5])}."
            )
        features[column] = converted
        missing_count = int(converted.isna().sum())
        if missing_count:
            missing_value_counts[column] = missing_count

    for column in CATEGORICAL_FEATURES:
        values = features[column]
        invalid_types = values.notna() & ~values.map(lambda value: isinstance(value, str))
        if invalid_types.any():
            raise PredictionInputError(
                f"Feature '{column}' must contain text values or be missing."
            )
        normalized = values.map(lambda value: value.strip() if isinstance(value, str) else value)
        blank_values = normalized.map(lambda value: isinstance(value, str) and not value)
        normalized = normalized.mask(blank_values, np.nan)
        features[column] = normalized
        missing_count = int(normalized.isna().sum())
        if missing_count:
            missing_value_counts[column] = missing_count

    return features, missing_value_counts


def predict_customers(
    customer_rows: pd.DataFrame,
    model_name: str,
    assets: PredictionAssets,
) -> PredictionBatch:
    """Score customer rows with one saved pipeline and its saved threshold."""
    if model_name not in assets.models or model_name not in assets.thresholds:
        raise ValueError(f"No saved model and threshold are available for {model_name!r}.")

    features, missing_value_counts = _validate_customer_rows(customer_rows)
    model = assets.models[model_name]
    classes = list(model.classes_)
    churn_class_index = classes.index(1)
    probabilities = np.asarray(model.predict_proba(features), dtype=float)
    if probabilities.ndim != 2 or probabilities.shape != (len(features), len(classes)):
        raise RuntimeError(f"Saved model for {model_name} returned an invalid probability shape.")
    churn_probabilities = probabilities[:, churn_class_index]
    if (
        not np.isfinite(churn_probabilities).all()
        or (churn_probabilities < 0.0).any()
        or (churn_probabilities > 1.0).any()
    ):
        raise RuntimeError(f"Saved model for {model_name} returned invalid churn scores.")

    threshold = assets.thresholds[model_name]
    results = pd.DataFrame(index=customer_rows.index.copy())
    if ID_COLUMN in customer_rows.columns:
        results[ID_COLUMN] = customer_rows[ID_COLUMN]
    results["model"] = model_name
    results["churn_probability"] = churn_probabilities
    results["threshold"] = threshold
    results["predicted_class"] = np.where(
        churn_probabilities >= threshold, "Churn", "Retained"
    )
    return PredictionBatch(results=results, missing_value_counts=missing_value_counts)
