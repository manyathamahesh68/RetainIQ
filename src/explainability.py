"""On-demand SHAP feature attribution for saved customer churn models."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from src.prediction_service import PredictionAssets, predict_customers
from src.preprocessing import (
    DATA_PATH,
    FEATURE_COLUMNS,
    clean_data,
    load_data,
    split_data,
)

BACKGROUND_SIZE = 20
BACKGROUND_RANDOM_STATE = 42
MAX_EVALS = 39
RECONSTRUCTION_TOLERANCE = 1e-3
SCORE_TOLERANCE = 1e-10


class ExplainabilityError(RuntimeError):
    """Raised when an input or SHAP result cannot be explained safely."""


@dataclass(frozen=True)
class CustomerExplanation:
    model_name: str
    churn_score: float
    base_value: float
    reconstruction_error: float
    contributions: pd.DataFrame


class RawFeatureMasker:
    """Replace masked raw features with values from a mixed-type background."""

    def __init__(self, background: pd.DataFrame) -> None:
        self.data = background.to_numpy(dtype=object, copy=True)
        self.shape = self.data.shape
        self.immutable_outputs = False

    def __call__(self, mask: np.ndarray, row: np.ndarray) -> tuple[np.ndarray]:
        mask_array = np.asarray(mask, dtype=bool)
        row_array = np.asarray(row, dtype=object)
        if row_array.ndim == 2 and row_array.shape[0] == 1:
            row_array = row_array[0]
        if mask_array.shape != (self.shape[1],) or row_array.shape != (self.shape[1],):
            raise ExplainabilityError("The SHAP masker received an unexpected row or mask shape.")
        return (np.where(mask_array[None, :], row_array[None, :], self.data),)


def build_training_background(data_path: Path = DATA_PATH) -> pd.DataFrame:
    """Rebuild the deterministic training split and sample only its raw features."""
    features, target = clean_data(load_data(data_path))
    X_train, _, _, _, _, _ = split_data(features, target)
    if len(X_train) < BACKGROUND_SIZE:
        raise ExplainabilityError(
            f"The training split must contain at least {BACKGROUND_SIZE} rows."
        )
    background = X_train.sample(
        n=BACKGROUND_SIZE,
        random_state=BACKGROUND_RANDOM_STATE,
    ).copy()
    if list(background.columns) != FEATURE_COLUMNS:
        raise ExplainabilityError("The training background does not match the raw feature schema.")
    return background


def predict_churn_scores(
    customer_rows: pd.DataFrame | np.ndarray,
    model_name: str,
    assets: PredictionAssets,
) -> np.ndarray:
    """Return one-dimensional continuous churn scores through the shared service."""
    if not isinstance(customer_rows, pd.DataFrame):
        values = np.asarray(customer_rows, dtype=object)
        if values.ndim == 1:
            values = values.reshape(1, -1)
        if values.ndim != 2 or values.shape[1] != len(FEATURE_COLUMNS):
            raise ExplainabilityError("SHAP supplied an invalid raw-feature array.")
        customer_rows = pd.DataFrame(values, columns=FEATURE_COLUMNS)
    if list(customer_rows.columns) != FEATURE_COLUMNS:
        raise ExplainabilityError(
            "Explanation input must contain exactly the 19 raw feature columns in schema order."
        )

    batch = predict_customers(customer_rows, model_name, assets)
    churn_scores = batch.results["churn_probability"].to_numpy(dtype=float)
    if churn_scores.ndim != 1 or len(churn_scores) != len(customer_rows):
        raise ExplainabilityError("The shared prediction service returned an invalid score shape.")
    if (
        not np.isfinite(churn_scores).all()
        or (churn_scores < 0.0).any()
        or (churn_scores > 1.0).any()
    ):
        raise ExplainabilityError("The shared prediction service returned invalid churn scores.")
    return churn_scores


def explain_customer(
    customer_rows: pd.DataFrame,
    model_name: str,
    assets: PredictionAssets,
    background: pd.DataFrame,
    expected_churn_score: float | None = None,
) -> CustomerExplanation:
    """Explain one customer's continuous churn score using permutation SHAP."""
    if not isinstance(customer_rows, pd.DataFrame) or len(customer_rows) != 1:
        raise ExplainabilityError("Explainability requires exactly one customer row.")
    if not isinstance(background, pd.DataFrame):
        raise ExplainabilityError("The SHAP background must be a pandas DataFrame.")
    if background.shape != (BACKGROUND_SIZE, len(FEATURE_COLUMNS)):
        raise ExplainabilityError(
            f"The SHAP background must contain exactly {BACKGROUND_SIZE} rows "
            f"and {len(FEATURE_COLUMNS)} features."
        )
    if list(background.columns) != FEATURE_COLUMNS:
        raise ExplainabilityError("The SHAP background does not match the raw feature schema.")
    if list(customer_rows.columns) != FEATURE_COLUMNS:
        raise ExplainabilityError(
            "Explanation input must contain exactly the 19 raw feature columns in schema order."
        )

    try:
        import shap
    except ImportError as error:
        raise ExplainabilityError(
            "SHAP is unavailable. Install the project's pinned SHAP dependency."
        ) from error

    customer_copy = customer_rows.copy(deep=True)
    score = float(predict_churn_scores(customer_copy, model_name, assets)[0])
    if expected_churn_score is not None and not np.isclose(
        score,
        expected_churn_score,
        rtol=0.0,
        atol=SCORE_TOLERANCE,
    ):
        raise ExplainabilityError(
            "The explanation score does not match the previously displayed model score."
        )

    masker = RawFeatureMasker(background)
    try:
        explainer = shap.PermutationExplainer(
            lambda values: predict_churn_scores(values, model_name, assets),
            masker,
            feature_names=list(FEATURE_COLUMNS),
            seed=BACKGROUND_RANDOM_STATE,
        )
        explanation = explainer(
            customer_copy,
            max_evals=MAX_EVALS,
            silent=True,
        )
    except (AttributeError, IndexError, RuntimeError, TypeError, ValueError) as error:
        raise ExplainabilityError(
            f"SHAP explanation failed for {model_name}: {error}"
        ) from error

    values = np.asarray(explanation.values, dtype=float)
    if values.shape != (1, len(FEATURE_COLUMNS)):
        raise ExplainabilityError(f"SHAP returned an invalid attribution shape: {values.shape}.")
    values = values[0]
    if not np.isfinite(values).all():
        raise ExplainabilityError("SHAP returned non-finite feature contributions.")

    base_values = np.asarray(explanation.base_values, dtype=float).reshape(-1)
    if base_values.size != 1 or not np.isfinite(base_values).all():
        raise ExplainabilityError("SHAP returned an invalid base value.")
    base_value = float(base_values[0])

    explanation_feature_names = explanation.feature_names
    if (
        explanation_feature_names is None
        or list(explanation_feature_names) != FEATURE_COLUMNS
    ):
        raise ExplainabilityError("SHAP did not preserve the 19 raw feature names.")

    reconstruction_error = abs(base_value + float(values.sum()) - score)
    if reconstruction_error > RECONSTRUCTION_TOLERANCE:
        raise ExplainabilityError(
            "SHAP base value and contributions do not reconstruct the model churn score "
            f"within {RECONSTRUCTION_TOLERANCE:g}."
        )

    contributions = pd.DataFrame(
        {
            "Feature": FEATURE_COLUMNS,
            "Customer value": customer_copy.iloc[0].to_numpy(copy=True),
            "SHAP contribution": values,
        }
    )
    return CustomerExplanation(
        model_name=model_name,
        churn_score=score,
        base_value=base_value,
        reconstruction_error=reconstruction_error,
        contributions=contributions,
    )
