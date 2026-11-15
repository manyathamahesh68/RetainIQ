"""Transparent, rule-based customer retention review prompts."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass

import pandas as pd

REVIEW_PROMPT_DISCLAIMER = (
    "This is a review prompt, not a proven churn-prevention intervention."
)
NO_REVIEW_PROMPT_MESSAGE = "No specific review prompt was triggered."

_KNOWN_INTERNET_SERVICES = frozenset({"DSL", "Fiber optic"})


@dataclass(frozen=True)
class RetentionRecommendation:
    title: str
    trigger_explanation: str
    suggested_action: str
    disclaimer: str = REVIEW_PROMPT_DISCLAIMER


@dataclass(frozen=True)
class RetentionRecommendationResult:
    model_churn_score: float
    predicted_class: str
    recommendations: tuple[RetentionRecommendation, ...]
    neutral_message: str | None


def _customer_values(
    customer_row: pd.DataFrame | pd.Series | Mapping[str, object],
) -> dict[str, object]:
    if isinstance(customer_row, pd.DataFrame):
        if len(customer_row) != 1:
            raise ValueError("Retention recommendations require exactly one customer row.")
        return customer_row.iloc[0].to_dict()
    if isinstance(customer_row, pd.Series):
        return customer_row.to_dict()
    if isinstance(customer_row, Mapping):
        return dict(customer_row)
    raise TypeError("Customer input must be a mapping, pandas Series, or one-row DataFrame.")


def _category(row: Mapping[str, object], field: str, accepted: frozenset[str]) -> str | None:
    value = row.get(field)
    if not isinstance(value, str):
        return None
    normalized = value.strip()
    return normalized if normalized in accepted else None


def _tenure_months(row: Mapping[str, object]) -> float | None:
    value = row.get("tenure")
    if value is None or isinstance(value, bool):
        return None
    try:
        months = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return months if math.isfinite(months) else None


def generate_retention_recommendations(
    customer_row: pd.DataFrame | pd.Series | Mapping[str, object],
    predicted_class: str,
    model_churn_score: float,
) -> RetentionRecommendationResult:
    """Return review prompts from explicit customer-field rules.

    The supplied model score is preserved as context only; it is not rescored
    or used to create another decision threshold.
    """
    if not isinstance(predicted_class, str):
        raise TypeError("Predicted class must be a string.")
    if isinstance(model_churn_score, bool):
        raise ValueError("Model Churn Score must be a finite value between 0 and 1.")
    try:
        score = float(model_churn_score)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError("Model Churn Score must be a finite value between 0 and 1.") from error
    if not math.isfinite(score) or not 0.0 <= score <= 1.0:
        raise ValueError("Model Churn Score must be a finite value between 0 and 1.")

    row = _customer_values(customer_row)
    recommendations: list[RetentionRecommendation] = []
    tenure = _tenure_months(row)
    contract = _category(
        row,
        "Contract",
        frozenset({"Month-to-month", "One year", "Two year"}),
    )
    internet_service = _category(
        row,
        "InternetService",
        frozenset({"No", *_KNOWN_INTERNET_SERVICES}),
    )

    if predicted_class == "Churn" and tenure is not None and 0 <= tenure <= 12:
        recommendations.append(
            RetentionRecommendation(
                title="Review onboarding and service fit",
                trigger_explanation=(
                    f"Predicted class = Churn; tenure = {tenure:g} months "
                    "(within the 0–12 month rule)."
                ),
                suggested_action=(
                    "Review the customer's onboarding experience and confirm that "
                    "their selected services fit their needs."
                ),
            )
        )

    if contract == "Month-to-month":
        recommendations.append(
            RetentionRecommendation(
                title="Discuss available contract options",
                trigger_explanation="Contract = Month-to-month.",
                suggested_action=(
                    "Explain available contract options and their terms; confirm "
                    "whether any option fits the customer's preferences."
                ),
            )
        )

    if (
        internet_service in _KNOWN_INTERNET_SERVICES
        and _category(row, "TechSupport", frozenset({"Yes", "No", "No internet service"}))
        == "No"
    ):
        recommendations.append(
            RetentionRecommendation(
                title="Review technical support needs",
                trigger_explanation=(
                    f"InternetService = {internet_service}; TechSupport = No."
                ),
                suggested_action=(
                    "Ask whether the customer needs technical assistance and offer "
                    "an appropriate support review."
                ),
            )
        )

    if (
        internet_service in _KNOWN_INTERNET_SERVICES
        and _category(
            row,
            "OnlineSecurity",
            frozenset({"Yes", "No", "No internet service"}),
        )
        == "No"
    ):
        recommendations.append(
            RetentionRecommendation(
                title="Review security needs",
                trigger_explanation=(
                    f"InternetService = {internet_service}; OnlineSecurity = No."
                ),
                suggested_action=(
                    "Ask whether the customer's security needs are being met and "
                    "review relevant service options if requested."
                ),
            )
        )

    return RetentionRecommendationResult(
        model_churn_score=score,
        predicted_class=predicted_class,
        recommendations=tuple(recommendations),
        neutral_message=NO_REVIEW_PROMPT_MESSAGE if not recommendations else None,
    )
