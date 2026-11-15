"""CSV upload parsing and presentation helpers for bulk customer scoring."""

from __future__ import annotations

import csv
import io
import unicodedata
from collections import Counter
from collections.abc import MutableMapping

import pandas as pd

from src.prediction_service import (
    ID_COLUMN,
    MODEL_NAMES,
    PredictionAssets,
    PredictionBatch,
    predict_customers,
)

REPORT_STATE_KEY = "bulk_prediction_report"
REPORT_CSV_STATE_KEY = "bulk_prediction_csv"
REPORT_FILENAME_STATE_KEY = "bulk_prediction_filename"
REPORT_STATE_KEYS = (
    REPORT_STATE_KEY,
    REPORT_CSV_STATE_KEY,
    REPORT_FILENAME_STATE_KEY,
)

_MODEL_FILENAME_SLUGS = {
    "Logistic Regression": "logistic_regression",
    "Random Forest": "random_forest",
    "Gradient Boosting": "gradient_boosting",
    "Neural Network": "neural_network",
}


class BulkCsvError(ValueError):
    """Raised when an uploaded file is not a well-formed UTF-8 CSV."""


def parse_customer_csv(contents: bytes) -> pd.DataFrame:
    """Parse UTF-8 CSV bytes and assign a one-based source data-row index."""
    if not isinstance(contents, bytes) or not contents:
        raise BulkCsvError("The uploaded file is empty.")
    try:
        text = contents.decode("utf-8-sig")
    except UnicodeDecodeError as error:
        raise BulkCsvError("The uploaded file must be UTF-8 encoded (UTF-8 BOM is supported).") from error
    if not text.strip():
        raise BulkCsvError("The uploaded file is empty.")
    if "\x00" in text:
        raise BulkCsvError("The uploaded file is not a supported UTF-8 text CSV.")

    try:
        reader = csv.reader(io.StringIO(text, newline=""), strict=True)
        header = next(reader, None)
        if not header:
            raise BulkCsvError("The CSV must include a header row.")
        if any(not column.strip() for column in header):
            raise BulkCsvError("The CSV header contains a blank column name.")
        duplicates = [
            column for column, count in Counter(header).items() if count > 1
        ]
        if duplicates:
            raise BulkCsvError(f"The CSV contains duplicate column names: {duplicates}")

        data_row_count = 0
        for row_number, row in enumerate(reader, start=2):
            if not row:
                continue
            if len(row) != len(header):
                raise BulkCsvError(
                    f"CSV data record {row_number} has {len(row)} fields; "
                    f"the header has {len(header)}."
                )
            data_row_count += 1
    except csv.Error as error:
        raise BulkCsvError(f"The CSV is malformed: {error}") from error

    if data_row_count == 0:
        raise BulkCsvError("The CSV contains a header but no customer rows.")

    dtype = {ID_COLUMN: "string"} if ID_COLUMN in header else None
    try:
        customer_rows = pd.read_csv(
            io.StringIO(text),
            dtype=dtype,
            keep_default_na=False,
            on_bad_lines="error",
        )
    except pd.errors.EmptyDataError as error:
        raise BulkCsvError("The uploaded file is empty or has no readable header.") from error
    except pd.errors.ParserError as error:
        raise BulkCsvError(f"The CSV is malformed: {error}") from error

    if customer_rows.empty:
        raise BulkCsvError("The CSV contains a header but no customer rows.")
    customer_rows.index = pd.RangeIndex(
        start=1,
        stop=len(customer_rows) + 1,
        name="source_row_number",
    )
    return customer_rows


def score_customer_csv(
    contents: bytes,
    model_name: str,
    assets: PredictionAssets,
) -> PredictionBatch:
    """Parse and validate a CSV, then score it through the shared service."""
    customer_rows = parse_customer_csv(contents)
    return predict_customers(customer_rows, model_name, assets)


def format_bulk_results(batch: PredictionBatch) -> pd.DataFrame:
    """Add customer identifiers and stable score ranks without reordering rows."""
    results = batch.results.copy()
    row_numbers = results.index.to_numpy(copy=True)
    if ID_COLUMN in results.columns:
        identifiers = results.pop(ID_COLUMN)
    else:
        identifiers = pd.Series(row_numbers, index=results.index)
    results.insert(0, "Customer identifier", identifiers.to_numpy())
    results.insert(1, "Original row number", row_numbers)
    results["Risk rank"] = (
        results["churn_probability"].rank(method="first", ascending=False).astype(int)
    )
    return (
        results.rename(
            columns={
                "model": "Selected model",
                "threshold": "Saved threshold",
                "churn_probability": "Model churn score",
                "predicted_class": "Predicted class",
            }
        )
        .loc[
            :,
            [
                "Customer identifier",
                "Original row number",
                "Selected model",
                "Saved threshold",
                "Model churn score",
                "Predicted class",
                "Risk rank",
            ],
        ]
        .reset_index(drop=True)
    )


def _is_formula_text(value: str) -> bool:
    for character in value:
        if character.isspace() or unicodedata.category(character) in {"Cc", "Cf"}:
            continue
        return character in "=+-@"
    return False


def serialize_bulk_report(report: pd.DataFrame) -> bytes:
    """Serialize a formula-safe copy without changing the displayed report."""
    export_report = report.copy(deep=True)
    for column in export_report.columns:
        export_report[column] = export_report[column].map(
            lambda value: f"'{value}"
            if isinstance(value, str) and _is_formula_text(value)
            else value
        )
    return export_report.to_csv(index=False).encode("utf-8-sig")


def bulk_report_filename(model_name: str) -> str:
    """Return a fixed filename for a supported saved model."""
    if model_name not in MODEL_NAMES:
        raise ValueError(f"Unsupported prediction model: {model_name}")
    return f"retainiq_bulk_predictions_{_MODEL_FILENAME_SLUGS[model_name]}.csv"


def clear_bulk_report_state(session_state: MutableMapping[str, object]) -> None:
    """Remove the previous report, its CSV bytes, and its filename."""
    for key in REPORT_STATE_KEYS:
        session_state.pop(key, None)


def store_bulk_report(
    session_state: MutableMapping[str, object],
    batch: PredictionBatch,
    model_name: str,
) -> None:
    """Store one scored report and its pre-generated CSV in session memory."""
    report = format_bulk_results(batch)
    csv_bytes = serialize_bulk_report(report)
    filename = bulk_report_filename(model_name)

    session_state[REPORT_STATE_KEY] = report
    session_state[REPORT_CSV_STATE_KEY] = csv_bytes
    session_state[REPORT_FILENAME_STATE_KEY] = filename


def get_stored_bulk_report(
    session_state: MutableMapping[str, object],
) -> tuple[pd.DataFrame, bytes, str] | None:
    """Read the complete report tuple from session memory, if available."""
    report = session_state.get(REPORT_STATE_KEY)
    csv_bytes = session_state.get(REPORT_CSV_STATE_KEY)
    filename = session_state.get(REPORT_FILENAME_STATE_KEY)
    if (
        isinstance(report, pd.DataFrame)
        and isinstance(csv_bytes, bytes)
        and isinstance(filename, str)
    ):
        return report, csv_bytes, filename
    return None
