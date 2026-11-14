"""CSV upload parsing and presentation helpers for bulk customer scoring."""

from __future__ import annotations

import csv
import io
from collections import Counter

import pandas as pd

from src.prediction_service import (
    ID_COLUMN,
    PredictionAssets,
    PredictionBatch,
    predict_customers,
)


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
    if ID_COLUMN in results.columns:
        identifiers = results.pop(ID_COLUMN)
    else:
        identifiers = pd.Series(results.index, index=results.index)
    results.insert(0, "Customer identifier", identifiers.to_numpy())
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
                "Risk rank",
                "Selected model",
                "Saved threshold",
                "Model churn score",
                "Predicted class",
            ],
        ]
        .reset_index(drop=True)
    )
