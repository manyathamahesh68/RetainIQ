import unittest

import numpy as np
import pandas as pd

from src.bulk_prediction import (
    BulkCsvError,
    format_bulk_results,
    parse_customer_csv,
    score_customer_csv,
)
from src.prediction_service import (
    MODEL_NAMES,
    PredictionInputError,
    load_prediction_assets,
)
from src.preprocessing import DATA_PATH, FEATURE_COLUMNS, ID_COLUMN, TARGET_COLUMN


class BulkPredictionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.assets = load_prediction_assets()
        source_rows = pd.read_csv(DATA_PATH).iloc[:3].copy()
        cls.customer_rows = source_rows.drop(columns=[TARGET_COLUMN])

    def csv_bytes(self, rows, encoding="utf-8"):
        return rows.to_csv(index=False).encode(encoding)

    def test_parses_valid_utf8_bom_and_preserves_ids_and_row_order(self):
        contents = self.csv_bytes(self.customer_rows.iloc[:2], encoding="utf-8-sig")

        parsed = parse_customer_csv(contents)

        self.assertEqual(parsed.index.tolist(), [1, 2])
        self.assertEqual(
            parsed[ID_COLUMN].tolist(),
            self.customer_rows[ID_COLUMN].iloc[:2].tolist(),
        )

    def test_rejects_empty_unsupported_and_malformed_files(self):
        with self.assertRaisesRegex(BulkCsvError, "empty"):
            parse_customer_csv(b"")
        with self.assertRaisesRegex(BulkCsvError, "UTF-8"):
            parse_customer_csv(b"\xff\xfe\x00\x00")
        with self.assertRaisesRegex(BulkCsvError, "malformed"):
            parse_customer_csv(b'customerID,tenure\n"unfinished,5\n')

    def test_rejects_ragged_rows_and_duplicate_headers(self):
        with self.assertRaisesRegex(BulkCsvError, "fields; the header has"):
            parse_customer_csv(b"customerID,tenure\nc-1\n")

        rows = self.customer_rows.iloc[:1]
        contents = self.csv_bytes(rows)
        csv_lines = contents.decode("utf-8").splitlines()
        header = csv_lines[0].split(",")
        header[1] = header[0]
        duplicate_header = ",".join(header) + "\n" + "\n".join(csv_lines[1:])
        with self.assertRaisesRegex(BulkCsvError, "duplicate column names"):
            parse_customer_csv(duplicate_header.encode("utf-8"))

    def test_rejects_invalid_feature_schema(self):
        missing_feature = self.customer_rows.drop(columns=[FEATURE_COLUMNS[0]])
        with self.assertRaisesRegex(PredictionInputError, "Missing required feature columns"):
            score_customer_csv(self.csv_bytes(missing_feature), MODEL_NAMES[0], self.assets)

        labeled_rows = pd.read_csv(DATA_PATH).iloc[:1]
        with self.assertRaisesRegex(PredictionInputError, "target column"):
            score_customer_csv(self.csv_bytes(labeled_rows), MODEL_NAMES[0], self.assets)

    def test_rejects_blank_customer_ids(self):
        rows_with_blank_id = self.customer_rows.iloc[:1].copy()
        rows_with_blank_id.loc[rows_with_blank_id.index[0], ID_COLUMN] = ""
        with self.assertRaisesRegex(PredictionInputError, "must not contain missing or blank"):
            score_customer_csv(
                self.csv_bytes(rows_with_blank_id),
                MODEL_NAMES[0],
                self.assets,
            )

    def test_reports_missing_feature_values_and_uses_fitted_imputers(self):
        rows_with_missing = self.customer_rows.iloc[:2].copy()
        rows_with_missing.iloc[0, rows_with_missing.columns.get_loc("TotalCharges")] = np.nan
        rows_with_missing.iloc[1, rows_with_missing.columns.get_loc("OnlineSecurity")] = None

        batch = score_customer_csv(self.csv_bytes(rows_with_missing), MODEL_NAMES[0], self.assets)

        self.assertEqual(
            batch.missing_value_counts,
            {"TotalCharges": 1, "OnlineSecurity": 1},
        )
        self.assertTrue(np.isfinite(batch.results["churn_probability"]).all())

    def test_preserves_customer_ids_and_adds_source_row_when_id_is_absent(self):
        contents_with_id = self.csv_bytes(self.customer_rows.iloc[:2])
        with_id = format_bulk_results(
            score_customer_csv(contents_with_id, MODEL_NAMES[0], self.assets)
        )
        self.assertEqual(
            with_id["Customer identifier"].tolist(),
            self.customer_rows[ID_COLUMN].iloc[:2].tolist(),
        )

        rows_without_id = self.customer_rows.iloc[:2].drop(columns=[ID_COLUMN])
        without_id = format_bulk_results(
            score_customer_csv(self.csv_bytes(rows_without_id), MODEL_NAMES[0], self.assets)
        )
        self.assertEqual(without_id["Customer identifier"].tolist(), [1, 2])

    def test_selected_model_and_saved_threshold_are_used(self):
        contents = self.csv_bytes(self.customer_rows.iloc[:2])
        for model_name in MODEL_NAMES:
            with self.subTest(model=model_name):
                results = format_bulk_results(
                    score_customer_csv(contents, model_name, self.assets)
                )
                self.assertEqual(results["Selected model"].tolist(), [model_name] * 2)
                self.assertEqual(
                    results["Saved threshold"].tolist(),
                    [self.assets.thresholds[model_name]] * 2,
                )
                expected_classes = np.where(
                    results["Model churn score"] >= self.assets.thresholds[model_name],
                    "Churn",
                    "Retained",
                )
                self.assertEqual(results["Predicted class"].tolist(), expected_classes.tolist())

    def test_ranks_scores_stably_without_reordering_source_rows(self):
        identical_rows = pd.concat(
            [
                self.customer_rows.iloc[[0]].drop(columns=[ID_COLUMN]),
                self.customer_rows.iloc[[0]].drop(columns=[ID_COLUMN]),
            ],
            ignore_index=True,
        )
        results = format_bulk_results(
            score_customer_csv(self.csv_bytes(identical_rows), MODEL_NAMES[0], self.assets)
        )

        self.assertEqual(results["Customer identifier"].tolist(), [1, 2])
        self.assertEqual(results["Risk rank"].tolist(), [1, 2])

        batch = score_customer_csv(
            self.csv_bytes(self.customer_rows),
            MODEL_NAMES[0],
            self.assets,
        )
        ordered_results = format_bulk_results(batch)
        self.assertEqual(
            ordered_results["Customer identifier"].tolist(),
            self.customer_rows[ID_COLUMN].tolist(),
        )
        expected_ranks = (
            batch.results["churn_probability"]
            .rank(method="first", ascending=False)
            .astype(int)
            .tolist()
        )
        self.assertEqual(ordered_results["Risk rank"].tolist(), expected_ranks)


if __name__ == "__main__":
    unittest.main()
