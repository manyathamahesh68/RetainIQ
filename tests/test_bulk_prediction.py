import io
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from src.bulk_prediction import (
    BulkCsvError,
    REPORT_CSV_STATE_KEY,
    REPORT_FILENAME_STATE_KEY,
    REPORT_STATE_KEY,
    bulk_report_filename,
    clear_bulk_report_state,
    format_bulk_results,
    get_stored_bulk_report,
    parse_customer_csv,
    score_customer_csv,
    serialize_bulk_report,
    store_bulk_report,
)
from src.prediction_service import (
    MODEL_NAMES,
    PredictionInputError,
    load_prediction_assets,
    predict_customers,
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
        self.assertEqual(with_id["Original row number"].tolist(), [1, 2])
        self.assertEqual(
            with_id["Customer identifier"].tolist(),
            self.customer_rows[ID_COLUMN].iloc[:2].tolist(),
        )

        rows_without_id = self.customer_rows.iloc[:2].drop(columns=[ID_COLUMN])
        without_id = format_bulk_results(
            score_customer_csv(self.csv_bytes(rows_without_id), MODEL_NAMES[0], self.assets)
        )
        self.assertEqual(without_id["Customer identifier"].tolist(), [1, 2])
        self.assertEqual(without_id["Original row number"].tolist(), [1, 2])

    def test_report_columns_and_values_are_export_ready(self):
        batch = score_customer_csv(
            self.csv_bytes(self.customer_rows.iloc[:1]),
            MODEL_NAMES[0],
            self.assets,
        )

        report = format_bulk_results(batch)

        self.assertEqual(
            report.columns.tolist(),
            [
                "Customer identifier",
                "Original row number",
                "Selected model",
                "Saved threshold",
                "Model churn score",
                "Predicted class",
                "Risk rank",
            ],
        )
        self.assertEqual(
            report.loc[0, "Customer identifier"],
            self.customer_rows[ID_COLUMN].iloc[0],
        )
        self.assertEqual(report.loc[0, "Original row number"], 1)
        self.assertEqual(report.loc[0, "Selected model"], MODEL_NAMES[0])
        self.assertEqual(
            report.loc[0, "Saved threshold"],
            self.assets.thresholds[MODEL_NAMES[0]],
        )
        self.assertEqual(
            report.loc[0, "Model churn score"],
            batch.results["churn_probability"].iloc[0],
        )
        self.assertEqual(
            report.loc[0, "Predicted class"],
            batch.results["predicted_class"].iloc[0],
        )
        self.assertEqual(report.loc[0, "Risk rank"], 1)

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
        self.assertEqual(ordered_results["Original row number"].tolist(), [1, 2, 3])
        expected_ranks = (
            batch.results["churn_probability"]
            .rank(method="first", ascending=False)
            .astype(int)
            .tolist()
        )
        self.assertEqual(ordered_results["Risk rank"].tolist(), expected_ranks)

    def test_formula_text_is_escaped_without_changing_numeric_values(self):
        report = pd.DataFrame(
            {
                "Customer identifier": [
                    "=FORMULA()",
                    "   +FORMULA()",
                    "\t-FORMULA()",
                    "\x01@FORMULA()",
                    "\ufeff=FORMULA()",
                    "customer-5",
                ],
                "Original row number": [1, 2, 3, 4, 5, 6],
                "Selected model": ["Logistic Regression"] * 6,
                "Saved threshold": [0.5] * 6,
                "Model churn score": [-12.5, 0.2, 0.3, 0.4, 0.5, 0.6],
                "Predicted class": ["Churn"] * 6,
                "Risk rank": [1, 2, 3, 4, 5, 6],
            }
        )

        csv_bytes = serialize_bulk_report(report)
        exported = pd.read_csv(
            io.BytesIO(csv_bytes),
            encoding="utf-8-sig",
            dtype={"Customer identifier": "string"},
        )

        self.assertEqual(
            exported["Customer identifier"].tolist(),
            [
                "'=FORMULA()",
                "'   +FORMULA()",
                "'\t-FORMULA()",
                "'\x01@FORMULA()",
                "'\ufeff=FORMULA()",
                "customer-5",
            ],
        )
        self.assertEqual(exported.loc[0, "Model churn score"], -12.5)
        self.assertTrue(pd.api.types.is_numeric_dtype(exported["Model churn score"]))
        self.assertEqual(report.loc[0, "Customer identifier"], "=FORMULA()")

    def test_csv_serialization_and_model_specific_filename(self):
        report = format_bulk_results(
            score_customer_csv(
                self.csv_bytes(self.customer_rows.iloc[:1]),
                MODEL_NAMES[0],
                self.assets,
            )
        )

        csv_bytes = serialize_bulk_report(report)

        self.assertTrue(csv_bytes.startswith(b"\xef\xbb\xbf"))
        self.assertEqual(
            pd.read_csv(io.BytesIO(csv_bytes), encoding="utf-8-sig").columns.tolist(),
            report.columns.tolist(),
        )
        self.assertEqual(
            bulk_report_filename(MODEL_NAMES[0]),
            "retainiq_bulk_predictions_logistic_regression.csv",
        )
        with self.assertRaisesRegex(ValueError, "Unsupported prediction model"):
            bulk_report_filename("user-supplied-name")

    def test_stored_report_reuses_prebuilt_bytes_without_rescoring(self):
        contents = self.csv_bytes(self.customer_rows.iloc[:2])
        session_state = {}
        with patch(
            "src.bulk_prediction.predict_customers",
            wraps=predict_customers,
        ) as prediction_call:
            batch = score_customer_csv(contents, MODEL_NAMES[0], self.assets)
            store_bulk_report(session_state, batch, MODEL_NAMES[0])
            stored_report = get_stored_bulk_report(session_state)
            self.assertIsNotNone(stored_report)
            _, csv_bytes, filename = stored_report
            second_read = get_stored_bulk_report(session_state)

        self.assertEqual(prediction_call.call_count, 1)
        self.assertIs(second_read[1], csv_bytes)
        self.assertEqual(
            filename,
            "retainiq_bulk_predictions_logistic_regression.csv",
        )
        self.assertIn(REPORT_STATE_KEY, session_state)
        self.assertIn(REPORT_CSV_STATE_KEY, session_state)
        self.assertIn(REPORT_FILENAME_STATE_KEY, session_state)

    def test_clearing_report_state_removes_stale_report_data(self):
        session_state = {
            REPORT_STATE_KEY: pd.DataFrame({"old": [1]}),
            REPORT_CSV_STATE_KEY: b"old report",
            REPORT_FILENAME_STATE_KEY: "old.csv",
        }

        clear_bulk_report_state(session_state)

        self.assertIsNone(get_stored_bulk_report(session_state))
        self.assertEqual(session_state, {})


if __name__ == "__main__":
    unittest.main()
