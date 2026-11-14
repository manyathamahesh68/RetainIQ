import unittest

import numpy as np
import pandas as pd

from src.prediction_service import (
    MODEL_NAMES,
    PredictionInputError,
    load_prediction_assets,
    predict_customers,
)
from src.preprocessing import DATA_PATH, FEATURE_COLUMNS, ID_COLUMN, TARGET_COLUMN


class PredictionServiceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.assets = load_prediction_assets()
        source_rows = pd.read_csv(DATA_PATH).iloc[:2].copy()
        cls.customer_rows = source_rows.drop(columns=[TARGET_COLUMN])

    def test_scores_all_saved_models_with_their_frozen_thresholds(self):
        for model_name in MODEL_NAMES:
            with self.subTest(model=model_name):
                batch = predict_customers(self.customer_rows, model_name, self.assets)
                self.assertEqual(len(batch.results), len(self.customer_rows))
                self.assertEqual(
                    batch.results[ID_COLUMN].tolist(),
                    self.customer_rows[ID_COLUMN].tolist(),
                )
                self.assertEqual(
                    batch.results["threshold"].tolist(),
                    [self.assets.thresholds[model_name]] * len(self.customer_rows),
                )
                expected_classes = np.where(
                    batch.results["churn_probability"] >= self.assets.thresholds[model_name],
                    "Churn",
                    "Retained",
                )
                self.assertEqual(batch.results["predicted_class"].tolist(), expected_classes.tolist())
                self.assertTrue(batch.results["churn_probability"].between(0, 1).all())

    def test_feature_order_does_not_change_scoring_and_original_index_is_preserved(self):
        reordered = self.customer_rows.loc[
            :, [ID_COLUMN, *reversed(FEATURE_COLUMNS)]
        ]
        batch = predict_customers(reordered, MODEL_NAMES[0], self.assets)
        self.assertEqual(batch.results.index.tolist(), reordered.index.tolist())
        self.assertEqual(
            batch.results[ID_COLUMN].tolist(),
            reordered[ID_COLUMN].tolist(),
        )

    def test_missing_cells_are_reported_and_left_to_fitted_pipeline_imputers(self):
        rows_with_missing = self.customer_rows.copy()
        rows_with_missing.iloc[0, rows_with_missing.columns.get_loc("TotalCharges")] = np.nan
        rows_with_missing.iloc[1, rows_with_missing.columns.get_loc("OnlineSecurity")] = None

        batch = predict_customers(rows_with_missing, MODEL_NAMES[0], self.assets)

        self.assertEqual(
            batch.missing_value_counts,
            {"TotalCharges": 1, "OnlineSecurity": 1},
        )
        self.assertTrue(np.isfinite(batch.results["churn_probability"]).all())

    def test_rejects_missing_feature_columns(self):
        incomplete = self.customer_rows.drop(columns=[FEATURE_COLUMNS[0]])
        with self.assertRaisesRegex(PredictionInputError, "Missing required feature columns"):
            predict_customers(incomplete, MODEL_NAMES[0], self.assets)

    def test_rejects_target_column_to_prevent_label_leakage(self):
        labeled_rows = pd.read_csv(DATA_PATH).iloc[:2].copy()
        with self.assertRaisesRegex(PredictionInputError, "target column"):
            predict_customers(labeled_rows, MODEL_NAMES[0], self.assets)

    def test_rejects_invalid_numeric_values(self):
        invalid_rows = self.customer_rows.copy()
        invalid_rows.iloc[0, invalid_rows.columns.get_loc("TotalCharges")] = "not-a-number"
        with self.assertRaisesRegex(PredictionInputError, "non-numeric values"):
            predict_customers(invalid_rows, MODEL_NAMES[0], self.assets)

    def test_rejects_non_text_categorical_values(self):
        invalid_rows = self.customer_rows.copy()
        invalid_rows["Contract"] = invalid_rows["Contract"].astype(object)
        invalid_rows.iloc[0, invalid_rows.columns.get_loc("Contract")] = 12
        with self.assertRaisesRegex(PredictionInputError, "must contain text values"):
            predict_customers(invalid_rows, MODEL_NAMES[0], self.assets)

    def test_rejects_blank_customer_identifiers(self):
        invalid_rows = self.customer_rows.copy()
        invalid_rows.iloc[0, invalid_rows.columns.get_loc(ID_COLUMN)] = " "
        with self.assertRaisesRegex(PredictionInputError, "must not contain missing or blank"):
            predict_customers(invalid_rows, MODEL_NAMES[0], self.assets)

    def test_scoring_without_customer_id_preserves_input_order(self):
        features_only = self.customer_rows.drop(columns=[ID_COLUMN])
        batch = predict_customers(features_only, MODEL_NAMES[0], self.assets)
        self.assertNotIn(ID_COLUMN, batch.results.columns)
        self.assertEqual(batch.results.index.tolist(), features_only.index.tolist())

    def test_rejects_duplicate_column_names(self):
        duplicated = pd.concat(
            [self.customer_rows, self.customer_rows[[FEATURE_COLUMNS[0]]]],
            axis=1,
        )
        with self.assertRaisesRegex(PredictionInputError, "duplicate column names"):
            predict_customers(duplicated, MODEL_NAMES[0], self.assets)


if __name__ == "__main__":
    unittest.main()
