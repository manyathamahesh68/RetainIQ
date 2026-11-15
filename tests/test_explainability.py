import unittest

import numpy as np
import pandas as pd

from src.explainability import (
    BACKGROUND_RANDOM_STATE,
    BACKGROUND_SIZE,
    RECONSTRUCTION_TOLERANCE,
    CustomerExplanation,
    ExplainabilityError,
    build_training_background,
    explain_customer,
    predict_churn_scores,
)
from src.prediction_service import (
    MODEL_NAMES,
    load_prediction_assets,
    predict_customers,
)
from src.preprocessing import (
    FEATURE_COLUMNS,
    clean_data,
    load_data,
    split_data,
)


class ExplainabilityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        source_features, source_target = clean_data(load_data())
        cls.X_train, _, _, _, _, _ = split_data(source_features, source_target)
        cls.background = cls.X_train.sample(
            n=BACKGROUND_SIZE,
            random_state=BACKGROUND_RANDOM_STATE,
        ).copy()
        cls.customer_rows = (
            cls.X_train.drop(index=cls.background.index).iloc[[0]].copy(deep=True)
        )
        cls.assets = load_prediction_assets()

    def test_background_is_the_deterministic_training_only_sample(self):
        expected = self.X_train.sample(
            n=BACKGROUND_SIZE,
            random_state=BACKGROUND_RANDOM_STATE,
        )

        background = build_training_background()

        pd.testing.assert_frame_equal(background, expected)
        self.assertEqual(list(background.columns), FEATURE_COLUMNS)
        self.assertEqual(len(background), BACKGROUND_SIZE)
        self.assertTrue(set(background.index).issubset(set(self.X_train.index)))

    def test_score_wrapper_uses_class_one_service_score_for_all_models(self):
        for model_name in MODEL_NAMES:
            with self.subTest(model=model_name):
                result = predict_churn_scores(
                    self.customer_rows,
                    model_name,
                    self.assets,
                )
                service_score = predict_customers(
                    self.customer_rows,
                    model_name,
                    self.assets,
                ).results["churn_probability"].to_numpy()
                model = self.assets.models[model_name]
                class_index = list(model.classes_).index(1)
                pipeline_score = model.predict_proba(self.customer_rows)[0, class_index]

                self.assertEqual(result.shape, (1,))
                self.assertTrue(np.isfinite(result).all())
                np.testing.assert_allclose(result, service_score, rtol=0.0, atol=1e-10)
                np.testing.assert_allclose(result[0], pipeline_score, rtol=0.0, atol=1e-10)

    def test_score_wrapper_accepts_shap_masked_raw_arrays(self):
        values = self.customer_rows.to_numpy(dtype=object)

        scores = predict_churn_scores(values, MODEL_NAMES[0], self.assets)

        self.assertEqual(scores.shape, (1,))
        self.assertTrue(np.isfinite(scores).all())

    def test_explains_all_saved_models_and_preserves_input(self):
        customer_before = self.customer_rows.copy(deep=True)

        for model_name in MODEL_NAMES:
            with self.subTest(model=model_name):
                batch = predict_customers(
                    self.customer_rows,
                    model_name,
                    self.assets,
                )
                expected_score = float(batch.results["churn_probability"].iloc[0])
                explanation = explain_customer(
                    self.customer_rows,
                    model_name,
                    self.assets,
                    self.background,
                    expected_churn_score=expected_score,
                )

                self.assertIsInstance(explanation, CustomerExplanation)
                self.assertEqual(explanation.model_name, model_name)
                self.assertAlmostEqual(explanation.churn_score, expected_score, places=10)
                self.assertEqual(
                    explanation.contributions["Feature"].tolist(),
                    FEATURE_COLUMNS,
                )
                self.assertTrue(
                    np.isfinite(
                        explanation.contributions["SHAP contribution"].to_numpy(dtype=float)
                    ).all()
                )
                self.assertLessEqual(
                    explanation.reconstruction_error,
                    RECONSTRUCTION_TOLERANCE,
                )
                self.assertAlmostEqual(
                    explanation.base_value
                    + explanation.contributions["SHAP contribution"].sum(),
                    expected_score,
                    delta=RECONSTRUCTION_TOLERANCE,
                )
                pd.testing.assert_frame_equal(self.customer_rows, customer_before)

    def test_missing_and_unseen_values_are_explained_for_all_models(self):
        edge_cases = {
            "missing numeric": ("TotalCharges", np.nan),
            "missing categorical": ("OnlineSecurity", np.nan),
            "unseen categorical": (
                "InternetService",
                "__RETAINIQ_UNSEEN_XAI_CATEGORY__",
            ),
        }

        for case_name, (feature, value) in edge_cases.items():
            customer = self.customer_rows.copy(deep=True)
            customer.loc[customer.index[0], feature] = value
            customer_before = customer.copy(deep=True)
            for model_name in MODEL_NAMES:
                with self.subTest(case=case_name, model=model_name):
                    service_score = float(
                        predict_customers(
                            customer,
                            model_name,
                            self.assets,
                        ).results["churn_probability"].iloc[0]
                    )
                    explanation = explain_customer(
                        customer,
                        model_name,
                        self.assets,
                        self.background,
                        expected_churn_score=service_score,
                    )

                    self.assertTrue(np.isfinite(explanation.churn_score))
                    self.assertTrue(
                        np.isfinite(
                            explanation.contributions["SHAP contribution"].to_numpy(
                                dtype=float
                            )
                        ).all()
                    )
                    self.assertLessEqual(
                        explanation.reconstruction_error,
                        RECONSTRUCTION_TOLERANCE,
                    )
                    self.assertEqual(
                        explanation.contributions["Feature"].tolist(),
                        FEATURE_COLUMNS,
                    )
                    pd.testing.assert_frame_equal(customer, customer_before)

    def test_rejects_non_training_shaped_background_and_score_mismatch(self):
        with self.assertRaisesRegex(ExplainabilityError, "exactly 20 rows"):
            explain_customer(
                self.customer_rows,
                MODEL_NAMES[0],
                self.assets,
                self.background.iloc[:-1],
            )

        with self.assertRaisesRegex(ExplainabilityError, "does not match"):
            explain_customer(
                self.customer_rows,
                MODEL_NAMES[0],
                self.assets,
                self.background,
                expected_churn_score=1.5,
            )


if __name__ == "__main__":
    unittest.main()
