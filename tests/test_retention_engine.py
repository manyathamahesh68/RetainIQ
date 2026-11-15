import unittest

import pandas as pd

from src.retention_engine import (
    NO_REVIEW_PROMPT_MESSAGE,
    REVIEW_PROMPT_DISCLAIMER,
    RetentionRecommendationResult,
    generate_retention_recommendations,
)


class RetentionEngineTests(unittest.TestCase):
    def setUp(self):
        self.customer = {
            "tenure": 24,
            "Contract": "One year",
            "InternetService": "No",
            "TechSupport": "No internet service",
            "OnlineSecurity": "No internet service",
            "gender": "Female",
            "SeniorCitizen": 0,
            "Partner": "No",
            "Dependents": "No",
        }

    def recommend(self, customer=None, predicted_class="Retained", score=0.4):
        return generate_retention_recommendations(
            self.customer if customer is None else customer,
            predicted_class,
            score,
        )

    def test_onboarding_rule_requires_churn_class_and_tenure_from_zero_to_twelve(self):
        customer = {**self.customer, "tenure": 12}

        result = self.recommend(customer, predicted_class="Churn")

        self.assertEqual(len(result.recommendations), 1)
        recommendation = result.recommendations[0]
        self.assertEqual(recommendation.title, "Review onboarding and service fit")
        self.assertIn("Predicted class = Churn", recommendation.trigger_explanation)
        self.assertIn("tenure = 12 months", recommendation.trigger_explanation)

        at_zero_months = self.recommend(
            {**self.customer, "tenure": 0},
            predicted_class="Churn",
        )
        self.assertEqual(len(at_zero_months.recommendations), 1)

        retained = self.recommend(customer, predicted_class="Retained")
        self.assertEqual(retained.recommendations, ())

    def test_month_to_month_contract_rule(self):
        result = self.recommend({**self.customer, "Contract": "Month-to-month"})

        self.assertEqual(len(result.recommendations), 1)
        self.assertEqual(
            result.recommendations[0].title,
            "Discuss available contract options",
        )
        self.assertIn("Contract = Month-to-month", result.recommendations[0].trigger_explanation)

    def test_technical_support_rule_requires_a_known_internet_service(self):
        customer = {
            **self.customer,
            "InternetService": "DSL",
            "TechSupport": "No",
        }

        result = self.recommend(customer)

        self.assertEqual(len(result.recommendations), 1)
        self.assertEqual(
            result.recommendations[0].title,
            "Review technical support needs",
        )
        self.assertIn("InternetService = DSL", result.recommendations[0].trigger_explanation)
        self.assertIn("TechSupport = No", result.recommendations[0].trigger_explanation)

    def test_security_rule_requires_a_known_internet_service(self):
        customer = {
            **self.customer,
            "InternetService": "Fiber optic",
            "OnlineSecurity": "No",
        }

        result = self.recommend(customer)

        self.assertEqual(len(result.recommendations), 1)
        self.assertEqual(result.recommendations[0].title, "Review security needs")
        self.assertIn(
            "OnlineSecurity = No",
            result.recommendations[0].trigger_explanation,
        )

    def test_multiple_rules_return_all_matching_prompts_in_stable_order(self):
        customer = {
            **self.customer,
            "tenure": 6,
            "Contract": "Month-to-month",
            "InternetService": "DSL",
            "TechSupport": "No",
            "OnlineSecurity": "No",
        }

        result = self.recommend(customer, predicted_class="Churn")

        self.assertEqual(
            [recommendation.title for recommendation in result.recommendations],
            [
                "Review onboarding and service fit",
                "Discuss available contract options",
                "Review technical support needs",
                "Review security needs",
            ],
        )
        self.assertIsNone(result.neutral_message)

    def test_no_matching_rules_returns_neutral_message(self):
        result = self.recommend()

        self.assertIsInstance(result, RetentionRecommendationResult)
        self.assertEqual(result.recommendations, ())
        self.assertEqual(result.neutral_message, NO_REVIEW_PROMPT_MESSAGE)

    def test_missing_fields_and_unexpected_categories_are_ignored_safely(self):
        missing_fields = self.recommend(
            {
                "InternetService": None,
                "TechSupport": "No",
                "OnlineSecurity": "No",
            },
            predicted_class="Churn",
        )
        unexpected_values = self.recommend(
            {
                "tenure": "unknown",
                "Contract": "Flexible",
                "InternetService": "Satellite",
                "TechSupport": "Unavailable",
                "OnlineSecurity": "Unknown",
            },
            predicted_class="Churn",
        )

        self.assertEqual(missing_fields.recommendations, ())
        self.assertEqual(missing_fields.neutral_message, NO_REVIEW_PROMPT_MESSAGE)
        self.assertEqual(unexpected_values.recommendations, ())
        self.assertEqual(unexpected_values.neutral_message, NO_REVIEW_PROMPT_MESSAGE)

    def test_one_row_dataframe_is_supported_and_multirow_input_is_rejected(self):
        one_row = pd.DataFrame([self.customer])

        result = self.recommend(one_row)

        self.assertEqual(result.recommendations, ())
        with self.assertRaisesRegex(ValueError, "exactly one customer row"):
            self.recommend(pd.DataFrame([self.customer, self.customer]))

    def test_output_is_deterministic_and_score_does_not_change_rule_matching(self):
        customer = {
            **self.customer,
            "Contract": "Month-to-month",
            "InternetService": "Fiber optic",
            "TechSupport": "No",
        }

        first = self.recommend(customer, score=0.12)
        repeated = self.recommend(customer, score=0.12)
        different_score = self.recommend(customer, score=0.98)

        self.assertEqual(first, repeated)
        self.assertEqual(first.recommendations, different_score.recommendations)
        self.assertEqual(first.model_churn_score, 0.12)
        self.assertEqual(different_score.model_churn_score, 0.98)

    def test_demographics_and_historical_churn_do_not_influence_recommendations(self):
        recommendation_trigger = {
            **self.customer,
            "Contract": "Month-to-month",
            "InternetService": "DSL",
            "TechSupport": "No",
        }
        baseline = self.recommend(recommendation_trigger, predicted_class="Churn")
        changed_demographics = {
            **recommendation_trigger,
            "gender": "Male",
            "SeniorCitizen": 1,
            "Partner": "Yes",
            "Dependents": "Yes",
            "Churn": "Yes",
        }
        changed = self.recommend(changed_demographics, predicted_class="Churn")

        self.assertTrue(baseline.recommendations)
        self.assertEqual(baseline.recommendations, changed.recommendations)
        self.assertEqual(baseline.neutral_message, changed.neutral_message)

    def test_each_recommendation_is_a_review_prompt_not_a_causal_or_savings_claim(self):
        customer = {
            **self.customer,
            "tenure": 6,
            "Contract": "Month-to-month",
            "InternetService": "Fiber optic",
            "TechSupport": "No",
            "OnlineSecurity": "No",
        }
        result = self.recommend(customer, predicted_class="Churn")

        self.assertTrue(result.recommendations)
        for recommendation in result.recommendations:
            self.assertEqual(recommendation.disclaimer, REVIEW_PROMPT_DISCLAIMER)
            text = " ".join(
                [
                    recommendation.title,
                    recommendation.trigger_explanation,
                    recommendation.suggested_action,
                ]
            ).casefold()
            self.assertNotIn("causes churn", text)
            self.assertNotIn("will prevent churn", text)
            self.assertNotIn("guaranteed savings", text)
            self.assertNotIn("roi", text)

    def test_invalid_model_churn_score_is_rejected(self):
        for score in (-0.1, 1.1, float("nan"), float("inf")):
            with self.subTest(score=score):
                with self.assertRaisesRegex(ValueError, "Model Churn Score"):
                    self.recommend(score=score)


if __name__ == "__main__":
    unittest.main()
