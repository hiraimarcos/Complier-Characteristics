"""Tests for built-in nuisance estimation helpers."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from complier_characteristics import ComplierDataset
from complier_characteristics.nuisance import (
    add_intercept,
    estimate_outcome_responses,
    estimate_propensity_scores,
    estimate_treatment_responses,
    expit,
    fit_linear_probability_model,
    fit_linear_regression,
    fit_logistic_regression,
    fit_probit_regression,
)


class LinkAndRegressionTests(unittest.TestCase):
    def test_expit_is_finite_for_large_inputs(self) -> None:
        probabilities = expit(np.array([-1000.0, 0.0, 1000.0]))

        self.assertTrue(np.all(probabilities > 0.0))
        self.assertTrue(np.all(probabilities < 1.0))
        self.assertAlmostEqual(probabilities[1], 0.5)

    def test_add_intercept_handles_populated_and_empty_feature_matrices(self) -> None:
        np.testing.assert_allclose(
            add_intercept(np.array([[2.0, 3.0], [4.0, 5.0]])),
            np.array([[1.0, 2.0, 3.0], [1.0, 4.0, 5.0]]),
        )
        np.testing.assert_allclose(add_intercept(np.empty((2, 0))), np.ones((2, 1)))

    def test_logistic_regression_handles_constant_targets_and_requires_matrix(self) -> None:
        predictions = fit_logistic_regression(
            np.array([[0.0], [1.0], [2.0]]),
            np.ones(3),
            clip=0.2,
        )

        np.testing.assert_allclose(predictions, np.full(3, 0.8))

        with self.assertRaisesRegex(ValueError, "two-dimensional"):
            fit_logistic_regression(np.array([0.0, 1.0, 2.0]), np.array([0.0, 1.0, 1.0]))

    def test_probit_regression_returns_clipped_varying_probabilities(self) -> None:
        predictions = fit_probit_regression(
            np.array([[-2.0], [-1.0], [-0.5], [0.0], [0.5], [1.0], [2.0], [3.0]]),
            np.array([0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 1.0]),
            clip=0.01,
        )

        self.assertEqual(predictions.shape, (8,))
        self.assertTrue(np.all(predictions >= 0.01))
        self.assertTrue(np.all(predictions <= 0.99))
        self.assertGreater(predictions[-1], predictions[0])

    def test_linear_regression_returns_predictions_for_requested_features(self) -> None:
        predictions = fit_linear_regression(
            np.array([[0.0], [1.0], [2.0]]),
            np.array([1.0, 3.0, 5.0]),
            prediction_features=np.array([[3.0], [4.0]]),
        )

        np.testing.assert_allclose(predictions, np.array([7.0, 9.0]))

        with self.assertRaisesRegex(ValueError, "two-dimensional"):
            fit_linear_regression(np.array([0.0, 1.0, 2.0]), np.array([1.0, 3.0, 5.0]))

    def test_linear_probability_model_clips_ols_predictions(self) -> None:
        predictions = fit_linear_probability_model(
            np.array([[-10.0], [0.0], [10.0]]),
            np.array([0.0, 1.0, 1.0]),
            prediction_features=np.array([[-100.0], [0.0], [100.0]]),
            clip=0.1,
        )

        self.assertEqual(predictions.shape, (3,))
        self.assertTrue(np.all(predictions >= 0.1))
        self.assertTrue(np.all(predictions <= 0.9))
        self.assertAlmostEqual(predictions[0], 0.1)
        self.assertAlmostEqual(predictions[-1], 0.9)


class NuisanceEstimatorTests(unittest.TestCase):
    def test_constant_models_return_clipped_sample_and_stratum_means(self) -> None:
        dataset = ComplierDataset.from_arrays(
            instrument=[0, 0, 1, 1],
            treatment=[0, 1, 1, 1],
            covariates={"x": [-1.0, 0.0, 1.0, 2.0]},
        )

        propensities = estimate_propensity_scores(dataset, model="constant", clip=0.05)
        treated_if_z0, treated_if_z1 = estimate_treatment_responses(dataset, model="constant", clip=0.05)

        np.testing.assert_allclose(propensities, np.full(4, 0.5))
        np.testing.assert_allclose(treated_if_z0, np.full(4, 0.5))
        np.testing.assert_allclose(treated_if_z1, np.full(4, 0.95))

    def test_logit_propensity_with_empty_covariate_subset_falls_back_to_constant(self) -> None:
        dataset = ComplierDataset.from_arrays(
            instrument=[0, 1, 1, 1],
            treatment=[0, 0, 1, 1],
            covariates={"x": [-100.0, 1.0, 2.0, 3.0]},
        )

        propensities = estimate_propensity_scores(
            dataset,
            model="logit",
            covariate_names=[],
            clip=0.01,
        )

        np.testing.assert_allclose(propensities, np.full(4, 0.75))

    def test_probit_models_estimate_propensity_and_treatment_responses(self) -> None:
        dataset = ComplierDataset.from_arrays(
            instrument=[0, 1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0],
            treatment=[0, 0, 1, 1, 0, 1, 0, 1, 1, 0, 0, 1],
            covariates={
                "x": [
                    -3.0,
                    -2.0,
                    -1.0,
                    0.0,
                    1.0,
                    2.0,
                    3.0,
                    -2.5,
                    -1.5,
                    -0.5,
                    0.5,
                    1.5,
                ]
            },
        )

        propensities = estimate_propensity_scores(
            dataset,
            model="probit",
            covariate_names=["x"],
            clip=0.01,
        )
        treated_if_z0, treated_if_z1 = estimate_treatment_responses(
            dataset,
            model="probit",
            covariate_names=["x"],
            clip=0.01,
        )

        for predictions in (propensities, treated_if_z0, treated_if_z1):
            self.assertEqual(predictions.shape, (dataset.n_obs,))
            self.assertTrue(np.all(predictions >= 0.01))
            self.assertTrue(np.all(predictions <= 0.99))

    def test_linear_probability_models_estimate_propensity_and_treatment_responses(self) -> None:
        dataset = ComplierDataset.from_arrays(
            instrument=[0, 0, 0, 1, 1, 1],
            treatment=[0, 0, 1, 0, 1, 1],
            covariates={"x": [-2.0, 0.0, 2.0, -2.0, 0.0, 2.0]},
        )

        propensities = estimate_propensity_scores(
            dataset,
            model="linear",
            covariate_names=["x"],
            clip=0.05,
        )
        treated_if_z0, treated_if_z1 = estimate_treatment_responses(
            dataset,
            model="linear",
            covariate_names=["x"],
            clip=0.05,
        )

        for predictions in (propensities, treated_if_z0, treated_if_z1):
            self.assertEqual(predictions.shape, (dataset.n_obs,))
            self.assertTrue(np.all(predictions >= 0.05))
            self.assertTrue(np.all(predictions <= 0.95))

        self.assertGreater(treated_if_z0[-1], treated_if_z0[0])
        self.assertGreater(treated_if_z1[-1], treated_if_z1[0])

    def test_nuisance_estimators_reject_unknown_models_and_missing_strata(self) -> None:
        dataset = ComplierDataset.from_arrays(
            instrument=[0, 1],
            treatment=[0, 1],
        )

        with self.assertRaisesRegex(ValueError, "Unsupported propensity model"):
            estimate_propensity_scores(dataset, model="forest")

        with self.assertRaisesRegex(ValueError, "Unsupported treatment model"):
            estimate_treatment_responses(dataset, model="forest")

        one_stratum = ComplierDataset.from_arrays(
            instrument=[1, 1],
            treatment=[0, 1],
        )
        with self.assertRaisesRegex(ValueError, "Each instrument stratum"):
            estimate_treatment_responses(one_stratum, model="constant")

    def test_constant_and_linear_outcome_models_estimate_assignment_responses(self) -> None:
        dataset = ComplierDataset.from_arrays(
            instrument=[0, 0, 0, 1, 1, 1],
            treatment=[0, 1, 0, 1, 0, 1],
            outcome=[1.0, 3.0, 5.0, 9.0, 11.0, 13.0],
            covariates={"x": [0.0, 1.0, 2.0, 0.0, 1.0, 2.0]},
        )

        outcome_if_z0, outcome_if_z1 = estimate_outcome_responses(dataset, model="constant")
        np.testing.assert_allclose(outcome_if_z0, np.full(dataset.n_obs, 3.0))
        np.testing.assert_allclose(outcome_if_z1, np.full(dataset.n_obs, 11.0))

        outcome_if_z0, outcome_if_z1 = estimate_outcome_responses(
            dataset,
            model="linear",
            covariate_names=["x"],
        )
        np.testing.assert_allclose(outcome_if_z0, np.array([1.0, 3.0, 5.0, 1.0, 3.0, 5.0]))
        np.testing.assert_allclose(outcome_if_z1, np.array([9.0, 11.0, 13.0, 9.0, 11.0, 13.0]))

    def test_outcome_models_reject_unknown_models_and_missing_strata(self) -> None:
        dataset = ComplierDataset.from_arrays(
            instrument=[0, 1],
            treatment=[0, 1],
            outcome=[1.0, 2.0],
        )

        with self.assertRaisesRegex(ValueError, "Unsupported outcome model"):
            estimate_outcome_responses(dataset, model="forest")

        one_stratum = ComplierDataset.from_arrays(
            instrument=[1, 1],
            treatment=[0, 1],
            outcome=[1.0, 2.0],
        )
        with self.assertRaisesRegex(ValueError, "Each instrument stratum"):
            estimate_outcome_responses(one_stratum, model="constant")


if __name__ == "__main__":
    unittest.main()
