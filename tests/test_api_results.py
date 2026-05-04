"""Tests for score construction, fitted results, and high-level API behavior."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from complier_characteristics import ComplierDataset, ComplierEstimator
from complier_characteristics.estimators import fit_abadie_backend, fit_doubly_robust_backend


def make_fixed_dataset() -> ComplierDataset:
    return ComplierDataset.from_arrays(
        instrument=[0, 0, 1, 1],
        treatment=[0, 0, 0, 1],
        covariates={
            "x": [10.0, 20.0, 30.0, 40.0],
            "high": [0.0, 1.0, 0.0, 1.0],
        },
    )


class BackendScoreTests(unittest.TestCase):
    def test_abadie_backend_constructs_expected_scores_and_diagnostics(self) -> None:
        dataset = make_fixed_dataset()
        backend_result = fit_abadie_backend(
            dataset,
            propensities=np.full(dataset.n_obs, 0.5),
            normalize=True,
        )

        expected_raw_scores = np.array([1.0, 1.0, -1.0, 1.0])
        np.testing.assert_allclose(backend_result.raw_scores, expected_raw_scores)
        np.testing.assert_allclose(backend_result.scaled_scores, expected_raw_scores / 0.5)
        self.assertAlmostEqual(backend_result.complier_share, 0.5)

        diagnostics = backend_result.diagnostics.to_dict()
        self.assertEqual(diagnostics["n_obs"], 4)
        self.assertEqual(diagnostics["backend"], "abadie")
        self.assertTrue(diagnostics["normalized"])
        self.assertAlmostEqual(diagnostics["instrument_rate"], 0.5)
        self.assertAlmostEqual(diagnostics["treatment_rate"], 0.25)
        self.assertAlmostEqual(diagnostics["first_stage"], 0.5)
        self.assertAlmostEqual(diagnostics["negative_score_fraction"], 0.25)
        self.assertAlmostEqual(diagnostics["score_mean"], 1.0)

    def test_doubly_robust_backend_constructs_expected_scores(self) -> None:
        dataset = make_fixed_dataset()
        treated_if_z0 = np.zeros(dataset.n_obs)
        treated_if_z1 = np.full(dataset.n_obs, 0.5)
        backend_result = fit_doubly_robust_backend(
            dataset,
            propensities=np.full(dataset.n_obs, 0.5),
            treated_if_z0=treated_if_z0,
            treated_if_z1=treated_if_z1,
            normalize=True,
        )

        expected_raw_scores = np.array([0.5, 0.5, -0.5, 1.5])
        np.testing.assert_allclose(backend_result.raw_scores, expected_raw_scores)
        np.testing.assert_allclose(backend_result.scaled_scores, expected_raw_scores / 0.5)
        np.testing.assert_allclose(backend_result.treated_if_z0, treated_if_z0)
        np.testing.assert_allclose(backend_result.treated_if_z1, treated_if_z1)
        self.assertAlmostEqual(backend_result.complier_share, 0.5)

    def test_zero_complier_share_raises(self) -> None:
        dataset = ComplierDataset.from_arrays(
            instrument=[0, 0, 1, 1],
            treatment=[0, 1, 0, 1],
        )

        with self.assertRaisesRegex(ValueError, "complier share is numerically zero"):
            fit_abadie_backend(
                dataset,
                propensities=np.full(dataset.n_obs, 0.5),
                normalize=True,
            )


class ResultFunctionalTests(unittest.TestCase):
    def test_result_functionals_match_hand_calculated_complier_moments(self) -> None:
        dataset = make_fixed_dataset()
        result = ComplierEstimator(backend="abadie").fit(
            dataset,
            propensity_scores=np.full(dataset.n_obs, 0.5),
        )

        self.assertAlmostEqual(result.mean("x").estimate, 20.0)
        self.assertAlmostEqual(result.mean(lambda data: data.covariates["x"] + 1.0).estimate, 21.0)
        self.assertAlmostEqual(result.share("high").estimate, 1.0)
        self.assertAlmostEqual(result.variance("x").estimate, 200.0)

        cdf = result.cdf("x", grid=[15.0, 25.0, 35.0, 45.0])
        np.testing.assert_allclose(cdf.grid, np.array([15.0, 25.0, 35.0, 45.0]))
        np.testing.assert_allclose(cdf.values, np.array([0.5, 1.0, 1.0, 1.0]))

    def test_moments_remain_complier_ratios_when_scores_are_not_normalized(self) -> None:
        dataset = make_fixed_dataset()
        result = ComplierEstimator(backend="abadie", normalize=False).fit(
            dataset,
            propensity_scores=np.full(dataset.n_obs, 0.5),
        )

        self.assertAlmostEqual(result.mean("x").numerator, 10.0)
        self.assertAlmostEqual(result.mean("x").denominator, 0.5)
        self.assertAlmostEqual(result.mean("x").estimate, 20.0)
        self.assertAlmostEqual(result.diagnostics.score_mean, 0.5)
        np.testing.assert_allclose(result.scaled_scores, result.raw_scores)

    def test_result_helpers_validate_inputs_and_summarize_covariates(self) -> None:
        dataset = make_fixed_dataset()
        result = ComplierEstimator(backend="abadie").fit(
            dataset,
            propensity_scores=np.full(dataset.n_obs, 0.5),
        )

        with self.assertRaisesRegex(ValueError, "share\\(\\) requires"):
            result.share("x")

        with self.assertRaisesRegex(ValueError, "grid must be one-dimensional"):
            result.cdf("x", grid=[[1.0, 2.0]])

        summary = result.summarize_covariates()
        self.assertEqual(set(summary), {"x", "high"})
        self.assertEqual(set(summary["x"]), {"mean", "variance"})
        self.assertEqual(set(summary["high"]), {"mean", "variance", "share"})


class EstimatorApiTests(unittest.TestCase):
    def test_fit_rejects_unsupported_backend_and_partial_treatment_nuisances(self) -> None:
        dataset = make_fixed_dataset()

        with self.assertRaisesRegex(ValueError, "Unsupported backend"):
            ComplierEstimator(backend="unknown").fit(dataset)

        with self.assertRaisesRegex(ValueError, "must be supplied together"):
            ComplierEstimator(backend="dr").fit(
                dataset,
                propensity_scores=np.full(dataset.n_obs, 0.5),
                treated_if_z0=np.zeros(dataset.n_obs),
            )

    def test_fit_clips_supplied_external_nuisances(self) -> None:
        dataset = ComplierDataset.from_arrays(
            instrument=[0, 1, 0, 1],
            treatment=[0, 1, 0, 1],
        )

        result = ComplierEstimator(backend="dr", clip=0.1).fit(
            dataset,
            propensity_scores=[0.0, 1.0, 0.2, 0.8],
            treated_if_z0=[0.0, 0.0, 0.0, 0.0],
            treated_if_z1=[1.0, 1.0, 1.0, 1.0],
        )

        np.testing.assert_allclose(result.propensities, np.array([0.1, 0.9, 0.2, 0.8]))
        np.testing.assert_allclose(result.treated_if_z0, np.full(dataset.n_obs, 0.1))
        np.testing.assert_allclose(result.treated_if_z1, np.full(dataset.n_obs, 0.9))
        self.assertEqual(result.backend, "dr")


if __name__ == "__main__":
    unittest.main()
