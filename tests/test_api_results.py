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
from complier_characteristics.estimators import (
    fit_abadie_backend,
    fit_doubly_robust_backend,
    fit_plugin_backend,
)


def make_fixed_dataset() -> ComplierDataset:
    return ComplierDataset.from_arrays(
        instrument=[0, 0, 1, 1],
        treatment=[0, 0, 0, 1],
        covariates={
            "x": [10.0, 20.0, 30.0, 40.0],
            "high": [0.0, 1.0, 0.0, 1.0],
        },
    )


def make_assignment_dataset() -> ComplierDataset:
    return ComplierDataset.from_arrays(
        instrument=[0, 0, 1, 1],
        treatment=[0, 0, 0, 1],
        outcome=[2.0, 4.0, 8.0, 10.0],
    )


def make_potential_outcome_dataset() -> ComplierDataset:
    return ComplierDataset.from_arrays(
        instrument=[0, 1, 0, 1, 0, 1, 0, 1],
        treatment=[0, 0, 0, 1, 0, 1, 1, 1],
        outcome=[1.0, 1.0, 2.0, 5.0, 4.0, 9.0, 7.0, 7.0],
    )


def make_balanced_latent_type_dataset() -> tuple[
    ComplierDataset,
    dict[str, float | np.ndarray],
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    """Build a finite population where complier estimands are known exactly."""

    # d0, d1, x, high, y0, y1. Each latent unit appears once with Z=0 and once
    # with Z=1, so p(Z=1) is exactly 0.5 within every latent type.
    latent_rows = np.array(
        [
            [0.0, 1.0, 1.0, 0.0, 2.0, 6.0],
            [0.0, 1.0, 3.0, 1.0, 5.0, 9.0],
            [0.0, 1.0, 5.0, 1.0, 6.0, 14.0],
            [0.0, 0.0, 2.0, 0.0, 4.0, 11.0],
            [0.0, 0.0, 6.0, 1.0, 8.0, 15.0],
            [1.0, 1.0, 0.0, 0.0, 1.0, 7.0],
            [1.0, 1.0, 4.0, 1.0, 3.0, 10.0],
        ],
        dtype=float,
    )
    d0, d1, x, high, y0, y1 = latent_rows.T

    instrument: list[float] = []
    treatment: list[float] = []
    observed_x: list[float] = []
    observed_high: list[float] = []
    outcome: list[float] = []
    treated_if_z0: list[float] = []
    treated_if_z1: list[float] = []

    for row_d0, row_d1, row_x, row_high, row_y0, row_y1 in latent_rows:
        for row_z in (0.0, 1.0):
            row_d = row_d1 if row_z == 1.0 else row_d0
            instrument.append(row_z)
            treatment.append(row_d)
            observed_x.append(row_x)
            observed_high.append(row_high)
            outcome.append(row_y1 if row_d == 1.0 else row_y0)
            treated_if_z0.append(row_d0)
            treated_if_z1.append(row_d1)

    dataset = ComplierDataset.from_arrays(
        instrument=instrument,
        treatment=treatment,
        outcome=outcome,
        covariates={
            "x": observed_x,
            "high": observed_high,
        },
    )

    complier = d1 > d0
    assigned_outcome = np.where(d1 == 1.0, y1, y0)
    unassigned_outcome = np.where(d0 == 1.0, y1, y0)
    expected: dict[str, float | np.ndarray] = {
        "complier_share": float(complier.mean()),
        "mean_x": float(x[complier].mean()),
        "share_high": float(high[complier].mean()),
        "variance_x": float(np.mean((x[complier] - x[complier].mean()) ** 2)),
        "cdf_x": np.array([1.0 / 3.0, 2.0 / 3.0, 1.0]),
        "untreated_mean": float(y0[complier].mean()),
        "treated_mean": float(y1[complier].mean()),
        "late": float((y1[complier] - y0[complier]).mean()),
        "assigned_mean": float(assigned_outcome.mean()),
        "unassigned_mean": float(unassigned_outcome.mean()),
        "assignment_ate": float((assigned_outcome - unassigned_outcome).mean()),
    }

    return (
        dataset,
        expected,
        np.full(dataset.n_obs, 0.5),
        np.asarray(treated_if_z0, dtype=float),
        np.asarray(treated_if_z1, dtype=float),
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

    def test_plugin_backend_constructs_conditional_first_stage_scores(self) -> None:
        dataset = make_fixed_dataset()
        treated_if_z0 = np.array([0.1, 0.1, 0.3, 0.5])
        treated_if_z1 = np.array([0.1, 0.3, 0.7, 0.9])
        backend_result = fit_plugin_backend(
            dataset,
            treated_if_z0=treated_if_z0,
            treated_if_z1=treated_if_z1,
            normalize=True,
        )

        expected_raw_scores = np.array([0.0, 0.2, 0.4, 0.4])
        np.testing.assert_allclose(backend_result.raw_scores, expected_raw_scores)
        np.testing.assert_allclose(backend_result.scaled_scores, expected_raw_scores / 0.25)
        np.testing.assert_allclose(backend_result.treated_if_z0, treated_if_z0)
        np.testing.assert_allclose(backend_result.treated_if_z1, treated_if_z1)
        self.assertIsNone(backend_result.propensities)
        self.assertAlmostEqual(backend_result.complier_share, 0.25)

        diagnostics = backend_result.diagnostics.to_dict()
        self.assertEqual(diagnostics["backend"], "plugin")
        self.assertIsNone(diagnostics["min_propensity"])
        self.assertIsNone(diagnostics["max_propensity"])
        self.assertAlmostEqual(diagnostics["negative_score_fraction"], 0.0)

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
        self.assertAlmostEqual(result.mean("x").standard_error, np.sqrt(150.0))
        self.assertAlmostEqual(result.mean(lambda data: data.covariates["x"] + 1.0).estimate, 21.0)
        self.assertAlmostEqual(result.share("high").estimate, 1.0)
        self.assertAlmostEqual(result.share("high").standard_error, np.sqrt(0.5))
        self.assertAlmostEqual(result.variance("x").estimate, 200.0)
        self.assertAlmostEqual(result.variance("x").standard_error, np.sqrt(25000.0))

        cdf = result.cdf("x", grid=[15.0, 25.0, 35.0, 45.0])
        np.testing.assert_allclose(cdf.grid, np.array([15.0, 25.0, 35.0, 45.0]))
        np.testing.assert_allclose(cdf.values, np.array([0.5, 1.0, 1.0, 1.0]))
        np.testing.assert_allclose(
            cdf.standard_errors,
            np.array([0.5, np.sqrt(0.5), 0.5, 0.0]),
        )

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

    def test_assignment_ate_ipw_matches_hand_calculation(self) -> None:
        dataset = make_assignment_dataset()
        result = ComplierEstimator(backend="abadie").fit(
            dataset,
            propensity_scores=np.full(dataset.n_obs, 0.5),
        )

        estimate = result.assignment_ate()

        self.assertEqual(estimate.name, "outcome")
        self.assertEqual(estimate.method, "ipw")
        self.assertAlmostEqual(estimate.assigned_mean, 9.0)
        self.assertAlmostEqual(estimate.unassigned_mean, 3.0)
        self.assertAlmostEqual(estimate.estimate, 6.0)
        self.assertAlmostEqual(estimate.standard_error, np.sqrt(37.0))

    def test_assignment_ate_accepts_explicit_outcome_feature_and_method_option(self) -> None:
        dataset = make_assignment_dataset()
        propensities = np.array([0.4, 0.6, 0.8, 0.5])
        custom_outcome = np.array([1.0, 3.0, 5.0, 7.0])
        result = ComplierEstimator(backend="abadie").fit(
            dataset,
            propensity_scores=propensities,
        )

        estimate = result.assignment_ate(custom_outcome, method="ipw", name="custom")

        expected_assigned = float(np.mean(dataset.instrument * custom_outcome / propensities))
        expected_unassigned = float(
            np.mean((1.0 - dataset.instrument) * custom_outcome / (1.0 - propensities))
        )
        self.assertEqual(estimate.name, "custom")
        self.assertAlmostEqual(estimate.assigned_mean, expected_assigned)
        self.assertAlmostEqual(estimate.unassigned_mean, expected_unassigned)
        self.assertAlmostEqual(estimate.estimate, expected_assigned - expected_unassigned)
        self.assertIsNotNone(estimate.standard_error)

    def test_assignment_ate_dr_matches_hand_calculation_with_supplied_outcome_models(self) -> None:
        dataset = make_assignment_dataset()
        propensities = np.array([0.4, 0.6, 0.8, 0.5])
        outcome_if_z0 = np.array([2.5, 3.5, 4.5, 5.5])
        outcome_if_z1 = np.array([7.5, 8.5, 9.5, 10.5])
        result = ComplierEstimator(backend="abadie").fit(
            dataset,
            propensity_scores=propensities,
        )

        estimate = result.assignment_ate(
            method="dr",
            outcome_if_z0=outcome_if_z0,
            outcome_if_z1=outcome_if_z1,
        )

        z = dataset.instrument
        y = dataset.outcome
        expected_assigned = float(np.mean(outcome_if_z1 + z * (y - outcome_if_z1) / propensities))
        expected_unassigned = float(
            np.mean(outcome_if_z0 + (1.0 - z) * (y - outcome_if_z0) / (1.0 - propensities))
        )
        self.assertEqual(estimate.method, "dr")
        self.assertAlmostEqual(estimate.assigned_mean, expected_assigned)
        self.assertAlmostEqual(estimate.unassigned_mean, expected_unassigned)
        self.assertAlmostEqual(estimate.estimate, expected_assigned - expected_unassigned)
        self.assertIsNotNone(estimate.standard_error)

    def test_assignment_ate_dr_on_result_estimates_default_outcome_model(self) -> None:
        dataset = make_assignment_dataset()
        result = ComplierEstimator(backend="abadie").fit(
            dataset,
            propensity_scores=np.full(dataset.n_obs, 0.5),
        )

        estimate = result.assignment_ate(method="dr")

        self.assertEqual(estimate.method, "dr")
        self.assertAlmostEqual(estimate.assigned_mean, 9.0)
        self.assertAlmostEqual(estimate.unassigned_mean, 3.0)
        self.assertAlmostEqual(estimate.estimate, 6.0)
        self.assertAlmostEqual(estimate.standard_error, 1.0)

    def test_potential_outcome_means_match_hand_calculation(self) -> None:
        dataset = make_potential_outcome_dataset()
        result = ComplierEstimator(backend="abadie").fit(
            dataset,
            propensity_scores=np.full(dataset.n_obs, 0.5),
        )

        untreated = result.untreated_outcome_mean()
        treated = result.treated_outcome_mean()

        self.assertEqual(untreated.name, "Y_0")
        self.assertEqual(treated.name, "Y_1")
        self.assertAlmostEqual(untreated.numerator, 1.5)
        self.assertAlmostEqual(treated.numerator, 3.5)
        self.assertAlmostEqual(untreated.denominator, 0.5)
        self.assertAlmostEqual(treated.denominator, 0.5)
        self.assertAlmostEqual(untreated.estimate, 3.0)
        self.assertAlmostEqual(treated.estimate, 7.0)
        self.assertIsNotNone(untreated.standard_error)
        self.assertIsNotNone(treated.standard_error)

    def test_potential_outcome_mean_accepts_generic_and_custom_outcome(self) -> None:
        dataset = make_potential_outcome_dataset()
        result = ComplierEstimator(backend="dr").fit(
            dataset,
            propensity_scores=np.full(dataset.n_obs, 0.5),
            treated_if_z0=np.full(dataset.n_obs, 0.25),
            treated_if_z1=np.full(dataset.n_obs, 0.75),
        )

        untreated = result.potential_outcome_mean(0)
        treated_custom = result.potential_outcome_mean(
            1,
            np.asarray(dataset.outcome) + 1.0,
            name="custom_Y_1",
        )

        self.assertAlmostEqual(untreated.estimate, 3.0)
        self.assertEqual(treated_custom.name, "custom_Y_1")
        self.assertAlmostEqual(treated_custom.estimate, 8.0)

    def test_assignment_ate_validates_method_and_default_outcome(self) -> None:
        dataset = make_fixed_dataset()
        result = ComplierEstimator(backend="abadie").fit(
            dataset,
            propensity_scores=np.full(dataset.n_obs, 0.5),
        )

        with self.assertRaisesRegex(ValueError, "requires an outcome"):
            result.assignment_ate()

        with self.assertRaisesRegex(ValueError, "Unsupported assignment ATE method"):
            result.assignment_ate([1.0, 2.0, 3.0, 4.0], method="aipw")

        with self.assertRaisesRegex(ValueError, "outcome_if_z0 and outcome_if_z1"):
            result.assignment_ate(
                [1.0, 2.0, 3.0, 4.0],
                method="dr",
                outcome_if_z0=np.zeros(dataset.n_obs),
            )

        with self.assertRaisesRegex(ValueError, "requires an outcome"):
            result.untreated_outcome_mean()

        with self.assertRaisesRegex(ValueError, "treatment_value must be 0 or 1"):
            result.potential_outcome_mean(2, [1.0, 2.0, 3.0, 4.0])

    def test_plugin_result_moments_work_without_propensities(self) -> None:
        dataset = make_fixed_dataset()
        result = ComplierEstimator(backend="plugin").fit(
            dataset,
            treated_if_z0=np.array([0.1, 0.1, 0.3, 0.5]),
            treated_if_z1=np.array([0.1, 0.3, 0.7, 0.9]),
        )

        self.assertEqual(result.backend, "plugin")
        self.assertIsNone(result.propensities)
        self.assertAlmostEqual(result.mean("x").estimate, 32.0)
        self.assertAlmostEqual(result.share("high").estimate, 0.6)

        with self.assertRaisesRegex(ValueError, "requires instrument propensities"):
            result.assignment_ate([1.0, 2.0, 3.0, 4.0])

        with self.assertRaisesRegex(ValueError, "requires instrument propensities"):
            result.potential_outcome_mean(0, [1.0, 2.0, 3.0, 4.0])

    def test_plugin_result_uses_supplied_propensities_for_ipw_only_methods(self) -> None:
        dataset = make_assignment_dataset()
        result = ComplierEstimator(backend="plugin").fit(
            dataset,
            propensity_scores=np.full(dataset.n_obs, 0.5),
            treated_if_z0=np.zeros(dataset.n_obs),
            treated_if_z1=np.full(dataset.n_obs, 0.5),
        )

        estimate = result.assignment_ate()

        np.testing.assert_allclose(result.propensities, np.full(dataset.n_obs, 0.5))
        self.assertAlmostEqual(estimate.assigned_mean, 9.0)
        self.assertAlmostEqual(estimate.unassigned_mean, 3.0)
        self.assertAlmostEqual(estimate.estimate, 6.0)


class SyntheticApiPointEstimateTests(unittest.TestCase):
    def assert_result_matches_known_estimands(
        self,
        result,
        expected: dict[str, float | np.ndarray],
    ) -> None:
        self.assertAlmostEqual(
            result.complier_share,
            float(expected["complier_share"]),
            places=12,
        )
        self.assertAlmostEqual(
            result.diagnostics.first_stage,
            float(expected["complier_share"]),
            places=12,
        )

        mean_x = result.mean("x")
        share_high = result.share("high")
        variance_x = result.variance("x")
        cdf_x = result.cdf("x", grid=[2.0, 4.0, 6.0])
        summary = result.summarize_covariates(["x", "high"])

        self.assertAlmostEqual(mean_x.estimate, float(expected["mean_x"]), places=12)
        self.assertAlmostEqual(share_high.estimate, float(expected["share_high"]), places=12)
        self.assertAlmostEqual(variance_x.estimate, float(expected["variance_x"]), places=12)
        np.testing.assert_allclose(cdf_x.values, expected["cdf_x"], atol=1e-12)
        self.assertAlmostEqual(summary["x"]["mean"], float(expected["mean_x"]), places=12)
        self.assertAlmostEqual(summary["high"]["share"], float(expected["share_high"]), places=12)

        untreated = result.untreated_outcome_mean()
        treated = result.treated_outcome_mean()
        self.assertAlmostEqual(
            untreated.estimate,
            float(expected["untreated_mean"]),
            places=12,
        )
        self.assertAlmostEqual(treated.estimate, float(expected["treated_mean"]), places=12)
        self.assertAlmostEqual(
            treated.estimate - untreated.estimate,
            float(expected["late"]),
            places=12,
        )

        assignment_ipw = result.assignment_ate(method="ipw")
        assignment_dr = result.assignment_ate(method="dr")
        for estimate in (assignment_ipw, assignment_dr):
            self.assertAlmostEqual(
                estimate.assigned_mean,
                float(expected["assigned_mean"]),
                places=12,
            )
            self.assertAlmostEqual(
                estimate.unassigned_mean,
                float(expected["unassigned_mean"]),
                places=12,
            )
            self.assertAlmostEqual(
                estimate.estimate,
                float(expected["assignment_ate"]),
                places=12,
            )

    def test_fitted_backends_recover_known_synthetic_point_estimates(self) -> None:
        dataset, expected, propensities, treated_if_z0, treated_if_z1 = (
            make_balanced_latent_type_dataset()
        )
        fitted_results = {
            "abadie": ComplierEstimator(
                backend="abadie",
                propensity_model="constant",
                clip=0.0,
            ).fit(dataset),
            "plugin": ComplierEstimator(
                backend="plugin",
                clip=0.0,
            ).fit(
                dataset,
                propensity_scores=propensities,
                treated_if_z0=treated_if_z0,
                treated_if_z1=treated_if_z1,
            ),
            "dr": ComplierEstimator(
                backend="dr",
                clip=0.0,
            ).fit(
                dataset,
                propensity_scores=propensities,
                treated_if_z0=treated_if_z0,
                treated_if_z1=treated_if_z1,
            ),
        }

        for backend, result in fitted_results.items():
            with self.subTest(backend=backend):
                self.assert_result_matches_known_estimands(result, expected)

    def test_direct_assignment_ate_api_recovers_known_synthetic_point_estimates(self) -> None:
        dataset, expected, _, _, _ = make_balanced_latent_type_dataset()
        estimator = ComplierEstimator(propensity_model="constant")

        for method in ("ipw", "dr"):
            with self.subTest(method=method):
                estimate = estimator.assignment_ate(dataset, method=method)
                self.assertAlmostEqual(
                    estimate.assigned_mean,
                    float(expected["assigned_mean"]),
                    places=12,
                )
                self.assertAlmostEqual(
                    estimate.unassigned_mean,
                    float(expected["unassigned_mean"]),
                    places=12,
                )
                self.assertAlmostEqual(
                    estimate.estimate,
                    float(expected["assignment_ate"]),
                    places=12,
                )


class EstimatorApiTests(unittest.TestCase):
    def test_assignment_ate_on_estimator_does_not_require_first_stage(self) -> None:
        dataset = ComplierDataset.from_arrays(
            instrument=[0, 0, 1, 1],
            treatment=[0, 1, 0, 1],
            outcome=[2.0, 4.0, 8.0, 10.0],
        )

        estimate = ComplierEstimator(propensity_model="constant").assignment_ate(dataset)

        self.assertAlmostEqual(estimate.assigned_mean, 9.0)
        self.assertAlmostEqual(estimate.unassigned_mean, 3.0)
        self.assertAlmostEqual(estimate.estimate, 6.0)

    def test_assignment_ate_dr_on_estimator_uses_outcome_model_without_first_stage(self) -> None:
        dataset = ComplierDataset.from_arrays(
            instrument=[0, 0, 1, 1],
            treatment=[0, 1, 0, 1],
            outcome=[2.0, 4.0, 8.0, 10.0],
        )

        estimate = ComplierEstimator(propensity_model="constant").assignment_ate(
            dataset,
            method="dr",
        )

        self.assertEqual(estimate.method, "dr")
        self.assertAlmostEqual(estimate.assigned_mean, 9.0)
        self.assertAlmostEqual(estimate.unassigned_mean, 3.0)
        self.assertAlmostEqual(estimate.estimate, 6.0)

    def test_assignment_ate_rejects_partial_external_outcome_models(self) -> None:
        dataset = make_assignment_dataset()

        with self.assertRaisesRegex(ValueError, "must be supplied together"):
            ComplierEstimator().assignment_ate(
                dataset,
                method="dr",
                outcome_if_z0=np.zeros(dataset.n_obs),
            )

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

        with self.assertRaisesRegex(ValueError, "must be supplied together"):
            ComplierEstimator(backend="plugin").fit(
                dataset,
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
