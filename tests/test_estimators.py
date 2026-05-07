"""Unit tests for the first package draft."""

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


def expit(values: np.ndarray) -> np.ndarray:
    values = np.clip(values, -35.0, 35.0)
    return 1.0 / (1.0 + np.exp(-values))


def make_dataset(
    *,
    seed: int,
    n_obs: int,
    conditional_instrument: bool,
) -> tuple[ComplierDataset, np.ndarray]:
    """Simulate a monotone design with a known complier population."""

    rng = np.random.default_rng(seed)
    x = rng.normal(size=n_obs)
    complier_probability = expit(-0.15 + 0.9 * x)
    complier = rng.binomial(1, complier_probability, size=n_obs).astype(float)

    if conditional_instrument:
        instrument_probability = expit(0.2 - 0.5 * x)
    else:
        instrument_probability = np.full(n_obs, 0.5)

    z = rng.binomial(1, instrument_probability, size=n_obs).astype(float)
    d = z * complier
    y = 1.0 + 2.0 * d + 0.5 * x + rng.normal(scale=0.25, size=n_obs)

    dataset = ComplierDataset.from_arrays(
        instrument=z,
        treatment=d,
        outcome=y,
        covariates={
            "x": x,
            "high_x": (x > 0.0).astype(float),
        },
    )
    return dataset, complier.astype(bool)


class DatasetValidationTests(unittest.TestCase):
    def test_non_binary_instrument_raises(self) -> None:
        with self.assertRaises(ValueError):
            ComplierDataset.from_arrays(
                instrument=[0, 1, 2],
                treatment=[0, 1, 0],
                covariates={"x": [1.0, 2.0, 3.0]},
            )


class AbadieBackendTests(unittest.TestCase):
    def test_abadie_backend_recovers_complier_profile_under_randomized_iv(self) -> None:
        dataset, complier = make_dataset(seed=123, n_obs=6000, conditional_instrument=False)

        estimator = ComplierEstimator(
            backend="abadie",
            normalize=True,
            propensity_model="constant",
        )
        result = estimator.fit(dataset)

        true_mean = float(dataset.covariates["x"][complier].mean())
        true_share = float(dataset.covariates["high_x"][complier].mean())

        self.assertAlmostEqual(result.mean("x").estimate, true_mean, delta=0.07)
        self.assertAlmostEqual(result.share("high_x").estimate, true_share, delta=0.05)

        cdf = result.cdf("x", grid=np.linspace(-1.5, 1.5, 7))
        self.assertTrue(np.all(np.diff(cdf.values) >= -1e-10))
        self.assertTrue(np.all((cdf.values >= 0.0) & (cdf.values <= 1.0)))


class DoublyRobustBackendTests(unittest.TestCase):
    def test_dr_backend_recovers_complier_profile_with_covariate_dependent_instrument(self) -> None:
        dataset, complier = make_dataset(seed=456, n_obs=7000, conditional_instrument=True)

        estimator = ComplierEstimator(
            backend="dr",
            normalize=True,
            propensity_model="logit",
            treatment_model="logit",
            covariate_names=["x"],
        )
        result = estimator.fit(dataset)

        true_mean = float(dataset.covariates["x"][complier].mean())
        true_share = float(dataset.covariates["high_x"][complier].mean())

        self.assertAlmostEqual(result.mean("x").estimate, true_mean, delta=0.10)
        self.assertAlmostEqual(result.share("high_x").estimate, true_share, delta=0.07)

        summary = result.summarize_covariates(["x", "high_x"])
        self.assertIn("mean", summary["x"])
        self.assertIn("variance", summary["x"])
        self.assertIn("share", summary["high_x"])

        diagnostics = result.diagnostics.to_dict()
        self.assertEqual(diagnostics["backend"], "dr")
        self.assertGreater(diagnostics["complier_share"], 0.0)


class PluginBackendTests(unittest.TestCase):
    def test_plugin_backend_recovers_complier_profile_without_propensity_scores(self) -> None:
        dataset, complier = make_dataset(seed=789, n_obs=7000, conditional_instrument=True)

        estimator = ComplierEstimator(
            backend="plugin",
            normalize=True,
            treatment_model="logit",
            covariate_names=["x"],
        )
        result = estimator.fit(dataset)

        true_mean = float(dataset.covariates["x"][complier].mean())
        true_share = float(dataset.covariates["high_x"][complier].mean())

        self.assertAlmostEqual(result.mean("x").estimate, true_mean, delta=0.10)
        self.assertAlmostEqual(result.share("high_x").estimate, true_share, delta=0.07)

        diagnostics = result.diagnostics.to_dict()
        self.assertEqual(diagnostics["backend"], "plugin")
        self.assertIsNone(diagnostics["min_propensity"])
        self.assertIsNone(diagnostics["max_propensity"])
        self.assertGreater(diagnostics["complier_share"], 0.0)


if __name__ == "__main__":
    unittest.main()
