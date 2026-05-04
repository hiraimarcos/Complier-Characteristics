"""Tests for validated data containers and feature resolution."""

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


def make_dataset() -> ComplierDataset:
    return ComplierDataset.from_arrays(
        instrument=[0, 1, 0],
        treatment=[0, 1, 0],
        outcome=[5.0, 6.0, 7.0],
        covariates={
            "x": [1.0, 2.0, 3.0],
            "w": [4.0, 5.0, 6.0],
        },
    )


class ComplierDatasetValidationTests(unittest.TestCase):
    def test_from_arrays_validates_core_inputs(self) -> None:
        with self.assertRaisesRegex(ValueError, "one-dimensional"):
            ComplierDataset.from_arrays(
                instrument=[[0, 1]],
                treatment=[[0, 1]],
            )

        with self.assertRaisesRegex(ValueError, "same length"):
            ComplierDataset.from_arrays(
                instrument=[0, 1],
                treatment=[0],
            )

        with self.assertRaisesRegex(ValueError, "missing values"):
            ComplierDataset.from_arrays(
                instrument=[0, 1],
                treatment=[0, 1],
                covariates={"x": [1.0, np.nan]},
            )

        with self.assertRaisesRegex(ValueError, "must have length 2"):
            ComplierDataset.from_arrays(
                instrument=[0, 1],
                treatment=[0, 1],
                covariates={"x": [1.0]},
            )

        with self.assertRaisesRegex(ValueError, "outcome must have the same length"):
            ComplierDataset.from_arrays(
                instrument=[0, 1],
                treatment=[0, 1],
                outcome=[1.0],
            )

    def test_covariate_matrix_preserves_order_and_accepts_empty_subset(self) -> None:
        dataset = make_dataset()

        np.testing.assert_allclose(
            dataset.covariate_matrix(),
            np.array(
                [
                    [1.0, 4.0],
                    [2.0, 5.0],
                    [3.0, 6.0],
                ]
            ),
        )
        np.testing.assert_allclose(dataset.covariate_matrix(["w"]), np.array([[4.0], [5.0], [6.0]]))
        self.assertEqual(dataset.covariate_matrix([]).shape, (3, 0))

        with self.assertRaisesRegex(KeyError, "Unknown covariate"):
            dataset.covariate_matrix(["missing"])

    def test_resolve_feature_accepts_public_feature_specs(self) -> None:
        dataset = make_dataset()

        np.testing.assert_allclose(dataset.resolve_feature("x"), np.array([1.0, 2.0, 3.0]))
        np.testing.assert_allclose(dataset.resolve_feature("outcome"), np.array([5.0, 6.0, 7.0]))
        np.testing.assert_allclose(dataset.resolve_feature([7.0, 8.0, 9.0]), np.array([7.0, 8.0, 9.0]))
        np.testing.assert_allclose(
            dataset.resolve_feature(lambda data: data.covariates["x"] + data.outcome),
            np.array([6.0, 8.0, 10.0]),
        )

    def test_resolve_feature_rejects_unknown_or_wrong_length_specs(self) -> None:
        dataset = make_dataset()

        with self.assertRaisesRegex(KeyError, "Unknown feature"):
            dataset.resolve_feature("missing")

        with self.assertRaisesRegex(ValueError, "must have length 3"):
            dataset.resolve_feature([1.0, 2.0], name="short_array")

        with self.assertRaisesRegex(ValueError, "must have length 3"):
            dataset.resolve_feature(lambda data: [1.0, 2.0], name="short_callable")


if __name__ == "__main__":
    unittest.main()
