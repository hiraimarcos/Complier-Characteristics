"""Data containers and feature-resolution helpers.

This module keeps all low-level validation in one place. The rest of the
package can then assume that:

- the instrument and treatment are one-dimensional binary arrays
- all stored covariates have the same length as the treatment sample
- feature maps resolve to one-dimensional numeric arrays
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Mapping

import numpy as np
from numpy.typing import ArrayLike, NDArray


FeatureSpec = str | ArrayLike | Callable[["ComplierDataset"], ArrayLike]


def _as_1d_array(values: ArrayLike, *, name: str) -> NDArray[np.float64]:
    """Convert arbitrary array-like input into a one-dimensional float array."""

    array = np.asarray(values, dtype=float)
    if array.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional; got shape {array.shape!r}.")
    if array.size == 0:
        raise ValueError(f"{name} must contain at least one observation.")
    if np.isnan(array).any():
        raise ValueError(f"{name} must not contain missing values.")
    return array


def _as_binary_array(values: ArrayLike, *, name: str) -> NDArray[np.float64]:
    """Convert an input into a numeric 0/1 array and reject non-binary data."""

    array = _as_1d_array(values, name=name)
    unique = np.unique(array)
    if not np.all(np.isin(unique, (0.0, 1.0))):
        raise ValueError(f"{name} must be binary with values in {{0, 1}}; got {unique!r}.")
    return array


@dataclass(frozen=True)
class ComplierDataset:
    """Validated observed data for a binary-instrument LATE design.

    Parameters
    ----------
    instrument:
        One-dimensional binary array containing the instrument `Z`.
    treatment:
        One-dimensional binary array containing the realized treatment `D`.
    covariates:
        Mapping from covariate names to one-dimensional numeric arrays.
    outcome:
        Optional one-dimensional numeric array. The v1 package focuses on
        covariate profiling, but storing an outcome here is still useful for
        custom feature maps.
    """

    instrument: NDArray[np.float64]
    treatment: NDArray[np.float64]
    covariates: Mapping[str, NDArray[np.float64]] = field(default_factory=dict)
    outcome: NDArray[np.float64] | None = None

    @classmethod
    def from_arrays(
        cls,
        *,
        instrument: ArrayLike,
        treatment: ArrayLike,
        covariates: Mapping[str, ArrayLike] | None = None,
        outcome: ArrayLike | None = None,
    ) -> "ComplierDataset":
        """Validate raw arrays and build a :class:`ComplierDataset` instance."""

        z = _as_binary_array(instrument, name="instrument")
        d = _as_binary_array(treatment, name="treatment")
        if z.shape != d.shape:
            raise ValueError("instrument and treatment must have the same length.")

        validated_covariates: dict[str, NDArray[np.float64]] = {}
        for key, value in (covariates or {}).items():
            covariate = _as_1d_array(value, name=f"covariates[{key!r}]")
            if covariate.shape != z.shape:
                raise ValueError(
                    f"covariate {key!r} must have length {z.size}; got {covariate.size}."
                )
            validated_covariates[key] = covariate

        validated_outcome = None
        if outcome is not None:
            validated_outcome = _as_1d_array(outcome, name="outcome")
            if validated_outcome.shape != z.shape:
                raise ValueError("outcome must have the same length as instrument.")

        return cls(
            instrument=z,
            treatment=d,
            covariates=validated_covariates,
            outcome=validated_outcome,
        )

    @property
    def n_obs(self) -> int:
        """Return the number of observations in the dataset."""

        return int(self.instrument.size)

    def covariate_names(self) -> list[str]:
        """Return the stored covariate names in insertion order."""

        return list(self.covariates.keys())

    def covariate_matrix(self, names: list[str] | None = None) -> NDArray[np.float64]:
        """Stack the requested covariates column-wise into an `n x p` matrix."""

        selected_names = self.covariate_names() if names is None else names
        if not selected_names:
            return np.empty((self.n_obs, 0), dtype=float)

        columns = []
        for name in selected_names:
            if name not in self.covariates:
                raise KeyError(f"Unknown covariate {name!r}.")
            columns.append(self.covariates[name])

        return np.column_stack(columns).astype(float)

    def resolve_feature(self, feature: FeatureSpec, *, name: str | None = None) -> NDArray[np.float64]:
        """Resolve a feature specification into a one-dimensional numeric array.

        Supported inputs are:

        - a covariate name
        - a one-dimensional array-like object
        - a callable that accepts the current dataset and returns an array
        """

        label = name or "feature"
        if isinstance(feature, str):
            if feature in self.covariates:
                return self.covariates[feature]
            if feature == "outcome" and self.outcome is not None:
                return self.outcome
            raise KeyError(f"Unknown feature {feature!r}.")

        if callable(feature):
            resolved = _as_1d_array(feature(self), name=label)
            if resolved.shape != self.instrument.shape:
                raise ValueError(f"{label} must have length {self.n_obs}; got {resolved.size}.")
            return resolved

        resolved = _as_1d_array(feature, name=label)
        if resolved.shape != self.instrument.shape:
            raise ValueError(f"{label} must have length {self.n_obs}; got {resolved.size}.")
        return resolved
