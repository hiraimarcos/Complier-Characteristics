"""Simple nuisance estimation helpers.

The package keeps nuisance estimation intentionally lightweight:

- constant models are available for intercept-only designs
- a small logistic-regression implementation is included so the package does not
  depend on scikit-learn or statsmodels

These routines are deliberately transparent rather than feature-rich.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from .data import ComplierDataset


def expit(values: NDArray[np.float64]) -> NDArray[np.float64]:
    """Numerically stable logistic link."""

    clipped = np.clip(values, -35.0, 35.0)
    return 1.0 / (1.0 + np.exp(-clipped))


def add_intercept(features: NDArray[np.float64]) -> NDArray[np.float64]:
    """Prepend an intercept column to a feature matrix."""

    n_obs = features.shape[0]
    intercept = np.ones((n_obs, 1), dtype=float)
    if features.shape[1] == 0:
        return intercept
    return np.column_stack([intercept, features])


def fit_logistic_regression(
    features: NDArray[np.float64],
    target: NDArray[np.float64],
    *,
    prediction_features: NDArray[np.float64] | None = None,
    ridge: float = 1e-6,
    max_iter: int = 200,
    tol: float = 1e-8,
    clip: float = 1e-6,
) -> NDArray[np.float64]:
    """Fit a small logistic regression by iteratively reweighted least squares.

    The goal is not to be a general-purpose classifier. The goal is to provide
    readable nuisance estimation for a package that otherwise only depends on
    `numpy`.
    """

    if features.ndim != 2:
        raise ValueError("features must be a two-dimensional matrix.")

    prediction_matrix = features if prediction_features is None else prediction_features

    if np.all(target == target[0]):
        constant = float(np.clip(target.mean(), clip, 1.0 - clip))
        return np.full(prediction_matrix.shape[0], constant, dtype=float)

    design = add_intercept(features)
    prediction_design = add_intercept(prediction_matrix)
    coefficients = np.zeros(design.shape[1], dtype=float)
    penalty = ridge * np.eye(design.shape[1], dtype=float)
    penalty[0, 0] = 0.0

    for _ in range(max_iter):
        linear_index = design @ coefficients
        probabilities = np.clip(expit(linear_index), clip, 1.0 - clip)
        weights = np.clip(probabilities * (1.0 - probabilities), clip, None)
        working_response = linear_index + (target - probabilities) / weights

        weighted_design = design * weights[:, None]
        lhs = design.T @ weighted_design + penalty
        rhs = design.T @ (weights * working_response)

        try:
            updated = np.linalg.solve(lhs, rhs)
        except np.linalg.LinAlgError:
            updated = np.linalg.lstsq(lhs, rhs, rcond=None)[0]

        if np.max(np.abs(updated - coefficients)) < tol:
            coefficients = updated
            break
        coefficients = updated

    return np.clip(expit(prediction_design @ coefficients), clip, 1.0 - clip)


def estimate_propensity_scores(
    dataset: ComplierDataset,
    *,
    model: str = "constant",
    covariate_names: list[str] | None = None,
    clip: float = 1e-6,
) -> NDArray[np.float64]:
    """Estimate `P(Z=1 | X)` under a simple constant or logit model."""

    if model not in {"constant", "logit"}:
        raise ValueError(f"Unsupported propensity model {model!r}.")

    if model == "constant":
        probability = float(np.clip(dataset.instrument.mean(), clip, 1.0 - clip))
        return np.full(dataset.n_obs, probability, dtype=float)

    features = dataset.covariate_matrix(covariate_names)
    if features.shape[1] == 0:
        probability = float(np.clip(dataset.instrument.mean(), clip, 1.0 - clip))
        return np.full(dataset.n_obs, probability, dtype=float)

    return fit_logistic_regression(features, dataset.instrument, clip=clip)


def estimate_treatment_responses(
    dataset: ComplierDataset,
    *,
    model: str = "constant",
    covariate_names: list[str] | None = None,
    clip: float = 1e-6,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Estimate `E[D | Z=0, X]` and `E[D | Z=1, X]`.

    The returned arrays are predicted treatment probabilities for every
    observation under each instrument state.
    """

    if model not in {"constant", "logit"}:
        raise ValueError(f"Unsupported treatment model {model!r}.")

    features = dataset.covariate_matrix(covariate_names)
    z = dataset.instrument.astype(bool)
    d = dataset.treatment

    def _fit_within_stratum(mask: NDArray[np.bool_]) -> NDArray[np.float64]:
        if not np.any(mask):
            raise ValueError("Each instrument stratum must contain at least one observation.")

        if model == "constant" or features.shape[1] == 0:
            probability = float(np.clip(d[mask].mean(), clip, 1.0 - clip))
            return np.full(dataset.n_obs, probability, dtype=float)

        return fit_logistic_regression(
            features[mask],
            d[mask],
            prediction_features=features,
            clip=clip,
        )

    treated_if_z0 = _fit_within_stratum(~z)
    treated_if_z1 = _fit_within_stratum(z)
    return treated_if_z0, treated_if_z1
