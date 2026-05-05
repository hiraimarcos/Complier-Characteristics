"""Simple nuisance estimation helpers.

The package keeps nuisance estimation intentionally lightweight:

- constant models are available for intercept-only designs
- logit and probit regression are delegated to statsmodels' binomial GLM

These routines are deliberately transparent rather than feature-rich.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from statsmodels.genmod.families import Binomial
from statsmodels.genmod.families.links import Logit, Probit
from statsmodels.genmod.generalized_linear_model import GLM

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
    """Fit logistic regression with statsmodels and return clipped probabilities."""

    return fit_binary_response_regression(
        features,
        target,
        link="logit",
        prediction_features=prediction_features,
        ridge=ridge,
        max_iter=max_iter,
        tol=tol,
        clip=clip,
    )


def fit_probit_regression(
    features: NDArray[np.float64],
    target: NDArray[np.float64],
    *,
    prediction_features: NDArray[np.float64] | None = None,
    ridge: float = 1e-6,
    max_iter: int = 200,
    tol: float = 1e-8,
    clip: float = 1e-6,
) -> NDArray[np.float64]:
    """Fit probit regression with statsmodels and return clipped probabilities."""

    return fit_binary_response_regression(
        features,
        target,
        link="probit",
        prediction_features=prediction_features,
        ridge=ridge,
        max_iter=max_iter,
        tol=tol,
        clip=clip,
    )


def fit_binary_response_regression(
    features: NDArray[np.float64],
    target: NDArray[np.float64],
    *,
    link: str,
    prediction_features: NDArray[np.float64] | None = None,
    ridge: float = 1e-6,
    max_iter: int = 200,
    tol: float = 1e-8,
    clip: float = 1e-6,
) -> NDArray[np.float64]:
    """Fit a statsmodels binomial GLM and return clipped probabilities."""

    if features.ndim != 2:
        raise ValueError("features must be a two-dimensional matrix.")

    if link not in {"logit", "probit"}:
        raise ValueError(f"Unsupported binary response link {link!r}.")

    prediction_matrix = features if prediction_features is None else prediction_features
    if prediction_matrix.ndim != 2:
        raise ValueError("prediction_features must be a two-dimensional matrix.")

    if np.all(target == target[0]):
        constant = float(np.clip(target.mean(), clip, 1.0 - clip))
        return np.full(prediction_matrix.shape[0], constant, dtype=float)

    design = add_intercept(features)
    prediction_design = add_intercept(prediction_matrix)
    family = Binomial(link=Logit() if link == "logit" else Probit())
    model = GLM(target, design, family=family)

    if ridge > 0.0:
        alpha = np.full(design.shape[1], ridge, dtype=float)
        alpha[0] = 0.0
        result = model.fit_regularized(
            alpha=alpha,
            L1_wt=0.0,
            maxiter=max_iter,
            cnvrg_tol=tol,
        )
    else:
        result = model.fit(maxiter=max_iter, tol=tol, disp=0)

    predictions = np.asarray(result.predict(prediction_design), dtype=float)
    return np.clip(predictions, clip, 1.0 - clip)


def estimate_propensity_scores(
    dataset: ComplierDataset,
    *,
    model: str = "constant",
    covariate_names: list[str] | None = None,
    clip: float = 1e-6,
) -> NDArray[np.float64]:
    """Estimate `P(Z=1 | X)` under a constant, logit, or probit model."""

    if model not in {"constant", "logit", "probit"}:
        raise ValueError(f"Unsupported propensity model {model!r}.")

    if model == "constant":
        probability = float(np.clip(dataset.instrument.mean(), clip, 1.0 - clip))
        return np.full(dataset.n_obs, probability, dtype=float)

    features = dataset.covariate_matrix(covariate_names)
    if features.shape[1] == 0:
        probability = float(np.clip(dataset.instrument.mean(), clip, 1.0 - clip))
        return np.full(dataset.n_obs, probability, dtype=float)

    return fit_binary_response_regression(
        features,
        dataset.instrument,
        link=model,
        clip=clip,
    )


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

    if model not in {"constant", "logit", "probit"}:
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

        return fit_binary_response_regression(
            features[mask],
            d[mask],
            link=model,
            prediction_features=features,
            clip=clip,
        )

    treated_if_z0 = _fit_within_stratum(~z)
    treated_if_z1 = _fit_within_stratum(z)
    return treated_if_z0, treated_if_z1
