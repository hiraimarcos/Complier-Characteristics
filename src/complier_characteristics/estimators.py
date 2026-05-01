"""Low-level score construction for each backend.

The package treats complier moments as weighted averages. The weight-like object
depends on the backend:

- Abadie backend:
  the raw score is the sample analogue of Abadie's `kappa`
- doubly robust backend:
  the raw score is an augmented estimate of complier membership that uses both
  the instrument propensity and treatment regressions

Both backends return a shared representation so the high-level result object can
evaluate different moments without duplicating estimator logic.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from .data import ComplierDataset
from .diagnostics import ComplierDiagnostics


@dataclass(frozen=True)
class BackendResult:
    """Common return object for fitted backends."""

    backend: str
    raw_scores: NDArray[np.float64]
    scaled_scores: NDArray[np.float64]
    propensities: NDArray[np.float64]
    complier_share: float
    diagnostics: ComplierDiagnostics
    treated_if_z0: NDArray[np.float64] | None = None
    treated_if_z1: NDArray[np.float64] | None = None


def _make_backend_result(
    *,
    dataset: ComplierDataset,
    backend: str,
    raw_scores: NDArray[np.float64],
    propensities: NDArray[np.float64],
    normalize: bool,
    treated_if_z0: NDArray[np.float64] | None = None,
    treated_if_z1: NDArray[np.float64] | None = None,
) -> BackendResult:
    """Finalize a backend fit by scaling scores and building diagnostics."""

    complier_share = float(np.mean(raw_scores))
    if abs(complier_share) < 1e-10:
        raise ValueError(
            "The estimated complier share is numerically zero. "
            "This usually indicates a vanishing first stage."
        )

    scaled_scores = raw_scores / complier_share if normalize else raw_scores.copy()
    first_stage = float(
        dataset.treatment[dataset.instrument == 1].mean()
        - dataset.treatment[dataset.instrument == 0].mean()
    )

    diagnostics = ComplierDiagnostics(
        n_obs=dataset.n_obs,
        backend=backend,
        normalized=normalize,
        instrument_rate=float(np.mean(dataset.instrument)),
        treatment_rate=float(np.mean(dataset.treatment)),
        first_stage=first_stage,
        complier_share=complier_share,
        min_propensity=float(np.min(propensities)),
        max_propensity=float(np.max(propensities)),
        negative_score_fraction=float(np.mean(raw_scores < 0.0)),
        score_mean=float(np.mean(scaled_scores)),
    )

    return BackendResult(
        backend=backend,
        raw_scores=raw_scores,
        scaled_scores=scaled_scores,
        propensities=propensities,
        complier_share=complier_share,
        diagnostics=diagnostics,
        treated_if_z0=treated_if_z0,
        treated_if_z1=treated_if_z1,
    )


def fit_abadie_backend(
    dataset: ComplierDataset,
    *,
    propensities: NDArray[np.float64],
    normalize: bool,
) -> BackendResult:
    """Construct the Abadie-style `kappa` scores.

    For binary `Z` and `D`, the raw score is

    `kappa_i = 1 - D_i (1-Z_i)/(1-p_i) - (1-D_i) Z_i / p_i`,

    where `p_i = P(Z_i = 1 | X_i)`.
    """

    z = dataset.instrument
    d = dataset.treatment
    p = propensities
    raw_scores = 1.0 - d * (1.0 - z) / (1.0 - p) - (1.0 - d) * z / p
    return _make_backend_result(
        dataset=dataset,
        backend="abadie",
        raw_scores=raw_scores,
        propensities=propensities,
        normalize=normalize,
    )


def fit_doubly_robust_backend(
    dataset: ComplierDataset,
    *,
    propensities: NDArray[np.float64],
    treated_if_z0: NDArray[np.float64],
    treated_if_z1: NDArray[np.float64],
    normalize: bool,
) -> BackendResult:
    """Construct a doubly robust score for average complier characteristics.

    The raw score is the augmented complier-membership signal

    `m1(X) - m0(X) + Z (D - m1(X)) / p(X) - (1-Z) (D - m0(X)) / (1-p(X))`,

    where:

    - `p(X) = P(Z=1 | X)`
    - `m1(X) = E[D | Z=1, X]`
    - `m0(X) = E[D | Z=0, X]`

    This is the object used by the result layer to estimate average complier
    characteristics of arbitrary feature maps `f(X)`.
    """

    z = dataset.instrument
    d = dataset.treatment
    p = propensities
    m0 = treated_if_z0
    m1 = treated_if_z1

    raw_scores = (
        (m1 - m0)
        + z * (d - m1) / p
        - (1.0 - z) * (d - m0) / (1.0 - p)
    )
    return _make_backend_result(
        dataset=dataset,
        backend="dr",
        raw_scores=raw_scores,
        propensities=propensities,
        normalize=normalize,
        treated_if_z0=treated_if_z0,
        treated_if_z1=treated_if_z1,
    )
