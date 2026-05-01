"""High-level estimator API."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike

from .data import ComplierDataset
from .estimators import fit_abadie_backend, fit_doubly_robust_backend
from .nuisance import estimate_propensity_scores, estimate_treatment_responses
from .results import ComplierResult


@dataclass
class ComplierEstimator:
    """Estimate complier descriptive functionals from a binary IV design.

    Parameters
    ----------
    backend:
        Either `"abadie"` for the default kappa-weighted estimator or `"dr"`
        for an augmented doubly robust score for average complier
        characteristics.
    normalize:
        If `True`, rescale the raw scores so their sample mean equals one.
        This leaves ratio estimates unchanged but makes the effective weights
        easier to interpret and aligns with the normalization emphasis in the
        recent weighting literature.
    propensity_model:
        Strategy used when propensity scores are not supplied explicitly.
        Supported values are `"constant"` and `"logit"`.
    treatment_model:
        Strategy used by the doubly robust backend when treatment regressions
        are not supplied explicitly. Supported values are `"constant"` and
        `"logit"`.
    covariate_names:
        Optional subset of covariates to use in nuisance estimation.
    clip:
        Lower and upper clipping threshold for nuisance probabilities.
    """

    backend: str = "abadie"
    normalize: bool = True
    propensity_model: str = "constant"
    treatment_model: str = "constant"
    covariate_names: list[str] | None = None
    clip: float = 1e-6

    def fit(
        self,
        dataset: ComplierDataset,
        *,
        propensity_scores: ArrayLike | None = None,
        treated_if_z0: ArrayLike | None = None,
        treated_if_z1: ArrayLike | None = None,
    ) -> ComplierResult:
        """Fit the chosen backend on a validated dataset.

        Users who want to integrate external machine-learning nuisance models
        can bypass the built-in estimators by supplying:

        - `propensity_scores`
        - `treated_if_z0`
        - `treated_if_z1`
        """

        if self.backend not in {"abadie", "dr"}:
            raise ValueError(f"Unsupported backend {self.backend!r}.")

        propensities = self._resolve_propensities(dataset, propensity_scores)

        if self.backend == "abadie":
            backend_result = fit_abadie_backend(
                dataset,
                propensities=propensities,
                normalize=self.normalize,
            )
            return ComplierResult(dataset, backend_result)

        m0, m1 = self._resolve_treatment_regressions(
            dataset,
            treated_if_z0=treated_if_z0,
            treated_if_z1=treated_if_z1,
        )
        backend_result = fit_doubly_robust_backend(
            dataset,
            propensities=propensities,
            treated_if_z0=m0,
            treated_if_z1=m1,
            normalize=self.normalize,
        )
        return ComplierResult(dataset, backend_result)

    def _resolve_propensities(
        self,
        dataset: ComplierDataset,
        propensity_scores: ArrayLike | None,
    ) -> np.ndarray:
        if propensity_scores is not None:
            supplied = dataset.resolve_feature(propensity_scores, name="propensity_scores")
            return np.clip(supplied, self.clip, 1.0 - self.clip)

        return estimate_propensity_scores(
            dataset,
            model=self.propensity_model,
            covariate_names=self.covariate_names,
            clip=self.clip,
        )

    def _resolve_treatment_regressions(
        self,
        dataset: ComplierDataset,
        *,
        treated_if_z0: ArrayLike | None,
        treated_if_z1: ArrayLike | None,
    ) -> tuple[np.ndarray, np.ndarray]:
        if treated_if_z0 is not None and treated_if_z1 is not None:
            m0 = np.clip(dataset.resolve_feature(treated_if_z0, name="treated_if_z0"), self.clip, 1.0 - self.clip)
            m1 = np.clip(dataset.resolve_feature(treated_if_z1, name="treated_if_z1"), self.clip, 1.0 - self.clip)
            return m0, m1

        return estimate_treatment_responses(
            dataset,
            model=self.treatment_model,
            covariate_names=self.covariate_names,
            clip=self.clip,
        )
