"""High-level estimator API."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike

from .data import ComplierDataset, FeatureSpec
from .estimators import (
    fit_abadie_backend,
    fit_doubly_robust_backend,
    fit_plugin_backend,
)
from .nuisance import (
    estimate_outcome_responses,
    estimate_propensity_scores,
    estimate_treatment_responses,
)
from .results import (
    ASSIGNMENT_ATE_METHODS,
    AssignmentEffectEstimate,
    ComplierResult,
    estimate_assignment_ate,
)


@dataclass
class ComplierEstimator:
    """Estimate complier descriptive functionals from a binary IV design.

    Parameters
    ----------
    backend:
        Either `"abadie"` for the default kappa-weighted estimator,
        `"plugin"` for a first-stage plug-in estimator, or `"dr"` for an
        augmented doubly robust score for average complier characteristics.
    normalize:
        If `True`, rescale the raw scores so their sample mean equals one.
        This leaves ratio estimates unchanged but makes the effective weights
        easier to interpret and aligns with the normalization emphasis in the
        recent weighting literature.
    propensity_model:
        Strategy used when propensity scores are not supplied explicitly.
        Supported values are `"constant"`, `"linear"`, `"logit"`, and
        `"probit"`.
        The `"plugin"` backend does not estimate propensities unless
        `propensity_scores` are supplied for post-fit IPW-only methods.
    treatment_model:
        Strategy used by the plug-in and doubly robust backends when treatment
        regressions are not supplied explicitly. Supported values are
        `"constant"`, `"linear"`, `"logit"`, and `"probit"`.
    assignment_outcome_model:
        Strategy used by doubly robust assignment ATE estimation when outcome
        regressions are not supplied explicitly. Supported values are
        `"constant"` and `"linear"`.
    covariate_names:
        Optional subset of covariates to use in nuisance estimation.
    clip:
        Lower and upper clipping threshold for nuisance probabilities.
    """

    backend: str = "abadie"
    normalize: bool = True
    propensity_model: str = "constant"
    treatment_model: str = "constant"
    assignment_outcome_model: str = "constant"
    covariate_names: list[str] | None = None
    clip: float = 1e-6

    def assignment_ate(
        self,
        dataset: ComplierDataset,
        *,
        outcome: FeatureSpec = "outcome",
        method: str = "ipw",
        propensity_scores: ArrayLike | None = None,
        outcome_if_z0: ArrayLike | None = None,
        outcome_if_z1: ArrayLike | None = None,
        name: str | None = None,
    ) -> AssignmentEffectEstimate:
        """Estimate the average effect of instrument assignment on an outcome.

        This path does not use treatment take-up scores, so it does not require
        a valid first stage for treatment take-up.
        """

        if method not in ASSIGNMENT_ATE_METHODS:
            raise ValueError(f"Unsupported assignment ATE method {method!r}.")

        propensities = self._resolve_propensities(dataset, propensity_scores)
        if method == "dr":
            outcome_if_z0, outcome_if_z1 = self._resolve_assignment_outcome_regressions(
                dataset,
                outcome=outcome,
                outcome_if_z0=outcome_if_z0,
                outcome_if_z1=outcome_if_z1,
            )
        return estimate_assignment_ate(
            dataset,
            propensities,
            outcome=outcome,
            method=method,
            outcome_if_z0=outcome_if_z0,
            outcome_if_z1=outcome_if_z1,
            name=name,
        )

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

        if self.backend not in {"abadie", "plugin", "dr"}:
            raise ValueError(f"Unsupported backend {self.backend!r}.")

        if self.backend == "abadie":
            propensities = self._resolve_propensities(dataset, propensity_scores)
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
        if self.backend == "plugin":
            propensities = (
                self._resolve_propensities(dataset, propensity_scores)
                if propensity_scores is not None
                else None
            )
            backend_result = fit_plugin_backend(
                dataset,
                treated_if_z0=m0,
                treated_if_z1=m1,
                normalize=self.normalize,
                propensities=propensities,
            )
            return ComplierResult(dataset, backend_result)

        propensities = self._resolve_propensities(dataset, propensity_scores)
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
        if (treated_if_z0 is None) != (treated_if_z1 is None):
            raise ValueError("treated_if_z0 and treated_if_z1 must be supplied together.")

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

    def _resolve_assignment_outcome_regressions(
        self,
        dataset: ComplierDataset,
        *,
        outcome: FeatureSpec,
        outcome_if_z0: ArrayLike | None,
        outcome_if_z1: ArrayLike | None,
    ) -> tuple[np.ndarray, np.ndarray]:
        if (outcome_if_z0 is None) != (outcome_if_z1 is None):
            raise ValueError("outcome_if_z0 and outcome_if_z1 must be supplied together.")

        if outcome_if_z0 is not None and outcome_if_z1 is not None:
            y0 = dataset.resolve_feature(outcome_if_z0, name="outcome_if_z0")
            y1 = dataset.resolve_feature(outcome_if_z1, name="outcome_if_z1")
            return y0, y1

        return estimate_outcome_responses(
            dataset,
            outcome=outcome,
            model=self.assignment_outcome_model,
            covariate_names=self.covariate_names,
        )
