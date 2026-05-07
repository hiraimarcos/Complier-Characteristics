"""Post-fit result objects and descriptive functionals."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from .data import ComplierDataset, FeatureSpec
from .diagnostics import ComplierDiagnostics
from .estimators import BackendResult
from .nuisance import estimate_outcome_responses


ASSIGNMENT_ATE_METHODS = {"ipw", "dr"}


def _standard_error_from_influence(influence_values: NDArray[np.float64]) -> float:
    """Return the plug-in standard error from fixed-nuisance influence values."""

    return float(np.sqrt(np.mean(influence_values**2) / influence_values.size))


@dataclass(frozen=True)
class ScalarEstimate:
    """Scalar estimate returned by `mean`, `share`, `variance`, or `moment`."""

    name: str
    estimate: float
    backend: str
    complier_share: float
    numerator: float | None = None
    denominator: float | None = None
    standard_error: float | None = None


@dataclass(frozen=True)
class DistributionEstimate:
    """Distributional estimate returned by `cdf`."""

    name: str
    grid: NDArray[np.float64]
    values: NDArray[np.float64]
    backend: str
    complier_share: float
    standard_errors: NDArray[np.float64] | None = None


@dataclass(frozen=True)
class AssignmentEffectEstimate:
    """Scalar estimate returned by `assignment_ate`."""

    name: str
    estimate: float
    method: str
    assigned_mean: float
    unassigned_mean: float
    standard_error: float | None = None


def estimate_assignment_ate(
    dataset: ComplierDataset,
    propensities: NDArray[np.float64],
    outcome: FeatureSpec = "outcome",
    *,
    method: str = "ipw",
    outcome_if_z0: FeatureSpec | None = None,
    outcome_if_z1: FeatureSpec | None = None,
    name: str | None = None,
) -> AssignmentEffectEstimate:
    """Estimate the average effect of instrument assignment on an outcome."""

    if method not in ASSIGNMENT_ATE_METHODS:
        raise ValueError(f"Unsupported assignment ATE method {method!r}.")

    if method == "ipw":
        return _estimate_assignment_ate_ipw(
            dataset,
            propensities,
            outcome=outcome,
            name=name,
        )

    if method == "dr":
        return _estimate_assignment_ate_dr(
            dataset,
            propensities,
            outcome=outcome,
            outcome_if_z0=outcome_if_z0,
            outcome_if_z1=outcome_if_z1,
            name=name,
        )

    raise NotImplementedError(
        f"Assignment ATE method {method!r} is registered but not implemented."
    )


def _estimate_assignment_ate_ipw(
    dataset: ComplierDataset,
    propensities: NDArray[np.float64],
    *,
    outcome: FeatureSpec,
    name: str | None,
) -> AssignmentEffectEstimate:
    """Estimate assignment ATE by inverse-propensity weighting."""

    if isinstance(outcome, str) and outcome == "outcome" and dataset.outcome is None:
        raise ValueError("assignment_ate() requires an outcome or an explicit outcome feature.")

    label = name or (outcome if isinstance(outcome, str) else "assignment_ate")
    values = dataset.resolve_feature(outcome, name=str(label))
    z = dataset.instrument
    p = propensities

    assigned_mean = float(np.mean(z * values / p))
    unassigned_mean = float(np.mean((1.0 - z) * values / (1.0 - p)))
    effect_scores = z * values / p - (1.0 - z) * values / (1.0 - p)
    estimate = assigned_mean - unassigned_mean
    return AssignmentEffectEstimate(
        name=str(label),
        estimate=estimate,
        method="ipw",
        assigned_mean=assigned_mean,
        unassigned_mean=unassigned_mean,
        standard_error=_standard_error_from_influence(effect_scores - estimate),
    )


def _estimate_assignment_ate_dr(
    dataset: ComplierDataset,
    propensities: NDArray[np.float64],
    *,
    outcome: FeatureSpec,
    outcome_if_z0: FeatureSpec | None,
    outcome_if_z1: FeatureSpec | None,
    name: str | None,
) -> AssignmentEffectEstimate:
    """Estimate assignment ATE by augmented inverse-propensity weighting."""

    if outcome_if_z0 is None or outcome_if_z1 is None:
        raise ValueError("outcome_if_z0 and outcome_if_z1 must be supplied for method='dr'.")

    if isinstance(outcome, str) and outcome == "outcome" and dataset.outcome is None:
        raise ValueError("assignment_ate() requires an outcome or an explicit outcome feature.")

    label = name or (outcome if isinstance(outcome, str) else "assignment_ate")
    values = dataset.resolve_feature(outcome, name=str(label))
    mu0 = dataset.resolve_feature(outcome_if_z0, name="outcome_if_z0")
    mu1 = dataset.resolve_feature(outcome_if_z1, name="outcome_if_z1")
    z = dataset.instrument
    p = propensities

    assigned_mean = float(np.mean(mu1 + z * (values - mu1) / p))
    unassigned_mean = float(np.mean(mu0 + (1.0 - z) * (values - mu0) / (1.0 - p)))
    effect_scores = (
        (mu1 - mu0)
        + z * (values - mu1) / p
        - (1.0 - z) * (values - mu0) / (1.0 - p)
    )
    estimate = assigned_mean - unassigned_mean
    return AssignmentEffectEstimate(
        name=str(label),
        estimate=estimate,
        method="dr",
        assigned_mean=assigned_mean,
        unassigned_mean=unassigned_mean,
        standard_error=_standard_error_from_influence(effect_scores - estimate),
    )


class ComplierResult:
    """High-level interface to complier descriptive functionals.

    The fitted object stores a common score representation, so descriptive
    functionals are just different feature maps `f(X)`.
    """

    def __init__(self, dataset: ComplierDataset, backend_result: BackendResult) -> None:
        self.dataset = dataset
        self.backend = backend_result.backend
        self.raw_scores = backend_result.raw_scores
        self.scaled_scores = backend_result.scaled_scores
        self.propensities = backend_result.propensities
        self.complier_share = backend_result.complier_share
        self.diagnostics = backend_result.diagnostics
        self.treated_if_z0 = backend_result.treated_if_z0
        self.treated_if_z1 = backend_result.treated_if_z1

    def _resolve(self, feature: FeatureSpec, *, name: str | None = None) -> NDArray[np.float64]:
        return self.dataset.resolve_feature(feature, name=name)

    def _standard_error_for_moment(
        self,
        values: NDArray[np.float64],
        estimate: float,
    ) -> float:
        influence_values = self.raw_scores * (values - estimate) / self.complier_share
        return _standard_error_from_influence(influence_values)

    def moment(self, feature: FeatureSpec, *, name: str | None = None) -> ScalarEstimate:
        """Estimate `E[f(X) | complier]` for a user-supplied feature map."""

        label = name or (feature if isinstance(feature, str) else "moment")
        values = self._resolve(feature, name=label)
        numerator = float(np.mean(self.raw_scores * values))
        estimate = numerator / self.complier_share
        return ScalarEstimate(
            name=str(label),
            estimate=estimate,
            backend=self.backend,
            complier_share=self.complier_share,
            numerator=numerator,
            denominator=self.complier_share,
            standard_error=self._standard_error_for_moment(values, estimate),
        )

    def mean(self, feature: FeatureSpec, *, name: str | None = None) -> ScalarEstimate:
        """Alias for :meth:`moment` to make the API explicit for common use."""

        return self.moment(feature, name=name)

    def share(self, feature: FeatureSpec, *, name: str | None = None) -> ScalarEstimate:
        """Estimate the complier share of a subgroup indicator.

        The feature must resolve to a binary 0/1 array or a boolean array.
        """

        label = name or (feature if isinstance(feature, str) else "share")
        values = self._resolve(feature, name=label)
        if values.dtype == bool:
            values = values.astype(float)

        rounded = np.round(values)
        if not np.allclose(values, rounded) or not np.all(np.isin(rounded, (0.0, 1.0))):
            raise ValueError("share() requires a binary or boolean feature map.")
        return self.moment(rounded, name=str(label))

    def variance(self, feature: FeatureSpec, *, name: str | None = None) -> ScalarEstimate:
        """Estimate the complier variance of a scalar feature."""

        label = name or (feature if isinstance(feature, str) else "variance")
        values = self._resolve(feature, name=label)
        mean_estimate = self.moment(values, name=f"{label}_mean")
        second_moment = self.moment(values**2, name=f"{label}_second_moment")
        variance = float(second_moment.estimate - mean_estimate.estimate**2)
        influence_values = (
            self.raw_scores
            * ((values - mean_estimate.estimate) ** 2 - variance)
            / self.complier_share
        )
        return ScalarEstimate(
            name=str(label),
            estimate=variance,
            backend=self.backend,
            complier_share=self.complier_share,
            standard_error=_standard_error_from_influence(influence_values),
        )

    def cdf(
        self,
        feature: FeatureSpec,
        *,
        grid: NDArray[np.float64] | list[float],
        name: str | None = None,
    ) -> DistributionEstimate:
        """Estimate the empirical complier CDF of a scalar feature on a grid."""

        label = name or (feature if isinstance(feature, str) else "cdf")
        values = self._resolve(feature, name=label)
        support = np.asarray(grid, dtype=float)
        if support.ndim != 1:
            raise ValueError("grid must be one-dimensional.")

        estimates = [
            self.share(values <= threshold, name=f"{label}_leq_{threshold:g}")
            for threshold in support
        ]
        cdf_values = np.array([estimate.estimate for estimate in estimates], dtype=float)
        cdf_values = np.clip(np.maximum.accumulate(cdf_values), 0.0, 1.0)
        standard_errors = np.array(
            [estimate.standard_error for estimate in estimates],
            dtype=float,
        )
        return DistributionEstimate(
            name=str(label),
            grid=support,
            values=cdf_values,
            backend=self.backend,
            complier_share=self.complier_share,
            standard_errors=standard_errors,
        )

    def assignment_ate(
        self,
        outcome: FeatureSpec = "outcome",
        *,
        method: str = "ipw",
        outcome_model: str = "constant",
        covariate_names: list[str] | None = None,
        outcome_if_z0: FeatureSpec | None = None,
        outcome_if_z1: FeatureSpec | None = None,
        name: str | None = None,
    ) -> AssignmentEffectEstimate:
        """Estimate the average effect of instrument assignment on an outcome.

        Supported methods:

        - `"ipw"`: inverse-propensity weighting using `P(Z=1 | X)`.
        - `"dr"`: augmented inverse-propensity weighting using outcome
          regressions for `E[Y | Z=z, X]`.
        """

        if method == "dr" and outcome_if_z0 is None and outcome_if_z1 is None:
            outcome_if_z0, outcome_if_z1 = estimate_outcome_responses(
                self.dataset,
                outcome=outcome,
                model=outcome_model,
                covariate_names=covariate_names,
            )

        return estimate_assignment_ate(
            self.dataset,
            self.propensities,
            outcome=outcome,
            method=method,
            outcome_if_z0=outcome_if_z0,
            outcome_if_z1=outcome_if_z1,
            name=name,
        )

    def summarize_covariates(self, names: list[str] | None = None) -> dict[str, dict[str, float]]:
        """Return a compact mean/variance summary for stored covariates."""

        summary: dict[str, dict[str, float]] = {}
        selected_names = names or self.dataset.covariate_names()
        for name in selected_names:
            values = self.dataset.covariates[name]
            entry = {
                "mean": self.mean(values, name=name).estimate,
                "variance": self.variance(values, name=name).estimate,
            }
            rounded = np.round(values)
            if np.allclose(values, rounded) and np.all(np.isin(rounded, (0.0, 1.0))):
                entry["share"] = self.share(values, name=name).estimate
            summary[name] = entry
        return summary
