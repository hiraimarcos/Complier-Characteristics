"""Post-fit result objects and descriptive functionals."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from .data import ComplierDataset, FeatureSpec
from .diagnostics import ComplierDiagnostics
from .estimators import BackendResult


@dataclass(frozen=True)
class ScalarEstimate:
    """Scalar estimate returned by `mean`, `share`, `variance`, or `moment`."""

    name: str
    estimate: float
    backend: str
    complier_share: float
    numerator: float | None = None
    denominator: float | None = None


@dataclass(frozen=True)
class DistributionEstimate:
    """Distributional estimate returned by `cdf`."""

    name: str
    grid: NDArray[np.float64]
    values: NDArray[np.float64]
    backend: str
    complier_share: float


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

    def moment(self, feature: FeatureSpec, *, name: str | None = None) -> ScalarEstimate:
        """Estimate `E[f(X) | complier]` for a user-supplied feature map."""

        label = name or (feature if isinstance(feature, str) else "moment")
        values = self._resolve(feature, name=label)
        numerator = float(np.mean(self.raw_scores * values))
        estimate = float(np.mean(self.scaled_scores * values))
        return ScalarEstimate(
            name=str(label),
            estimate=estimate,
            backend=self.backend,
            complier_share=self.complier_share,
            numerator=numerator,
            denominator=self.complier_share,
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
        return ScalarEstimate(
            name=str(label),
            estimate=variance,
            backend=self.backend,
            complier_share=self.complier_share,
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

        cdf_values = np.array(
            [self.share(values <= threshold, name=f"{label}_leq_{threshold:g}").estimate for threshold in support],
            dtype=float,
        )
        cdf_values = np.clip(np.maximum.accumulate(cdf_values), 0.0, 1.0)
        return DistributionEstimate(
            name=str(label),
            grid=support,
            values=cdf_values,
            backend=self.backend,
            complier_share=self.complier_share,
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
