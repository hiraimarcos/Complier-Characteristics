"""Diagnostic summaries returned with every fitted estimator."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ComplierDiagnostics:
    """Compact diagnostic summary for a fitted complier-characteristics model."""

    n_obs: int
    backend: str
    normalized: bool
    instrument_rate: float
    treatment_rate: float
    first_stage: float
    complier_share: float
    min_propensity: float | None
    max_propensity: float | None
    negative_score_fraction: float
    score_mean: float

    def to_dict(self) -> dict[str, float | int | str | bool | None]:
        """Return a plain-Python representation for logging or debugging."""

        return {
            "n_obs": self.n_obs,
            "backend": self.backend,
            "normalized": self.normalized,
            "instrument_rate": self.instrument_rate,
            "treatment_rate": self.treatment_rate,
            "first_stage": self.first_stage,
            "complier_share": self.complier_share,
            "min_propensity": self.min_propensity,
            "max_propensity": self.max_propensity,
            "negative_score_fraction": self.negative_score_fraction,
            "score_mean": self.score_mean,
        }
