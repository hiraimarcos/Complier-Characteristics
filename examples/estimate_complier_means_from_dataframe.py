"""Estimate complier means for many dataframe columns.

Run from the project root after installing the package:

    python examples/estimate_complier_means_from_dataframe.py

In your own analysis, replace ``make_example_dataframe()`` with something like
``pd.read_csv(...)`` and update the column lists in ``main()``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from complier_characteristics import ComplierDataset, ComplierEstimator, ComplierResult
from complier_characteristics.nuisance import estimate_propensity_scores


def estimate_complier_means_from_dataframe(
    df: pd.DataFrame,
    *,
    instrument_col: str,
    treatment_col: str,
    mean_columns: list[str],
    outcome_col: str | None = None,
    nuisance_columns: list[str] | None = None,
    propensity_model: str = "constant",
    treatment_model: str = "constant",
    backend: str = "abadie",
    clip: float = 1e-6,
) -> tuple[pd.DataFrame, ComplierResult]:
    """Fit once and estimate complier means for every column in mean_columns.

    ``instrument_col`` and ``treatment_col`` must be binary 0/1 columns. All
    ``mean_columns`` and ``nuisance_columns`` must be numeric and non-missing
    after the complete-case filter below. ``clip`` bounds nuisance probabilities
    away from 0 and 1.

    For ``backend="plugin"``, set ``treatment_model`` to ``"logit"`` or
    ``"probit"`` and pass ``nuisance_columns`` to estimate covariate-varying
    first-stage functions ``E[D | Z=z, X]``. If an outcome column is supplied,
    this helper also estimates and stores instrument propensities so the
    returned result can still compute IPW-based outcome summaries. Those
    propensities are not used for the plug-in complier means.
    """

    nuisance_columns = nuisance_columns or []
    covariate_columns = _ordered_unique([*mean_columns, *nuisance_columns])
    required_columns = _ordered_unique(
        [
            instrument_col,
            treatment_col,
            *([] if outcome_col is None else [outcome_col]),
            *covariate_columns,
        ]
    )

    missing_columns = [column for column in required_columns if column not in df.columns]
    if missing_columns:
        raise KeyError(f"Missing required dataframe columns: {missing_columns}")

    complete_df = df.loc[:, required_columns].dropna()
    if complete_df.empty:
        raise ValueError("No complete observations remain after dropping missing values.")

    dataset = ComplierDataset.from_arrays(
        instrument=complete_df[instrument_col].to_numpy(),
        treatment=complete_df[treatment_col].to_numpy(),
        outcome=None if outcome_col is None else complete_df[outcome_col].to_numpy(),
        covariates={
            column: complete_df[column].to_numpy()
            for column in covariate_columns
        },
    )

    estimator = ComplierEstimator(
        backend=backend,
        normalize=True,
        propensity_model=propensity_model,
        treatment_model=treatment_model,
        covariate_names=nuisance_columns or None,
        clip=clip,
    )

    propensity_scores = None
    if backend == "plugin" and outcome_col is not None:
        propensity_scores = estimate_propensity_scores(
            dataset,
            model=propensity_model,
            covariate_names=nuisance_columns or None,
            clip=clip,
        )

    result = estimator.fit(dataset, propensity_scores=propensity_scores)

    rows = []
    for column in mean_columns:
        estimate = result.mean(column)
        rows.append(
            {
                "variable": column,
                "sample_mean": float(complete_df[column].mean()),
                "complier_mean": estimate.estimate,
                "complier_share": estimate.complier_share,
                "n_complete": dataset.n_obs,
            }
        )

    return pd.DataFrame(rows), result


def make_example_dataframe(n_obs: int = 5_000, seed: int = 20240506) -> pd.DataFrame:
    """Create a toy binary-IV dataframe with baseline covariates."""

    rng = np.random.default_rng(seed)
    age = np.clip(rng.normal(loc=35.0, scale=9.0, size=n_obs), 18.0, 65.0)
    female = rng.binomial(1, 0.55, size=n_obs)
    pre_score = rng.normal(size=n_obs)
    neighborhood_index = rng.normal(size=n_obs)
    baseline_earnings_k = (
        28.0
        + 4.0 * pre_score
        + 2.0 * neighborhood_index
        + rng.normal(scale=5.0, size=n_obs)
    )

    instrument_prob = _logistic(
        -0.10 + 0.35 * female - 0.25 * neighborhood_index + 0.20 * pre_score
    )
    z = rng.binomial(1, instrument_prob, size=n_obs)

    always_taker_prob = _logistic(-2.60 + 0.20 * female - 0.10 * pre_score)
    complier_prob = _logistic(
        -0.20 + 0.70 * pre_score + 0.40 * neighborhood_index - 0.25 * female
    )
    always_taker = rng.binomial(1, always_taker_prob, size=n_obs)
    complier = rng.binomial(1, complier_prob, size=n_obs)
    d = np.maximum(always_taker, z * complier)

    outcome = (
        5.0
        + 1.5 * d
        + 0.4 * pre_score
        + 0.2 * neighborhood_index
        + rng.normal(scale=1.0, size=n_obs)
    )

    return pd.DataFrame(
        {
            "z": z,
            "d": d,
            "outcome": outcome,
            "age": age,
            "age_centered": age - age.mean(),
            "female": female,
            "pre_score": pre_score,
            "high_pre_score": (pre_score > 0.0).astype(float),
            "neighborhood_index": neighborhood_index,
            "baseline_earnings_k": baseline_earnings_k,
        }
    )


def _ordered_unique(values: list[str]) -> list[str]:
    """Return unique strings while preserving first occurrence order."""

    return list(dict.fromkeys(values))


def _logistic(values: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-values))


def _masked_mean(values: np.ndarray, mask: np.ndarray) -> float:
    if not np.any(mask):
        return float("nan")
    return float(values[mask].mean())


def main() -> None:
    df = make_example_dataframe()

    mean_columns = [
        "age",
        "female",
        "pre_score",
        "high_pre_score",
        "neighborhood_index",
        "baseline_earnings_k",
    ]
    nuisance_columns = [
        "age_centered",
        "female",
        "pre_score",
        "neighborhood_index",
        "baseline_earnings_k",
    ]

    means, result = estimate_complier_means_from_dataframe(
        df,
        instrument_col="z",
        treatment_col="d",
        outcome_col="outcome",
        mean_columns=mean_columns,
        nuisance_columns=nuisance_columns,
        backend="plugin",
        propensity_model="logit",
        treatment_model="logit",
    )

    if result.dataset.outcome is None:
        raise ValueError("The example outcome summary requires an outcome column.")

    outcome = result.dataset.outcome
    treatment = result.dataset.treatment
    instrument = result.dataset.instrument
    untreated_mean = result.untreated_outcome_mean()
    treated_mean = result.treated_outcome_mean()
    outcome_rows = [
        {
            "estimand": "Observed E[Y | D = 0]",
            "estimate": _masked_mean(outcome, treatment == 0.0),
            "standard_error": np.nan,
        },
        {
            "estimand": "Observed E[Y | D = 1]",
            "estimate": _masked_mean(outcome, treatment == 1.0),
            "standard_error": np.nan,
        },
        {
            "estimand": "Observed E[Y | Z = 0]",
            "estimate": _masked_mean(outcome, instrument == 0.0),
            "standard_error": np.nan,
        },
        {
            "estimand": "Observed E[Y | Z = 1]",
            "estimate": _masked_mean(outcome, instrument == 1.0),
            "standard_error": np.nan,
        },
    ]
    for d_value in [0, 1]:
        for z_value in [0, 1]:
            mask = (treatment == float(d_value)) & (instrument == float(z_value))
            outcome_rows.append(
                {
                    "estimand": f"Observed E[Y | D = {d_value}, Z = {z_value}]",
                    "estimate": _masked_mean(outcome, mask),
                    "standard_error": np.nan,
                }
            )

    outcome_rows.extend(
        [
            {
                "estimand": "E[Y_0 | D_1 > D_0]",
                "estimate": untreated_mean.estimate,
                "standard_error": untreated_mean.standard_error,
            },
            {
                "estimand": "E[Y_1 | D_1 > D_0]",
                "estimate": treated_mean.estimate,
                "standard_error": treated_mean.standard_error,
            },
        ]
    )
    outcome_means = pd.DataFrame(outcome_rows)

    print("Complier means")
    print(means.round(4).to_string(index=False))

    print("\nOutcome means")
    print(outcome_means.round(4).to_string(index=False))

    diagnostics = result.diagnostics.to_dict()
    print("\nDiagnostics")
    for key in [
        "n_obs",
        "instrument_rate",
        "treatment_rate",
        "first_stage",
        "complier_share",
        "negative_score_fraction",
    ]:
        print(f"{key}: {diagnostics[key]:.4f}")


if __name__ == "__main__":
    main()
