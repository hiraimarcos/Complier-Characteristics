# `complier-characteristics`

`complier-characteristics` is a small Python package for estimating descriptive
statistics of the complier population in the standard binary-instrument,
binary-treatment framework.

The first version is intentionally narrow and readable:

- binary instrument `Z`
- binary treatment `D`
- user-supplied feature maps `f(X)`
- Abadie-style `kappa` weighting as the default backend
- a first-stage plug-in backend that avoids division by instrument
  propensities for average complier characteristics
- an optional doubly robust backend for average complier characteristics
- helpers for means, variances, subgroup shares, and empirical CDFs
- IPW estimators for `E[Y_0 | D_1 > D_0]` and `E[Y_1 | D_1 > D_0]`
- IPW and doubly robust estimators for the average effect of instrument
  assignment
- analytical fixed-nuisance standard errors for scalar estimates

The package is designed to make the literature legible in code. Most public IV
packages focus on treatment effects. This one focuses on the latent complier
population itself.

## Literature Basis

The implementation follows the scope laid out in
[docs/literature_review.tex](docs/literature_review.tex) and is anchored in a
small subset of the local source PDFs:

- Abadie (2003):
  [docs/papers/abadie_2003_semiparametric_iv_treatment_response.pdf](docs/papers/abadie_2003_semiparametric_iv_treatment_response.pdf)
- Singh and Sun (2024):
  [docs/papers/singh_sun_2024_double_robustness_complier_parameters_arxiv.pdf](docs/papers/singh_sun_2024_double_robustness_complier_parameters_arxiv.pdf)
- Sloczynski, Uysal, and Wooldridge (2025):
  [docs/papers/sloczynski_uysal_wooldridge_2025_abadie_kappa.pdf](docs/papers/sloczynski_uysal_wooldridge_2025_abadie_kappa.pdf)
- Froelich (working-paper version used for scope extension):
  [docs/papers/froelich_2002_nonparametric_iv_late_covariates_iza_dp588.pdf](docs/papers/froelich_2002_nonparametric_iv_late_covariates_iza_dp588.pdf)

The resulting design choices are:

1. treat complier moments as the primitive target
2. use Abadie's `kappa` representation as the default estimator
3. include the first-stage plug-in representation
   `E[D | Z=1, X] - E[D | Z=0, X]` as a propensity-free alternative for
   average complier characteristics
4. keep nuisance estimation modular so covariate-adjusted and doubly robust
   estimators can reuse the same high-level API
5. include diagnostics because weight normalization and weak first stages are
   implementation issues, not afterthoughts

## Installation

```bash
python3 -m pip install -e .
```

The package depends on `numpy` and `statsmodels`.

## Quick Start

```python
import numpy as np

from complier_characteristics import ComplierDataset, ComplierEstimator

rng = np.random.default_rng(42)
n = 4000
x = rng.normal(size=n)
z = rng.binomial(1, 0.5, size=n)
complier = rng.binomial(1, 1 / (1 + np.exp(-(0.2 + 0.8 * x))), size=n)
d = z * complier
y = 1.0 + 0.2 * d + 0.5 * x + rng.normal(scale=0.25, size=n)

dataset = ComplierDataset.from_arrays(
    instrument=z,
    treatment=d,
    outcome=y,
    covariates={
        "x": x,
        "high_x": (x > 0).astype(float),
    },
)

estimator = ComplierEstimator(
    backend="abadie",
    normalize=True,
    propensity_model="constant",
)
result = estimator.fit(dataset)

mean_x = result.mean("x")
share_high_x = result.share("high_x")
cdf_x = result.cdf("x", grid=np.linspace(-2.0, 2.0, 5))
untreated_mean = result.untreated_outcome_mean()
treated_mean = result.treated_outcome_mean()
assignment_effect = result.assignment_ate(method="ipw")

print(mean_x.estimate)
print(mean_x.standard_error)
print(share_high_x.estimate)
print(cdf_x.values)
print(cdf_x.standard_errors)
print(untreated_mean.estimate)
print(treated_mean.estimate)
print(assignment_effect.estimate)
print(assignment_effect.standard_error)
print(result.diagnostics.to_dict())
```

## Main Objects

### `ComplierDataset`

`ComplierDataset` validates and stores the observed sample:

- `instrument`: binary `Z`
- `treatment`: binary `D`
- `covariates`: mapping from names to one-dimensional numeric arrays
- `outcome`: optional one-dimensional numeric array

The package focuses on complier characteristics, so the most common usage is to
store baseline covariates in `covariates` and then evaluate functions of those
covariates.

### `ComplierEstimator`

`ComplierEstimator` is the high-level API.

Supported backends:

- `backend="abadie"`:
  computes Abadie-style `kappa` scores and uses them as complier-membership
  weights
- `backend="plugin"`:
  computes first-stage plug-in scores
  `E[D | Z=1, X] - E[D | Z=0, X]`; this avoids inverse-propensity division but
  relies on the treatment-response regressions
- `backend="dr"`:
  computes an augmented score for average complier characteristics using
  estimated nuisance functions for `P(Z=1 | X)` and `E[D | Z=z, X]`

Supported nuisance strategies:

- `propensity_model="constant"`:
  uses the sample instrument rate
- `propensity_model="linear"`:
  fits a clipped OLS linear probability model on the supplied covariates
- `propensity_model="logit"`:
  fits a statsmodels logistic regression on the supplied covariates
- `propensity_model="probit"`:
  fits a statsmodels probit regression on the supplied covariates
- `treatment_model="constant"`:
  uses within-instrument sample means for `E[D | Z=z, X]`
- `treatment_model="linear"`:
  fits separate clipped OLS linear probability models in the `Z=0` and `Z=1`
  strata
- `treatment_model="logit"`:
  fits separate statsmodels logistic regressions in the `Z=0` and `Z=1` strata
- `treatment_model="probit"`:
  fits separate statsmodels probit regressions in the `Z=0` and `Z=1` strata
- `assignment_outcome_model="constant"`:
  uses within-instrument sample means for `E[Y | Z=z, X]`
- `assignment_outcome_model="linear"`:
  fits separate least squares regressions in the `Z=0` and `Z=1` strata

The plug-in backend is useful when extreme values of `P(Z=1 | X)` make
Abadie-style weights numerically unstable:

```python
plugin_result = ComplierEstimator(
    backend="plugin",
    treatment_model="linear",
    covariate_names=["x"],
).fit(dataset)
```

This backend estimates average complier characteristics as

```text
E[f(X) | complier]
  = E[f(X) {m1(X) - m0(X)}] / E[m1(X) - m0(X)]
```

where `m1(X) = E[D | Z=1, X]` and `m0(X) = E[D | Z=0, X]`. It does not estimate
instrument propensities unless `propensity_scores` are supplied to `fit()`.

The default is deliberately conservative:

```python
ComplierEstimator(
    backend="abadie",
    normalize=True,
    propensity_model="constant",
    treatment_model="constant",
    assignment_outcome_model="constant",
)
```

The same estimator can compute assignment effects directly:

```python
assignment_effect = estimator.assignment_ate(dataset, method="ipw")
assignment_effect_dr = estimator.assignment_ate(dataset, method="dr")
```

This direct path does not require a nonzero first stage for realized treatment
take-up. The doubly robust method uses both `P(Z=1 | X)` and outcome
regressions for `E[Y | Z=z, X]`.

## Estimands Exposed by `ComplierResult`

The fitted result object exposes common descriptive functionals:

- `mean(feature)`
- `variance(feature)`
- `share(feature)`
- `cdf(feature, grid)`
- `moment(feature)`
- `potential_outcome_mean(treatment_value, outcome="outcome")`
- `untreated_outcome_mean(outcome="outcome")`
- `treated_outcome_mean(outcome="outcome")`
- `assignment_ate(outcome="outcome", method="ipw")`
- `assignment_ate(outcome="outcome", method="dr")`
- `summarize_covariates(names=None)`

Feature inputs can be:

- a covariate name, for example `"age"`
- a one-dimensional array of length `n`
- a callable that accepts the fitted `ComplierDataset`

Examples:

```python
result.mean("x")
result.share(lambda data: data.covariates["x"] > 0)
result.mean(np.square(dataset.covariates["x"]))
result.untreated_outcome_mean()
result.treated_outcome_mean()
result.potential_outcome_mean(0)
result.assignment_ate(method="ipw")
result.assignment_ate(method="dr")
```

`untreated_outcome_mean` estimates `E[Y_0 | D_1 > D_0]`, and
`treated_outcome_mean` estimates `E[Y_1 | D_1 > D_0]`. These methods use
inverse-assignment weighted contrasts of observed untreated or treated outcomes
and accept the same outcome feature inputs as `assignment_ate`. If the fitted
object came from `backend="plugin"`, these post-fit outcome methods require
explicit `propensity_scores` at fit time because the plug-in backend does not
estimate instrument propensities.

`assignment_ate` estimates the average effect of instrument assignment `Z` on an
outcome using propensity scores `P(Z=1 | X)`. The default outcome is the
optional `outcome` stored on `ComplierDataset`, and explicit one-dimensional
outcome arrays or feature maps are also accepted. The `dr` method augments IPW
with outcome regressions; by default it uses constant within-instrument outcome
means, and it can also use `outcome_model="linear"` or externally supplied
`outcome_if_z0` and `outcome_if_z1` arrays.

Scalar estimates include `standard_error`, computed from the analytical
influence function while treating fitted nuisance quantities as fixed. For
complier moments, this uses the ratio influence form. For assignment ATEs, this
uses the IPW or doubly robust observation-level score. Empirical CDF results
include pointwise `standard_errors`.

## Diagnostics

Every fit returns a `ComplierDiagnostics` object with:

- sample size
- instrument rate
- treatment rate
- first-stage difference in treatment rates
- estimated complier share
- min and max instrument propensities when the fit used propensities
- fraction of negative raw scores
- whether normalized weights were used

This reflects the implementation emphasis in the weighting literature: a v1
package should make weak first stages, overlap problems, and unstable weights
easy to see.

## What Is Implemented in Version 1

- binary-IV identification logic
- Abadie-style `kappa` weights
- first-stage plug-in scores for average complier characteristics
- a readable doubly robust score for average complier characteristics
- simple nuisance estimation with intercept-only, clipped OLS linear
  probability, logit, or probit models
- helpers for means, variances, subgroup shares, and empirical CDFs
- IPW and doubly robust estimators for the average effect of instrument
  assignment
- analytical fixed-nuisance standard errors for scalar estimates
- unit tests based on simulated data with known complier populations

## What Is Not Implemented Yet

- bootstrap inference
- nuisance-estimation uncertainty in analytical standard errors
- cross-fitting
- multiple instruments
- multi-valued treatments
- defiers or monotonicity violations

Those are natural next steps, but they would make the first version much harder
to read.

## Running the Tests

```bash
python3 -m unittest discover -s tests -p "test_*.py"
```

## Package Layout

```text
src/complier_characteristics/
    __init__.py
    api.py
    data.py
    diagnostics.py
    estimators.py
    nuisance.py
    results.py
tests/
    test_api_results.py
    test_data.py
    test_estimators.py
    test_nuisance.py
```

## Reading Guide for the Code

If you want to understand the package from top to bottom, the best order is:

1. `src/complier_characteristics/data.py`
2. `src/complier_characteristics/nuisance.py`
3. `src/complier_characteristics/estimators.py`
4. `src/complier_characteristics/results.py`
5. `src/complier_characteristics/api.py`

That sequence mirrors the estimation pipeline.
