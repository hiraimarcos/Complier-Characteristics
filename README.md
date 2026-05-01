# `complier-characteristics`

`complier-characteristics` is a small Python package for estimating descriptive
statistics of the complier population in the standard binary-instrument,
binary-treatment framework.

The first version is intentionally narrow and readable:

- binary instrument `Z`
- binary treatment `D`
- user-supplied feature maps `f(X)`
- Abadie-style `kappa` weighting as the default backend
- an optional doubly robust backend for average complier characteristics
- helpers for means, variances, subgroup shares, and empirical CDFs

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
3. keep nuisance estimation modular so covariate-adjusted and doubly robust
   estimators can reuse the same high-level API
4. include diagnostics because weight normalization and weak first stages are
   implementation issues, not afterthoughts

## Installation

```bash
python3 -m pip install -e .
```

The package only depends on `numpy`.

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

dataset = ComplierDataset.from_arrays(
    instrument=z,
    treatment=d,
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

print(mean_x.estimate)
print(share_high_x.estimate)
print(cdf_x.values)
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
- `backend="dr"`:
  computes an augmented score for average complier characteristics using
  estimated nuisance functions for `P(Z=1 | X)` and `E[D | Z=z, X]`

Supported nuisance strategies:

- `propensity_model="constant"`:
  uses the sample instrument rate
- `propensity_model="logit"`:
  fits a simple logistic regression on the supplied covariates
- `treatment_model="constant"`:
  uses within-instrument sample means for `E[D | Z=z, X]`
- `treatment_model="logit"`:
  fits separate logistic regressions in the `Z=0` and `Z=1` strata

The default is deliberately conservative:

```python
ComplierEstimator(
    backend="abadie",
    normalize=True,
    propensity_model="constant",
    treatment_model="constant",
)
```

## Estimands Exposed by `ComplierResult`

The fitted result object exposes common descriptive functionals:

- `mean(feature)`
- `variance(feature)`
- `share(feature)`
- `cdf(feature, grid)`
- `moment(feature)`
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
```

## Diagnostics

Every fit returns a `ComplierDiagnostics` object with:

- sample size
- instrument rate
- treatment rate
- first-stage difference in treatment rates
- estimated complier share
- min and max instrument propensities
- fraction of negative raw scores
- whether normalized weights were used

This reflects the implementation emphasis in the weighting literature: a v1
package should make weak first stages, overlap problems, and unstable weights
easy to see.

## What Is Implemented in Version 1

- binary-IV identification logic
- Abadie-style `kappa` weights
- a readable doubly robust score for average complier characteristics
- simple nuisance estimation with intercept-only or logit models
- helpers for means, variances, subgroup shares, and empirical CDFs
- unit tests based on simulated data with known complier populations

## What Is Not Implemented Yet

- standard errors or bootstrap inference
- cross-fitting
- multiple instruments
- multi-valued treatments
- defiers or monotonicity violations
- treatment-effect functionals that depend on post-treatment outcomes

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
    test_estimators.py
```

## Reading Guide for the Code

If you want to understand the package from top to bottom, the best order is:

1. `src/complier_characteristics/data.py`
2. `src/complier_characteristics/nuisance.py`
3. `src/complier_characteristics/estimators.py`
4. `src/complier_characteristics/results.py`
5. `src/complier_characteristics/api.py`

That sequence mirrors the estimation pipeline.
