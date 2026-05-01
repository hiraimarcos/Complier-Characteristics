"""Public package exports for `complier_characteristics`.

The package is intentionally small. The main entry points are:

- :class:`ComplierDataset` for validated input data
- :class:`ComplierEstimator` for fitting a backend
- :class:`ComplierResult` for post-fit descriptive functionals
"""

from .api import ComplierEstimator
from .data import ComplierDataset
from .diagnostics import ComplierDiagnostics
from .results import ComplierResult, DistributionEstimate, ScalarEstimate

__all__ = [
    "ComplierDataset",
    "ComplierDiagnostics",
    "ComplierEstimator",
    "ComplierResult",
    "DistributionEstimate",
    "ScalarEstimate",
]

__version__ = "0.1.0"
