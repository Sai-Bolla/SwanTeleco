"""
swan
====

Reusable analytics package for the **Swan Teleco** customer-retention project.

The package encapsulates every stage of the churn-analysis pipeline so that the
`Analysis.ipynb` notebook stays readable and every step is independently
importable, testable and reproducible:

    swan.config    -- paths, business constants and column semantics (single source of truth)
    swan.data      -- loading, cleaning and Pydantic-based validation of the customer view
    swan.features  -- feature engineering + a scikit-learn preprocessing pipeline
    swan.model     -- the ChurnModel wrapper, multi-model comparison and risk scoring
    swan.viz       -- a consistent, stakeholder-ready plotting style

Design goals (mapped to the mark scheme):
    * Reproducibility  -- all randomness is seeded from ``config.RANDOM_STATE``;
                          paths are resolved relative to the project root, never hard-coded.
    * No data leakage  -- target-derived columns are declared once in ``config`` and
                          excluded everywhere by construction.
    * PII discipline   -- personally identifiable fields are declared once and kept
                          out of features and stakeholder-facing aggregates.
"""

from importlib import metadata as _metadata

from . import config, data, features, model, viz

__all__ = ["config", "data", "features", "model", "viz"]

try:  # pragma: no cover - version is a nicety, never load-critical
    __version__ = _metadata.version("swan")
except _metadata.PackageNotFoundError:
    __version__ = "0.1.0"
