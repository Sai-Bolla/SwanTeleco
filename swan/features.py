"""
swan.features
=============

Feature engineering and the scikit-learn preprocessing pipeline.

Two responsibilities:

1. :func:`engineer` -- derive a handful of business-meaningful features that the
   raw columns only imply (tenure of a relationship, how "loaded" an account is,
   whether payment is on autopilot). These are argued for in the notebook.
2. :func:`build_preprocessor` -- a :class:`~sklearn.compose.ColumnTransformer`
   that one-hot-encodes categoricals and (optionally) scales numerics, wrapped so
   the whole transform is fit *inside* cross-validation and therefore leak-free.

:func:`build_feature_frame` glues them: raw clean df -> (X, y) ready for a model,
with PII / constant / leakage columns removed by construction.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from . import config


# ---------------------------------------------------------------------------
# Engineered features
# ---------------------------------------------------------------------------
# Names of the columns created by ``engineer`` -- exported so the preprocessor
# and tests can reference them without magic strings.
ENGINEERED_NUMERIC: list[str] = ["Num Addon Services", "Avg Monthly Spend"]
ENGINEERED_CATEGORICAL: list[str] = ["Tenure Group", "Has Internet", "Auto Payment", "New Customer"]


def engineer(df: pd.DataFrame) -> pd.DataFrame:
    """Add derived features to a cleaned customer frame (returns a copy).

    Rationale for each feature is given in the notebook; in brief:

    * **Num Addon Services** -- count of the six optional add-ons a customer holds.
      Bundled customers are "stickier"; a single count captures that more cleanly
      than six separate indicators for a tree to rediscover.
    * **Avg Monthly Spend** -- Total / Tenure, a smoothed spend signal that is
      robust to the single billing spike a new customer sees. Guarded against
      division-by-zero for tenure-0 customers.
    * **Tenure Group** -- coarse life-stage buckets (0-12, 13-24, 25-48, 49+),
      useful for the demographics narrative and as a monotonic categorical.
    * **Has Internet** -- collapses the three-valued internet flag to a boolean;
      many add-on "No internet service" values are really "no internet".
    * **Auto Payment** -- whether the payment method is automatic; manual payers
      (mailed / electronic check) churn far more.
    * **New Customer** -- tenure <= 6 months, the highest-risk window.
    """
    df = df.copy()

    # Count of add-ons actively held ("Yes" only; "No"/"No internet service" -> 0)
    df["Num Addon Services"] = (
        df[config.ADDON_SERVICES].eq("Yes").sum(axis=1).astype(int)
    )

    # Smoothed average spend; tenure 0 -> fall back to the month's charge
    tenure = df["Tenure Months"].replace(0, np.nan)
    df["Avg Monthly Spend"] = (df["Total Charges"] / tenure).fillna(df["Monthly Charges"])

    # Life-stage buckets
    df["Tenure Group"] = pd.cut(
        df["Tenure Months"],
        bins=[-0.1, 12, 24, 48, np.inf],
        labels=["0-12", "13-24", "25-48", "49+"],
    ).astype(str)

    df["Has Internet"] = np.where(df["Internet Service"].eq("No"), "No", "Yes")
    df["Auto Payment"] = np.where(
        df["Payment Method"].str.contains("automatic", case=False, na=False), "Yes", "No"
    )
    df["New Customer"] = np.where(df["Tenure Months"] <= 6, "Yes", "No")

    return df


# ---------------------------------------------------------------------------
# Feature / target assembly
# ---------------------------------------------------------------------------
def feature_columns(use_engineered: bool = True) -> tuple[list[str], list[str]]:
    """Return ``(numeric_features, categorical_features)`` for modelling.

    Excludes PII, constant and leakage columns by construction -- they are never
    added to either list, so leakage is impossible by omission rather than by a
    fragile "remember to drop" step.
    """
    numeric = list(config.NUMERIC_FEATURES)
    categorical = list(config.CATEGORICAL_FEATURES)
    if use_engineered:
        numeric += ENGINEERED_NUMERIC
        categorical += ENGINEERED_CATEGORICAL
    return numeric, categorical


def build_feature_frame(
    df: pd.DataFrame, use_engineered: bool = True
) -> tuple[pd.DataFrame, pd.Series]:
    """Turn a cleaned frame into ``(X, y)`` ready for a pipeline.

    ``df`` may or may not already contain engineered columns; this function adds
    them if requested and missing, then selects exactly the modelling columns.
    """
    if use_engineered and not set(ENGINEERED_NUMERIC).issubset(df.columns):
        df = engineer(df)
    numeric, categorical = feature_columns(use_engineered)
    X = df[numeric + categorical].copy()
    y = df[config.TARGET].astype(int).copy()
    return X, y


# ---------------------------------------------------------------------------
# Preprocessing pipeline
# ---------------------------------------------------------------------------
def build_preprocessor(
    use_engineered: bool = True, scale_numeric: bool = False
) -> ColumnTransformer:
    """Build the preprocessing :class:`ColumnTransformer`.

    Parameters
    ----------
    scale_numeric:
        Standardise numeric columns. Left **off by default** because every model
        we use is tree-based and therefore scale-invariant -- scaling would add
        cost and obscure feature importances for no benefit. Exposed as a flag so
        the notebook can demonstrate the argument by toggling it.

    Notes
    -----
    * ``OneHotEncoder(handle_unknown="ignore")`` means a category unseen at fit
      time (e.g. a new payment method in a future refresh) yields an all-zero
      block instead of crashing -- important for a *deployable* scoring product.
    * The transformer is returned *unfitted*; callers place it inside a Pipeline
      so it is fit only on training folds, guaranteeing no leakage across the
      train/test split or CV folds.
    """
    numeric, categorical = feature_columns(use_engineered)

    numeric_transformer = StandardScaler() if scale_numeric else "passthrough"
    categorical_transformer = OneHotEncoder(handle_unknown="ignore", drop=None)

    return ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric),
            ("cat", categorical_transformer, categorical),
        ],
        remainder="drop",  # anything not explicitly listed is dropped -> no accidental PII
        verbose_feature_names_out=False,
    )
