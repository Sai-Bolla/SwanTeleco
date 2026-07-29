"""
swan.model
==========

Modelling layer: a thin, well-behaved wrapper around a scikit-learn pipeline
plus helpers for model comparison, threshold-aware evaluation and the
business-facing risk score.

Public surface
--------------
    candidate_models()                 -> dict[str, (estimator, param_grid)]
    ChurnModel(estimator, ...)         -> fit / predict_proba / evaluate / risk_score
    compare_models(X, y, ...)          -> tidy cross-validated leaderboard
    risk_band(prob) / risk_score(prob) -> interpretable outputs for the CS team

All four classifiers requested in the brief (Decision Tree, Random Forest,
Extra Trees, Bagging) are covered. Everything is seeded from
``config.RANDOM_STATE`` for exact reproducibility.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import (
    BaggingClassifier,
    ExtraTreesClassifier,
    RandomForestClassifier,
)
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier

from . import config, features


# ---------------------------------------------------------------------------
# Candidate estimators + tuning grids
# ---------------------------------------------------------------------------
def candidate_models() -> dict[str, tuple[Any, dict[str, list]]]:
    """Return the four requested tree-based classifiers with compact tuning grids.

    ``class_weight="balanced"`` is set where supported because churn is the
    minority class (~26.5%); it tells the model that missing a churner is costlier
    than a false alarm without us having to resample the data.

    Grid keys are prefixed ``clf__`` so they address the classifier step inside
    the full preprocessing pipeline built in :func:`ChurnModel.fit`.
    """
    rs = config.RANDOM_STATE
    return {
        "Decision Tree": (
            DecisionTreeClassifier(random_state=rs, class_weight="balanced"),
            {
                "clf__max_depth": [4, 6, 8, None],
                "clf__min_samples_leaf": [1, 20, 50],
            },
        ),
        "Random Forest": (
            RandomForestClassifier(random_state=rs, class_weight="balanced", n_jobs=config.N_JOBS),
            {
                "clf__n_estimators": [300],
                "clf__max_depth": [8, 12, None],
                "clf__min_samples_leaf": [1, 5],
            },
        ),
        "Extra Trees": (
            ExtraTreesClassifier(random_state=rs, class_weight="balanced", n_jobs=config.N_JOBS),
            {
                "clf__n_estimators": [300],
                "clf__max_depth": [12, None],
                "clf__min_samples_leaf": [1, 5],
            },
        ),
        "Bagging": (
            BaggingClassifier(
                estimator=DecisionTreeClassifier(
                    random_state=rs, class_weight="balanced", max_depth=8
                ),
                random_state=rs,
                n_jobs=config.N_JOBS,
            ),
            {"clf__n_estimators": [100, 300]},
        ),
    }


# ---------------------------------------------------------------------------
# Risk scoring -- interpretable outputs for the customer-service team
# ---------------------------------------------------------------------------
def risk_score(prob: np.ndarray | float) -> np.ndarray | int:
    """Convert a churn probability in [0, 1] to a 0-100 integer risk score.

    A 0-100 score is far more intuitive on a live call than "0.63".
    """
    scaled = np.rint(np.asarray(prob, dtype=float) * 100).astype(int)
    return int(scaled) if np.ndim(prob) == 0 else scaled


def risk_band(prob: np.ndarray | float) -> np.ndarray | str:
    """Map a churn probability to a ``Low`` / ``Medium`` / ``High`` band."""
    bands = config.RISK_BANDS

    def _band(p: float) -> str:
        for name, (lo, hi) in bands.items():
            if lo <= p < hi:
                return name
        return "High"  # numerical safety net

    arr = np.asarray(prob, dtype=float)
    if arr.ndim == 0:
        return _band(float(arr))
    return np.array([_band(p) for p in arr])


# ---------------------------------------------------------------------------
# The model wrapper
# ---------------------------------------------------------------------------
@dataclass
class ChurnModel:
    """A fit-once, score-many wrapper around a preprocessing + classifier pipeline.

    Bundling the preprocessing *with* the estimator means the object is the whole
    contract: hand it a raw (cleaned) customer frame and it returns calibrated
    probabilities, risk scores and bands -- no risk of applying the wrong encoder
    at scoring time.

    Parameters
    ----------
    estimator:
        Any fitted-or-unfitted sklearn classifier exposing ``predict_proba``.
    use_engineered / scale_numeric:
        Forwarded to :func:`swan.features.build_preprocessor`.
    """

    estimator: Any
    use_engineered: bool = True
    scale_numeric: bool = False
    pipeline: Pipeline = field(init=False, default=None)

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "ChurnModel":
        """Fit the full pipeline (preprocessor + classifier) on training data."""
        pre = features.build_preprocessor(self.use_engineered, self.scale_numeric)
        self.pipeline = Pipeline([("pre", pre), ("clf", self.estimator)])
        self.pipeline.fit(X, y)
        return self

    def _check_fitted(self) -> None:
        if self.pipeline is None:
            raise RuntimeError("ChurnModel is not fitted yet; call .fit(X, y) first.")

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Probability of churn (positive class) for each row."""
        self._check_fitted()
        return self.pipeline.predict_proba(X)[:, 1]

    def predict(self, X: pd.DataFrame, threshold: float = 0.5) -> np.ndarray:
        """Hard class prediction at an explicit decision ``threshold``."""
        return (self.predict_proba(X) >= threshold).astype(int)

    def score_frame(self, X: pd.DataFrame) -> pd.DataFrame:
        """Return a tidy per-customer frame: probability, 0-100 score and band.

        The index of ``X`` is preserved so the caller can re-attach CustomerID.
        """
        proba = self.predict_proba(X)
        return pd.DataFrame(
            {
                "churn_probability": np.round(proba, 4),
                "risk_score": risk_score(proba),
                "risk_band": risk_band(proba),
            },
            index=X.index,
        )

    def evaluate(
        self, X: pd.DataFrame, y: pd.Series, threshold: float = 0.5
    ) -> dict[str, float]:
        """Compute the full metric suite on a held-out set.

        Threshold-independent metrics (ROC-AUC, PR-AUC) judge the *ranking*, which
        is what the mailer shortlist relies on; threshold-dependent metrics
        (precision/recall/F1) judge the default 0.5 decision.
        """
        self._check_fitted()
        proba = self.predict_proba(X)
        pred = (proba >= threshold).astype(int)
        return {
            "accuracy": accuracy_score(y, pred),
            "precision": precision_score(y, pred, zero_division=0),
            "recall": recall_score(y, pred, zero_division=0),
            "f1": f1_score(y, pred, zero_division=0),
            "roc_auc": roc_auc_score(y, proba),
            "pr_auc": average_precision_score(y, proba),
        }

    def confusion(
        self, X: pd.DataFrame, y: pd.Series, threshold: float = 0.5
    ) -> np.ndarray:
        """Confusion matrix at ``threshold`` (rows = actual, cols = predicted)."""
        return confusion_matrix(y, self.predict(X, threshold))

    def text_report(self, X: pd.DataFrame, y: pd.Series, threshold: float = 0.5) -> str:
        """Human-readable per-class precision/recall/F1 report."""
        return classification_report(
            y, self.predict(X, threshold), target_names=["Retained", "Churned"]
        )

    def feature_importances(self, top: int | None = None) -> pd.Series:
        """Return classifier feature importances aligned to encoded feature names.

        Works for any tree/ensemble exposing ``feature_importances_``. Names come
        from the fitted preprocessor so they are readable (e.g. ``Contract_Two year``).
        """
        self._check_fitted()
        clf = self.pipeline.named_steps["clf"]
        if not hasattr(clf, "feature_importances_"):
            raise AttributeError(f"{type(clf).__name__} has no feature_importances_.")
        names = self.pipeline.named_steps["pre"].get_feature_names_out()
        imp = pd.Series(clf.feature_importances_, index=names).sort_values(ascending=False)
        return imp.head(top) if top else imp


# ---------------------------------------------------------------------------
# Cross-validated model comparison
# ---------------------------------------------------------------------------
def compare_models(
    X: pd.DataFrame,
    y: pd.Series,
    models: dict | None = None,
    cv_splits: int = 5,
    use_engineered: bool = True,
) -> pd.DataFrame:
    """Cross-validate every candidate model and return a tidy leaderboard.

    Uses *stratified* K-fold so each fold preserves the ~26.5% churn rate, and
    scores on ROC-AUC / PR-AUC / F1 / recall -- the metrics that matter for an
    imbalanced retention problem. The preprocessor is rebuilt inside the pipeline
    for every fold, so there is no cross-fold leakage.
    """
    models = models or candidate_models()
    cv = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=config.RANDOM_STATE)
    scoring = ["roc_auc", "average_precision", "f1", "recall", "accuracy"]

    rows = []
    for name, (estimator, _grid) in models.items():
        pre = features.build_preprocessor(use_engineered)
        pipe = Pipeline([("pre", pre), ("clf", estimator)])
        cvres = cross_validate(pipe, X, y, cv=cv, scoring=scoring, n_jobs=config.N_JOBS)
        rows.append(
            {
                "model": name,
                "roc_auc": cvres["test_roc_auc"].mean(),
                "roc_auc_std": cvres["test_roc_auc"].std(),
                "pr_auc": cvres["test_average_precision"].mean(),
                "f1": cvres["test_f1"].mean(),
                "recall": cvres["test_recall"].mean(),
                "accuracy": cvres["test_accuracy"].mean(),
                "fit_time_s": cvres["fit_time"].mean(),
            }
        )
    return (
        pd.DataFrame(rows)
        .sort_values("roc_auc", ascending=False)
        .reset_index(drop=True)
    )


def tune(
    name: str,
    X: pd.DataFrame,
    y: pd.Series,
    cv_splits: int = 5,
    use_engineered: bool = True,
    scoring: str = "roc_auc",
) -> GridSearchCV:
    """Grid-search the named candidate model and return the fitted search object.

    The best estimator can be lifted straight into a :class:`ChurnModel`.
    """
    models = candidate_models()
    if name not in models:
        raise KeyError(f"Unknown model {name!r}. Options: {list(models)}")
    estimator, grid = models[name]
    pre = features.build_preprocessor(use_engineered)
    pipe = Pipeline([("pre", pre), ("clf", estimator)])
    cv = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=config.RANDOM_STATE)
    search = GridSearchCV(pipe, grid, scoring=scoring, cv=cv, n_jobs=config.N_JOBS, refit=True)
    search.fit(X, y)
    return search
