"""
Unit tests for the ``swan`` package.

Run with::

    pytest -q

The tests use a small synthetic fixture where possible so they run in
milliseconds and do not depend on the real Excel file; a couple of integration
tests read the real data if it is present and are skipped otherwise. The intent
is to guarantee the *contracts* the notebook relies on: corruption is fixed,
PII never reaches the feature matrix, and the model exposes a clean interface.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from swan import config, data, features, model


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
def _synthetic(n: int = 40) -> pd.DataFrame:
    """A tiny frame with the columns the pipeline touches, incl. one corrupt row."""
    rng = np.random.default_rng(config.RANDOM_STATE)
    df = pd.DataFrame(
        {
            "CustomerID": [f"ID-{i:04d}" for i in range(n)],
            "Count": 1,
            "Country": "United States",
            "State": "California",
            "City": "Los Angeles",
            "Zip Code": 90001,
            "Lat Long": "34.0, -118.2",
            "Latitude": 34.0,
            "Longitude": -118.2,
            "Gender": rng.choice(["Male", "Female"], n),
            "Senior Citizen": rng.choice(["Yes", "No"], n),
            "Partner": rng.choice(["Yes", "No"], n),
            "Dependents": rng.choice(["Yes", "No"], n),
            "Tenure Months": rng.integers(0, 72, n),
            "Phone Service": rng.choice(["Yes", "No"], n),
            "Multiple Lines": rng.choice(["Yes", "No", "No phone service"], n),
            "Internet Service": rng.choice(["DSL", "Fiber optic", "No"], n),
            "Online Security": rng.choice(["Yes", "No", "No internet service"], n),
            "Online Backup": rng.choice(["Yes", "No", "No internet service"], n),
            "Device Protection": rng.choice(["Yes", "No", "No internet service"], n),
            "Tech Support": rng.choice(["Yes", "No", "No internet service"], n),
            "Streaming TV": rng.choice(["Yes", "No", "No internet service"], n),
            "Streaming Movies": rng.choice(["Yes", "No", "No internet service"], n),
            "Contract": rng.choice(["Month-to-month", "One year", "Two year"], n),
            "Paperless Billing": rng.choice(["Yes", "No"], n),
            "Payment Method": rng.choice(
                ["Electronic check", "Mailed check",
                 "Bank transfer (automatic)", "Credit card (automatic)"], n
            ),
            "Monthly Charges": rng.uniform(20, 120, n).round(2),
            "Total Charges": rng.uniform(20, 8000, n).round(2).astype(object),
            "Churn Label": rng.choice(["Yes", "No"], n),
            "Churn Value": rng.integers(0, 2, n),
            "Churn Reason": None,
        }
    )
    # Inject the real-world corruption: a zero-tenure customer with a blank charge.
    df.loc[0, "Tenure Months"] = 0
    df.loc[0, "Total Charges"] = " "
    return df


@pytest.fixture()
def raw_df() -> pd.DataFrame:
    return _synthetic()


@pytest.fixture()
def clean_df(raw_df) -> pd.DataFrame:
    return data.clean(raw_df)


# ---------------------------------------------------------------------------
# Cleaning contracts
# ---------------------------------------------------------------------------
def test_total_charges_becomes_numeric_and_blank_imputed_to_zero(clean_df):
    assert pd.api.types.is_numeric_dtype(clean_df["Total Charges"])
    assert clean_df["Total Charges"].isna().sum() == 0
    # The corrupted zero-tenure row must have been imputed with 0.0
    assert clean_df.loc[0, "Total Charges"] == 0.0


def test_clean_is_idempotent(clean_df):
    twice = data.clean(clean_df)
    pd.testing.assert_frame_equal(clean_df, twice)


def test_clean_raises_if_nonzero_tenure_has_blank_charge(raw_df):
    raw_df.loc[1, "Tenure Months"] = 12
    raw_df.loc[1, "Total Charges"] = " "  # corruption that should NOT be silently imputed
    with pytest.raises(AssertionError):
        data.clean(raw_df)


def test_target_is_integer(clean_df):
    assert pd.api.types.is_integer_dtype(clean_df[config.TARGET])


# ---------------------------------------------------------------------------
# Feature / leakage / PII contracts
# ---------------------------------------------------------------------------
def test_no_pii_or_leakage_in_feature_matrix(clean_df):
    X, y = features.build_feature_frame(clean_df)
    forbidden = set(config.PII_COLUMNS) | set(config.LEAKAGE_COLUMNS) | set(config.CONSTANT_COLUMNS)
    assert not (set(X.columns) & forbidden), "PII / leakage / constant column leaked into X"
    assert y.name == config.TARGET
    assert len(X) == len(y)


def test_engineered_features_are_present_and_valid(clean_df):
    eng = features.engineer(clean_df)
    assert eng["Num Addon Services"].between(0, 6).all()
    assert eng["Avg Monthly Spend"].notna().all()  # tenure-0 fallback must fire
    assert set(eng["New Customer"].unique()) <= {"Yes", "No"}


def test_preprocessor_output_is_all_numeric(clean_df):
    X, _ = features.build_feature_frame(clean_df)
    pre = features.build_preprocessor()
    arr = pre.fit_transform(X)
    arr = arr.toarray() if hasattr(arr, "toarray") else np.asarray(arr)
    assert np.isfinite(arr).all()


# ---------------------------------------------------------------------------
# Model interface contracts
# ---------------------------------------------------------------------------
def test_risk_score_and_band_boundaries():
    assert model.risk_score(0.0) == 0
    assert model.risk_score(1.0) == 100
    assert model.risk_band(0.1) == "Low"
    assert model.risk_band(0.45) == "Medium"
    assert model.risk_band(0.9) == "High"
    assert model.risk_band(1.0) == "High"  # boundary safety


def test_churnmodel_predict_proba_in_unit_interval(clean_df):
    from sklearn.tree import DecisionTreeClassifier

    X, y = features.build_feature_frame(clean_df)
    m = model.ChurnModel(DecisionTreeClassifier(max_depth=3, random_state=0)).fit(X, y)
    proba = m.predict_proba(X)
    assert proba.shape == (len(X),)
    assert ((proba >= 0) & (proba <= 1)).all()

    scored = m.score_frame(X)
    assert list(scored.columns) == ["churn_probability", "risk_score", "risk_band"]
    assert scored.index.equals(X.index)


def test_churnmodel_raises_before_fit():
    from sklearn.tree import DecisionTreeClassifier

    m = model.ChurnModel(DecisionTreeClassifier())
    with pytest.raises(RuntimeError):
        m.predict_proba(pd.DataFrame({"x": [1]}))


# ---------------------------------------------------------------------------
# Integration test against the real workbook (skipped if absent)
# ---------------------------------------------------------------------------
@pytest.mark.skipif(not config.DATA_PATH.exists(), reason="real customer_data.xlsx not present")
def test_real_data_loads_and_validates():
    df, report = data.load_clean()
    assert df.shape == (7043, 31)
    assert df["Total Charges"].isna().sum() == 0
    data.validate(df, sample=200)  # must not raise
    assert report.loc[report["column"] == "Churn Reason", "n_missing"].iloc[0] == 5174
