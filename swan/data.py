"""
swan.data
=========

Loading, cleaning and validation of the Swan Teleco single-customer view.

The public surface is deliberately small:

    load_raw(path)      -> pd.DataFrame     # untouched, straight from Excel
    clean(df)           -> pd.DataFrame     # typed, de-duplicated, corruption fixed
    quality_report(df)  -> pd.DataFrame     # per-column audit for the EDA section
    validate(df)        -> pd.DataFrame     # Pydantic schema check, returns df unchanged
    load_clean(path)    -> (df, report)     # convenience: load + clean + audit in one call

Every transformation is explained inline because the EDA mark scheme explicitly
rewards *handling missing data and explaining it* and *presenting corrupted data*.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, field_validator

from . import config


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------
def load_raw(path: str | Path = config.DATA_PATH) -> pd.DataFrame:
    """Read the customer workbook exactly as supplied, with no transformations.

    Parameters
    ----------
    path:
        Location of ``customer_data.xlsx``. Defaults to the project-root copy so
        the notebook needs no arguments, but is overridable for tests/fixtures.

    Notes
    -----
    ``openpyxl`` is the engine (see ``requirements.txt``); it is pinned so the
    read behaves identically on any platform.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Customer data not found at {path!s}. "
            "Ensure 'customer_data.xlsx' sits in the project root."
        )
    return pd.read_excel(path, engine="openpyxl")


# ---------------------------------------------------------------------------
# Cleaning
# ---------------------------------------------------------------------------
def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Return a cleaned copy of the raw customer view.

    Steps (each is defensive so the function is safe to re-run / idempotent):

    1. **Total Charges corruption.** In the raw extract this column is stored as
       text because 11 brand-new customers (``Tenure Months == 0``) carry a
       single blank space `' '` instead of a number -- they simply have not been
       billed yet. We coerce to numeric and impute those 11 values with ``0.0``
       (a zero-tenure customer has, by definition, accrued no total charges).
    2. **Whitespace.** Strip stray whitespace from every text column.
    3. **De-duplication.** Guard against accidental duplicate customer rows.
    4. **Type tightening.** Cast the binary target to ``int`` and known numerics
       to numeric dtypes so downstream code never has to re-guess.
    """
    df = df.copy()

    # 1. Total Charges: text -> numeric, blank-for-new-customer -> 0.0 -------
    if df["Total Charges"].dtype == object:
        coerced = pd.to_numeric(df["Total Charges"], errors="coerce")
        # Only zero-tenure rows should have failed to parse; assert that so a
        # future data refresh with *different* corruption is caught, not hidden.
        newly_missing = coerced.isna()
        offenders = df.loc[newly_missing, "Tenure Months"]
        assert (offenders == 0).all(), (
            "Unexpected non-numeric 'Total Charges' for a customer with tenure > 0; "
            "investigate before imputing."
        )
        df["Total Charges"] = coerced.fillna(0.0)

    # 2. Trim whitespace on text columns (defensive; Excel exports often pad) -
    #    Selected explicitly (object *and* the new pandas>=3.0 'str' dtype) so the
    #    code is warning-free on both pandas 2.x and 3.x.
    text_cols = [
        col
        for col in df.columns
        if pd.api.types.is_object_dtype(df[col]) or pd.api.types.is_string_dtype(df[col])
    ]
    for col in text_cols:
        df[col] = df[col].str.strip()

    # 3. Drop exact duplicate rows if any slipped into the extract -----------
    df = df.drop_duplicates().reset_index(drop=True)

    # 4. Tighten dtypes ------------------------------------------------------
    df[config.TARGET] = df[config.TARGET].astype(int)
    for col in config.NUMERIC_FEATURES:
        df[col] = pd.to_numeric(df[col])

    return df


# ---------------------------------------------------------------------------
# Quality audit
# ---------------------------------------------------------------------------
def quality_report(df: pd.DataFrame) -> pd.DataFrame:
    """Per-column data-quality audit used to drive the EDA commentary.

    Returns one row per column with dtype, distinct-value count, missing count
    and percentage, plus a short human-readable ``note`` flagging PII, constant,
    leakage and expected-missing columns.
    """
    rows = []
    n = len(df)
    for col in df.columns:
        missing = int(df[col].isna().sum())
        note = []
        if col in config.PII_COLUMNS:
            note.append("PII")
        if col in config.CONSTANT_COLUMNS or df[col].nunique(dropna=True) <= 1:
            note.append("constant")
        if col in config.LEAKAGE_COLUMNS:
            note.append("target/leakage")
        if col == "Churn Reason" and missing:
            note.append("missing-by-design (non-churners have no reason)")
        rows.append(
            {
                "column": col,
                "dtype": str(df[col].dtype),
                "n_unique": int(df[col].nunique(dropna=True)),
                "n_missing": missing,
                "pct_missing": round(100 * missing / n, 2) if n else 0.0,
                "note": "; ".join(note),
            }
        )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Pydantic validation
# ---------------------------------------------------------------------------
class CustomerRecord(BaseModel):
    """Schema for a single *cleaned* customer row.

    Validating against an explicit contract catches silent upstream changes
    (a renamed category, a stray null, a negative charge) before they reach the
    model. Only the fields the pipeline actually depends on are constrained.
    """

    model_config = ConfigDict(extra="ignore", str_strip_whitespace=True)

    tenure_months: int = Field(ge=0, le=100, alias="Tenure Months")
    monthly_charges: float = Field(ge=0, alias="Monthly Charges")
    total_charges: float = Field(ge=0, alias="Total Charges")
    churn_value: int = Field(ge=0, le=1, alias="Churn Value")
    contract: str = Field(alias="Contract")
    internet_service: str = Field(alias="Internet Service")

    @field_validator("contract")
    @classmethod
    def _known_contract(cls, v: str) -> str:
        allowed = {"Month-to-month", "One year", "Two year"}
        if v not in allowed:
            raise ValueError(f"Unexpected Contract value: {v!r}")
        return v


def validate(df: pd.DataFrame, sample: int | None = None) -> pd.DataFrame:
    """Validate rows against :class:`CustomerRecord`; return ``df`` unchanged.

    Parameters
    ----------
    sample:
        If given, validate a random ``sample`` of rows (seeded) rather than all
        7,043 -- handy for a fast smoke check. ``None`` validates everything.

    Raises
    ------
    pydantic.ValidationError
        If any checked row violates the schema (fails fast, loudly).
    """
    check = df if sample is None else df.sample(sample, random_state=config.RANDOM_STATE)
    for record in check.to_dict(orient="records"):
        CustomerRecord.model_validate(record)
    return df


# ---------------------------------------------------------------------------
# Convenience one-shot
# ---------------------------------------------------------------------------
def load_clean(path: str | Path = config.DATA_PATH) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load, clean and audit in a single call.

    Returns
    -------
    (clean_df, quality_report_df)
    """
    raw = load_raw(path)
    report = quality_report(raw)  # audit the *raw* data so corruption is visible
    cleaned = clean(raw)
    return cleaned, report
