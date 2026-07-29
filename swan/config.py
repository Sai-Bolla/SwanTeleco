"""
swan.config
===========

Single source of truth for every constant used across the project.

Keeping paths, business rules and column semantics in one place means the
notebook, the package and the unit tests can never drift apart, and a reviewer
can audit *what counts as PII* or *what leaks the target* in one glance.
"""

from __future__ import annotations

from pathlib import Path

# ---------------------------------------------------------------------------
# Paths -- resolved relative to the project root so the project is portable
# across machines and operating systems (no hard-coded C:\ or /home paths).
# ---------------------------------------------------------------------------
PROJECT_ROOT: Path = Path(__file__).resolve().parents[1]
DATA_PATH: Path = PROJECT_ROOT / "customer_data.xlsx"
OUTPUT_DIR: Path = PROJECT_ROOT / "outputs"

# ---------------------------------------------------------------------------
# Reproducibility -- one seed, referenced everywhere randomness is involved.
# ---------------------------------------------------------------------------
RANDOM_STATE: int = 42

# Parallelism for scikit-learn / joblib. Defaults to 1 (single-process) because
# joblib's ``loky`` backend can exhaust memory spawning worker processes on some
# Windows/low-RAM machines. It is a single knob: bump to -1 (all cores) on a
# well-resourced machine for a speed-up. Keeping it here makes the whole project
# behave identically and reproducibly regardless of the host.
N_JOBS: int = 1

# ---------------------------------------------------------------------------
# Business constants (straight from Eliza Schuyler's brief)
# ---------------------------------------------------------------------------
INCENTIVE_BUDGET_PER_METRIC: float = 2.50  # $ the new-customer team may spend per retention metric
MAILER_UPTAKE_RATE: float = 0.20           # expected response rate to a mailer offer
TOP_N_AT_RISK: int = 500                   # size of the mailer campaign shortlist

# ---------------------------------------------------------------------------
# Target
# ---------------------------------------------------------------------------
TARGET: str = "Churn Value"  # 1 = churned, 0 = retained

# ---------------------------------------------------------------------------
# Column semantics
# ---------------------------------------------------------------------------
# Personally identifiable / precise-geolocation fields.
# Used only as join keys or for internal processing -- never fed to a model and
# never shown in a stakeholder aggregate. ``CustomerID`` is the one exception:
# it is retained purely as the contact key for the mailer list (a mail-out is
# impossible without it), and is still excluded from every model feature.
PII_COLUMNS: list[str] = [
    "CustomerID",
    "Lat Long",
    "Latitude",
    "Longitude",
    "Zip Code",
    "City",
]

# Fields that are constant for every one of the 7,043 rows -- all customers are
# in California, so ``Count`` (=1), ``Country`` and ``State`` carry zero signal.
CONSTANT_COLUMNS: list[str] = ["Count", "Country", "State"]

# Columns that ARE the target (or a direct post-hoc encoding of it). Including
# any of these as a feature would leak the answer, so they are dropped before
# modelling. ``Churn Reason`` is only ever populated *after* a customer churns.
LEAKAGE_COLUMNS: list[str] = ["Churn Label", "Churn Value", "Churn Reason"]

# ---------------------------------------------------------------------------
# Feature groups (used to build the preprocessing ColumnTransformer)
# ---------------------------------------------------------------------------
# Continuous numeric predictors.
NUMERIC_FEATURES: list[str] = [
    "Tenure Months",
    "Monthly Charges",
    "Total Charges",
]

# Categorical predictors (binary and multi-class). ``Senior Citizen`` arrives as
# "Yes"/"No" text in this extract, so it is treated as categorical, not numeric.
CATEGORICAL_FEATURES: list[str] = [
    "Gender",
    "Senior Citizen",
    "Partner",
    "Dependents",
    "Phone Service",
    "Multiple Lines",
    "Internet Service",
    "Online Security",
    "Online Backup",
    "Device Protection",
    "Tech Support",
    "Streaming TV",
    "Streaming Movies",
    "Contract",
    "Paperless Billing",
    "Payment Method",
]

# The six optional add-on services -- referenced by feature engineering to count
# how many extras a customer holds, and by the retention/incentive analysis.
ADDON_SERVICES: list[str] = [
    "Online Security",
    "Online Backup",
    "Device Protection",
    "Tech Support",
    "Streaming TV",
    "Streaming Movies",
]

# Risk-band thresholds (on churn probability) for the customer-service team.
# Chosen so the bands are interpretable during a live call.
RISK_BANDS: dict[str, tuple[float, float]] = {
    "Low": (0.00, 0.30),
    "Medium": (0.30, 0.60),
    "High": (0.60, 1.01),  # upper bound > 1 so a probability of exactly 1.0 lands in High
}
