# Swan Teleco — Customer Churn & Retention Analysis

End-to-end churn analysis for **Swan Teleco** (7,043 California customers). The project
profiles who is churning and why, quantifies what promotes retention, and ships a
tuned model that scores every active customer and produces a Top-500 mailer list.

Stakeholder: **Eliza Schuyler, Retention Marketing Manager.**

## What's here

| Path | Purpose |
|------|---------|
| `Analysis.ipynb` | The narrative deliverable — quality checks → EDA → feature engineering → modelling → evaluation → outputs. Run top-to-bottom. |
| `swan/` | Reusable, unit-tested package that does the work (`config`, `data`, `features`, `model`, `viz`). |
| `tests/` | `pytest` unit tests for the package (cleaning, no-PII-leakage, feature shapes, model interface). |
| `outputs/` | Generated deliverables (created when the notebook runs). |
| `customer_data.xlsx` | The single customer view (input). |
| `requirements.txt` | Pinned dependencies. |

### Package layout (`swan/`)
- **`config`** — single source of truth: paths, business constants (\$2.50 budget, 20% uptake, Top-500), and column semantics (PII / constant / leakage / feature groups). One seed (`RANDOM_STATE = 42`).
- **`data`** — `load_raw`, `clean` (fixes the corrupted `Total Charges`), `quality_report`, and Pydantic `validate`.
- **`features`** — feature engineering + a leak-free scikit-learn preprocessing `ColumnTransformer`.
- **`model`** — the `ChurnModel` wrapper, four classifiers with tuning grids, `compare_models` (K-fold CV), `tune` (GridSearch), and interpretable `risk_score` / `risk_band`.
- **`viz`** — one consistent chart theme + reusable plot helpers.

## Setup & run

```bash
pip install -r requirements.txt      # 1. dependencies
pytest -q                            # 2. (optional) run the unit tests — 11 should pass
jupyter notebook Analysis.ipynb      # 3. open, then Kernel → Restart & Run All
```

Runtime ≈ 3 minutes on a laptop. All logic is deterministic (seed = 42).
Set `swan.config.N_JOBS = -1` to use all CPU cores (defaults to 1 for portability).

> **Environment note.** `requirements.txt` pins `numpy` to the 2.3.x line: the installed
> `matplotlib 3.11.0` build is ABI-incompatible with `numpy 2.5.0` and segfaults when
> rendering. The pin resolves it while satisfying pandas / scikit-learn / scipy.
> This project's tree-based models avoid `scipy.linalg`, which is independently unstable
> in some builds of this environment.

## Deliverables (written to `outputs/`)

| File | Contents |
|------|----------|
| `top_500_at_risk.csv` | Ranked, contact-ready mailer list of the 500 highest-risk **active** customers, with recommended actions. |
| `churn_risk_scores_all_customers.csv` | 0–100 risk score, Low/Medium/High band and recommended actions for **every** active customer. |
| `swan_churn_deliverables.xlsx` | The above as stakeholder-friendly tabs, plus the model leaderboard. |
| `model_leaderboard.csv` | Cross-validated comparison of all four classifiers. |
| `final_churn_model.joblib` | The final model (preprocessing + classifier), retrained on all data, for scoring future customers. |

## Headline results

- Overall churn **26.5%**, heavily **front-loaded** in the first months.
- Biggest lever: **contract type** (month-to-month ≈ 43% churn vs two-year ≈ 3%).
- Recommended **\$2.50 incentive**: subsidise **Online Security** (then **Tech Support**) sign-ups.
- Final model: tuned **Random Forest**, held-out **ROC-AUC ≈ 0.85**, churn **recall ≈ 0.80**.

## Data responsibility

PII / geolocation fields (`CustomerID`, `Lat Long`, `Latitude`, `Longitude`, `Zip Code`,
`City`) are declared once in `swan.config` and excluded from every model feature and every
stakeholder aggregate by construction. `CustomerID` is retained **only** as the mailer
contact key. `Churn Reason` (populated only after churn) is treated as target leakage and
never used as a predictor.
