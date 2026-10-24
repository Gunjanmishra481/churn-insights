# Customer Churn & Feedback Intelligence Platform

**Business question:** Which customers are about to leave, why, and what are they complaining about?

**Data:** [Olist Brazilian E-Commerce Public Dataset](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce): 99,441 orders (Sep 2016 – Oct 2018) across 9 linked tables, including Portuguese customer reviews.

## Setup
```bash
py -3.11 -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
# put the 9 Kaggle CSVs in data/raw/
copy .env.example .env   # then fill in Snowflake credentials
```
Requires Java 8/11/17 for PySpark (`JAVA_HOME` set).

### Snowflake setup (least privilege)
The pipeline logs in as a dedicated service user `CHURN_ETL_USER` (key-pair auth, no password) with role `CHURN_ETL_ROLE`. An admin creates the database and schema once; the role gets only:

| Privilege | Why it's needed |
|---|---|
| `USAGE` on warehouse `CHURN_WH` | run queries (no `OPERATE`/`MODIFY`: can't resize or reconfigure it) |
| `USAGE` on database `CHURN_INSIGHTS` and schema `CORE` | see the schema |
| `CREATE TABLE` on `CORE` | create the output tables (the role then owns them, so it can overwrite them) |
| `CREATE STAGE`, `CREATE FILE FORMAT` on `CORE` | `write_pandas` creates a temporary stage and file format for each upload |

The role cannot create databases or schemas, and has no access to any other database. It is granted to `SYSADMIN` so admins can still manage the tables it creates.

## Phase 1: ETL + data validation

```bash
python -m src.etl                   # full pipeline incl. Snowflake upload
python -m src.etl --skip-snowflake  # local outputs only
python -m pytest -v                 # tests
```

**Pipeline** (`src/etl.py`): load CSVs with explicit schemas → validate raw data → clean → aggregate child tables to one row per order → join → validate outputs → write Parquet to `data/processed/` → upload to Snowflake (`CHURN_INSIGHTS.CORE`, warehouse `CHURN_WH`).

| Output table | Grain | Rows |
|---|---|---|
| `orders_enriched` | one row per order: customer, delivery timing, item/payment/review aggregates, data-quality flags | 99,441 |
| `order_items_clean` | one row per order line, with category and seller | 112,650 |
| `reviews_clean` | one row per (review, order), Portuguese text kept | 99,224 |
| `geo_by_zip` | median lat/lng per zip prefix | 19,010 |

**Key design decisions**
- **Explicit schemas, `FAILFAST` parsing.** Zip prefixes stay strings ("01037"; 23,995 orders have a leading zero that type inference would drop), and malformed rows crash the load instead of silently becoming nulls.
- **Aggregate before joining.** Items, payments and reviews have several rows per order; they're reduced to one row per order first, so joins can't multiply rows or double-count money.
- **Flag, don't drop.** Suspect rows get `flag_*` columns; each later phase decides what to exclude.
- **Spark for transforms, pyarrow for writes.** The outputs are small (≤113k rows), so they're collected to pandas and written with pyarrow, which avoids Spark's Windows-only `winutils.exe` dependency.
- **`write_pandas` for Snowflake.** Parquet → internal stage → `COPY INTO`, no extra JARs; tables are overwritten on each run, so reruns are idempotent.

**Validation results** (full report: `reports/validation_report.csv`)
- All error-level checks pass: primary-key uniqueness, foreign-key integrity across all joins, required fields, value ranges, `orders_enriched` row count = raw order count, and price / freight / payment totals reconcile exactly (R$13,591,643.70 / R$2,251,909.54 / R$16,008,872.12).
- Known data quirks (warnings, kept and flagged):

| Quirk | Count |
|---|---|
| Orders with no items | 775 |
| Orders with no payment record | 1 |
| Orders with more than one review (latest kept per order) | 547 |
| Review IDs shared across several orders | 814 duplicate rows |
| Products with no category (→ `unknown`) | 610 |
| Categories missing from translation file (manually translated) | 2 |
| Delivered orders missing a delivery date | 8 |
| Timestamps out of sequence (e.g. delivered before handed to carrier) | 189 |
| Payment ≠ items + freight beyond max(R$1, 1%) | 249 of 98,665 orders (0.25%): 232 overpaid, 17 underpaid |
| Customer zip prefixes with no geolocation | 278 |
| Exact duplicate geolocation rows | 261,831 |

**First look:** 6.8% of delivered orders arrived after the estimated date; median delivery time was 10 days.

## Phase 2: EDA + hypothesis tests

```bash
jupyter notebook notebooks/02_eda_hypothesis_tests.ipynb
```

Analysis notebook with charts in `reports/figures/02_*.png` and all test results in `reports/phase2_hypothesis_tests.csv`. Shared definitions (what counts as a customer, a purchase and a "return") live in `src/customers.py`; test helpers with effect sizes are in `src/stats_tests.py`.

**Definitions**
- Customer = `customer_unique_id` (Olist issues a new `customer_id` per order).
- Purchase = order not `canceled`/`unavailable`.
- Analysis period ends 2018-09-01: Sep–Oct 2018 contain only 20 orders, an incomplete tail.
- A "return" is a purchase on a *later calendar day*; split orders from the same basket don't count.

**Overview:** 96,470 delivered orders with known timing (median 10 days, typically 12 days earlier than estimated); 6.8% arrived after the estimated date. 3.0% of customers ever bought more than once, and only 1.97% bought again within 180 days of their first order.

| | Hypothesis | Test | Result | Effect size | Holm-adjusted p |
|---|---|---|---|---|---|
| H1 | Late delivery → lower review score | Mann-Whitney U | 2.27 vs 4.29 stars | rank-biserial r = −0.64 (large) | < 0.001 |
| H2 | Late delivery → 1–2 star review | Chi-square | 62.4% vs 9.3% | 6.7× more likely (Cramér's V = 0.40) | < 0.001 |
| H3 | Late first order → no return in 180 days | Chi-square | 1.58% vs 2.01% return | relative risk 0.79 | 0.069 (not significant) |
| H4 | 1–2 star first review → no return in 180 days | Chi-square | 1.62% vs 2.04% return | relative risk 0.79 (very small) | 0.024 |
| H5 | Delivery time differs by state | Kruskal-Wallis | SP median 7 days vs AM 26 | ε² = 0.25 (large) | < 0.001 |

**Why these tests:** review scores are ordinal and skewed toward 5, so rank-based tests (Mann-Whitney, Kruskal-Wallis) are used instead of t-tests/ANOVA. Five tests are run together, so p-values are Holm-corrected. With ~100k rows nearly everything is "significant", so effect sizes decide what matters.

**Takeaways**
- Late delivery is the clearest driver of bad reviews.
- A bad first experience is linked to a slightly lower return rate, but the effect is small: most customers don't come back regardless.
- Delivery speed is strongly regional (fast south-east, slow north), which supports the Phase 4 delay model.
- These are associations, not causal effects; Phase 7's A/B test design is how to test a causal claim.

## Phase 3: Churn / return model

```bash
python -m src.train_churn
python -m pytest tests/test_churn_features.py -v
```

**Target.** For a cutoff date, `will_return` = the customer makes another purchase in the next 180 days; churn = `1 − will_return`. The rare event (returning, ~1–2%) is the positive class, because a model that predicts "everyone churns" would be ~98% accurate and useless.

**Snapshot design and leakage guards** (`src/churn_features.py`). Each cutoff produces one row per customer, using only data from before the cutoff: reviews written after the cutoff and deliveries completed after it are hidden (the order is marked `last_order_in_transit`). Features: RFM (recency, frequency in distinct shopping days, monetary), tenure, average order value, delivery and review history, and the most recent order's details (delay, review, installments, voucher, payment type, category, state).

**Time-based evaluation** (`src/train_churn.py`)
- Model selection: rolling-origin validation over 3 folds (validate on 2017-08, 2017-09, 2017-10 cutoffs).
- Final test: train on cutoffs 2017-02 … 2017-09 (all label windows close before 2018-03-01), test once on cutoff **2018-03-01**: 56,630 customers, 670 returners (1.18%).
- Imbalance: class weighting (`class_weight="balanced"`, `scale_pos_weight`), F1-tuned thresholds, and PR-AUC / top-5% recall instead of accuracy.

**Test results**

| Model | PR-AUC | ROC-AUC | Recall in top 5% | Lift in top 5% | Precision | Recall | F1 |
|---|---|---|---|---|---|---|---|
| Random | 0.012 | 0.500 | 5% | 1.0× | | | |
| **RFM rule (no ML)** | **0.034** | **0.588** | **13.9%** | **2.8×** | | | |
| Logistic regression (balanced) | 0.024 | 0.578 | 13.3% | 2.7× | 0.053 | 0.067 | 0.059 |
| XGBoost (weighted, all features) | 0.019 | 0.568 | 11.0% | 2.2× | 0.021 | 0.230 | 0.039 |

| Shopping days before cutoff | Customers | Returned within 180 days |
|---|---|---|
| 1 | 55,618 | 1.1% |
| 2 | 947 | 4.3% |
| 3+ | 65 | 24.6% |

**What this means**
- Repeat purchasing on Olist is close to unpredictable from order history: ~98% of customers buy once, and the best ranking finds about 2.8× more returners than random targeting in the top 5%.
- **The simple RFM rule beat both ML models**, in validation and on test. The one strong signal is whether a customer has *already* bought more than once; the ML models spread their capacity over weak, noisy signals (the XGBoost importance chart ranks individual states near the top, a sign of fitting noise).
- Recommendation: use the RFM rule (or logistic regression, which is close and gives probabilities) to rank customers for retention campaigns, and don't deploy the more complex XGBoost model on this data. Better signals would need data Olist doesn't share: browsing, marketing contacts, or customer identity across marketplaces.
- Phase 5 (SHAP) will explain what the models picked up and check performance by segment.

Outputs: `reports/phase3_validation_folds.csv`, `phase3_validation_summary.csv`, `phase3_test_results.csv`, `phase3_return_rate_by_frequency.csv`, `phase3_metrics.json`, `reports/figures/03_*.png`, `models/churn_xgb.json` (git-ignored), `data/processed/churn_scores_test.parquet`.
