"""Tests for Phase 1.

Part 1 checks that the validation functions catch problems in tiny fake tables
(a check that never fails is useless). Part 2 checks the real processed outputs.
Run with:  python -m pytest -v
"""
from pathlib import Path

import pandas as pd
import pytest

from src import validation as v

PROCESSED = Path(__file__).resolve().parents[1] / "data" / "processed"


# --- Part 1: the checks themselves --------------------------------------------------------
def test_check_unique_detects_duplicates(spark):
    df = spark.createDataFrame([("a",), ("a",), ("b",)], "id STRING")
    result = v.check_unique(df, ["id"], "t")
    assert not result.passed and result.n_failing == 1


def test_check_foreign_key_detects_orphans(spark):
    parent = spark.createDataFrame([("c1",)], "customer_id STRING")
    child = spark.createDataFrame([("o1", "c1"), ("o2", "c_missing")], "order_id STRING, customer_id STRING")
    result = v.check_foreign_key(child, "customer_id", parent, "customer_id", "orders", "customers")
    assert not result.passed and result.n_failing == 1


def test_check_not_null_and_range(spark):
    df = spark.createDataFrame([(1,), (5,), (None,), (7,)], "score INT")
    assert v.check_not_null(df, "score", "t").n_failing == 1
    assert v.check_in_range(df, "score", "t", 1, 5).n_failing == 1  # only the 7


def test_payment_check_respects_tolerance(spark):
    df = spark.createDataFrame(
        [
            ("exact", 100.0, 90.0, 10.0),      # matches
            ("interest", 100.9, 90.0, 10.0),   # R$0.90 gap: within tolerance
            ("overpaid", 120.0, 90.0, 10.0),   # R$20 gap: mismatch (paid more)
            ("underpaid", 50.0, 90.0, 10.0),   # R$50 gap: mismatch (paid less)
            ("no_items", 30.0, None, None),    # can't compare -> excluded
        ],
        "order_id STRING, payment_total DOUBLE, items_price_total DOUBLE, freight_total DOUBLE",
    )
    result = v.check_payment_matches_items(df)
    assert result.n_total == 4
    assert result.n_failing == 2
    assert result.severity == "warning"
    assert "1 paid more, 1 paid less" in result.detail


def test_raise_on_failures_ignores_warnings():
    warning = v.CheckResult("x", "t", "c", "warning", False, 5, 10)
    error = v.CheckResult("x", "t", "c", "error", False, 1, 10)
    v.raise_on_failures([warning])  # warnings alone must not stop the pipeline
    with pytest.raises(v.DataValidationError):
        v.raise_on_failures([warning, error])


# --- Part 2: the real outputs ---------------------------------------------------------------
@pytest.fixture(scope="module")
def orders():
    path = PROCESSED / "orders_enriched.parquet"
    if not path.exists():
        pytest.skip("run `python -m src.etl` first")
    return pd.read_parquet(path)


def test_orders_enriched_one_row_per_order(orders):
    # Same count as the raw orders file and unique ids => joins did not fan out.
    assert len(orders) == 99_441
    assert orders["order_id"].is_unique


def test_zip_codes_keep_leading_zeros(orders):
    zips = orders["customer_zip_code_prefix"]
    assert zips.str.len().eq(5).all()
    assert zips.str.startswith("0").any()


def test_late_flag_consistent_with_delay(orders):
    delivered = orders.dropna(subset=["delay_days"])
    assert (delivered["is_late"] == (delivered["delay_days"] > 0)).all()
    assert orders.loc[orders["delay_days"].isna(), "is_late"].isna().all()
