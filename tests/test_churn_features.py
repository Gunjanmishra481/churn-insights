"""Tests for Phases 2-3: customer definitions and leakage guards.

These use tiny hand-made purchase tables, so every expected value can be checked
by eye. Run with:  python -m pytest tests/test_churn_features.py -v
"""
import pandas as pd
import pytest

from src.churn_features import FEATURES, TARGET, build_snapshot
from src.customers import first_orders_with_return_flag

T = pd.Timestamp


def make_purchases(rows):
    """rows: (customer, order_id, purchased, delivered, review_created, review_score)"""
    df = pd.DataFrame(rows, columns=["customer_unique_id", "order_id", "order_purchase_timestamp",
                                     "order_delivered_customer_date", "review_creation_date", "review_score"])
    for col in ["order_purchase_timestamp", "order_delivered_customer_date", "review_creation_date"]:
        df[col] = pd.to_datetime(df[col], format="ISO8601")
    n = len(df)
    df = df.assign(
        order_status="delivered", customer_state="SP", main_payment_type="credit_card",
        main_category="health_beauty", payment_total=100.0, items_price_total=90.0, freight_total=10.0,
        n_items=1, max_installments=1, used_voucher=False, review_has_comment=True,
        estimated_days=20,
    )
    df["delivery_days"] = (df["order_delivered_customer_date"] - df["order_purchase_timestamp"]).dt.days
    df["delay_days"] = df["delivery_days"] - df["estimated_days"]
    df["is_late"] = df["delay_days"] > 0
    df.loc[df["order_delivered_customer_date"].isna(), ["delivery_days", "delay_days"]] = None
    df["review_score"] = df["review_score"].astype("Int64")
    assert len(df) == n
    return df.sort_values("order_purchase_timestamp").reset_index(drop=True)


@pytest.fixture
def purchases():
    return make_purchases([
        # A: bought twice before the cutoff, returns after it
        ("A", "a1", "2018-01-05", "2018-01-15", "2018-01-16", 5),
        ("A", "a2", "2018-02-10", "2018-02-20", "2018-02-21", 4),
        ("A", "a3", "2018-04-01", "2018-04-10", "2018-04-11", 5),
        # B: one order just before the cutoff; delivered and reviewed AFTER the cutoff
        ("B", "b1", "2018-02-25", "2018-03-10", "2018-03-11", 1),
        # C: bought once, returns only after the 180-day window
        ("C", "c1", "2018-01-10", "2018-01-20", "2018-01-21", 3),
        ("C", "c2", "2018-12-01", "2018-12-10", "2018-12-11", 3),
    ])


def test_snapshot_has_one_row_per_customer_seen_before_cutoff(purchases):
    snap = build_snapshot(purchases, "2018-03-01")
    assert sorted(snap["customer_unique_id"]) == ["A", "B", "C"]
    assert set(FEATURES) <= set(snap.columns)


def test_label_only_counts_purchases_inside_window(purchases):
    snap = build_snapshot(purchases, "2018-03-01").set_index("customer_unique_id")
    assert snap.loc["A", TARGET] == 1   # returned 2018-04-01, inside the window
    assert snap.loc["B", TARGET] == 0   # never returned
    assert snap.loc["C", TARGET] == 0   # returned 2018-12-01, after the window closed


def test_no_information_from_after_cutoff_leaks_into_features(purchases):
    snap = build_snapshot(purchases, "2018-03-01").set_index("customer_unique_id")
    # B's order was delivered and reviewed after the cutoff -> must be unknown
    assert pd.isna(snap.loc["B", "last_review_score"])
    assert pd.isna(snap.loc["B", "last_delivery_days"])
    assert snap.loc["B", "last_order_in_transit"] == 1
    # A's future purchase (a3) must not count toward history
    assert snap.loc["A", "frequency"] == 2
    assert snap.loc["A", "last_review_score"] == 4


def test_rfm_values(purchases):
    snap = build_snapshot(purchases, "2018-03-01").set_index("customer_unique_id")
    assert snap.loc["A", "recency_days"] == (T("2018-03-01") - T("2018-02-10")).days
    assert snap.loc["A", "tenure_days"] == (T("2018-03-01") - T("2018-01-05")).days
    assert snap.loc["A", "monetary_total"] == 200.0


def test_same_day_split_orders_are_not_a_return():
    p = make_purchases([
        ("D", "d1", "2017-05-01 10:00:00", "2017-05-10", "2017-05-11", 5),
        ("D", "d2", "2017-05-01 10:00:05", "2017-05-10", "2017-05-11", 5),  # same basket, split order
        ("E", "e1", "2017-05-01", "2017-05-10", "2017-05-11", 5),
        ("E", "e2", "2017-06-15", "2017-06-25", "2017-06-26", 5),          # a real return
    ])
    first = first_orders_with_return_flag(p).set_index("customer_unique_id")
    assert not first.loc["D", "returned_within_window"]
    assert first.loc["E", "returned_within_window"]


def test_customers_without_a_full_window_are_excluded():
    p = make_purchases([("F", "f1", "2018-08-01", "2018-08-10", "2018-08-11", 5)])
    assert first_orders_with_return_flag(p).empty  # 2018-08-01 + 180 days runs past DATA_END
