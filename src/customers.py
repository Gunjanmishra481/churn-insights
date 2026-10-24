"""Shared customer-level helpers used by Phase 2 (EDA) and Phase 3 (churn model).

Key definitions, kept in one place so every phase uses the same ones:

- A *customer* is `customer_unique_id`. Olist issues a new `customer_id` for every
  order, so only `customer_unique_id` links a person's repeat purchases.
- A *purchase* is any order that wasn't canceled or unavailable. Those two
  statuses mean no sale happened, so they shouldn't count as "coming back".
- DATA_END is the first day with no reliable data. Order volume drops from
  ~6,500 in Aug 2018 to 16 in Sep 2018 and 4 in Oct 2018, so those last few
  orders are treated as an incomplete tail, not real activity. Any window that
  would run past DATA_END can't be observed fully and is excluded.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED = PROJECT_ROOT / "data" / "processed"

DATA_END = pd.Timestamp("2018-09-01")
NON_PURCHASE_STATUSES = ["canceled", "unavailable"]
RETURN_WINDOW_DAYS = 180


def load_orders() -> pd.DataFrame:
    """orders_enriched restricted to the reliable period (before DATA_END)."""
    orders = pd.read_parquet(PROCESSED / "orders_enriched.parquet")
    return orders[orders["order_purchase_timestamp"] < DATA_END].copy()


def load_purchases(orders: pd.DataFrame | None = None) -> pd.DataFrame:
    """Orders that count as purchases (not canceled/unavailable), sorted by time."""
    if orders is None:
        orders = load_orders()
    purchases = orders[~orders["order_status"].isin(NON_PURCHASE_STATUSES)]
    return purchases.sort_values("order_purchase_timestamp").reset_index(drop=True)


def first_orders_with_return_flag(purchases: pd.DataFrame,
                                  window_days: int = RETURN_WINDOW_DAYS) -> pd.DataFrame:
    """One row per customer: their first purchase, plus whether they bought again
    within `window_days` of it.

    Customers whose first purchase is too recent for the full window to be
    observed (first purchase + window > DATA_END) are dropped. Keeping them would
    wrongly label recent customers as "never came back" simply because we haven't
    seen enough time yet (right-censoring).
    """
    p = purchases[["customer_unique_id", "order_id", "order_purchase_timestamp"]]
    # drop_duplicates (not groupby().first()) keeps each first order's row intact;
    # groupby().first() would fill nulls from *other* orders of the same customer.
    first = (purchases.sort_values(["order_purchase_timestamp", "order_id"])
             .drop_duplicates("customer_unique_id", keep="first").reset_index(drop=True))

    first_ts = first.set_index("customer_unique_id")["order_purchase_timestamp"]
    later = p.copy()
    start = later["customer_unique_id"].map(first_ts)
    # A return must be on a later calendar day. Olist sometimes splits one basket
    # into several orders placed seconds apart; those are the same shopping trip,
    # not a customer coming back.
    later = later[(later["order_purchase_timestamp"].dt.normalize() > start.dt.normalize())
                  & (later["order_purchase_timestamp"] <= start + pd.Timedelta(days=window_days))]
    returned = set(later["customer_unique_id"])

    first["returned_within_window"] = first["customer_unique_id"].isin(returned)
    observable = first["order_purchase_timestamp"] + pd.Timedelta(days=window_days) <= DATA_END
    return first[observable].reset_index(drop=True)
