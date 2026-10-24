"""Phase 3: build customer-level snapshots for the churn / return model.

How the dataset is built (the "snapshot" design)
------------------------------------------------
Pick a cutoff date, e.g. 2018-03-01, and pretend it is "today":

    |---- history: everything before the cutoff ----|---- next 180 days ----|
                    -> features                           -> label

- Features use ONLY information that existed before the cutoff.
- The label is whether the customer makes another purchase in the 180 days
  after the cutoff (`will_return`). Churn is the opposite: `churned = 1 - will_return`.

We repeat this for several cutoffs. Older cutoffs become training data and the
newest cutoff becomes the test set, so the model is always evaluated on a
period it never saw: a *time-based split*, which is how it would be used in
real life.

Why predict "will return" instead of "will churn"?
About 98% of customers never come back (Phase 2). If "churned" were the
positive class, a model that says "everyone churns" would score ~98% accuracy
and an F1 near 0.99 while being useless. The rare, valuable event is a customer
returning, so we make that the positive class and judge the model on how well
it finds those customers (precision, recall, F1, PR-AUC). Churn risk is just
1 - P(return).

Leakage guards (things that would quietly inflate scores)
---------------------------------------------------------
- A review only counts if it was written before the cutoff.
- Delivery timing only counts if the order was delivered before the cutoff;
  otherwise it's missing and `last_order_in_transit` is set.
- The estimated delivery date is known at purchase, so it's always allowed.
- Nothing from the label window is used to build features.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.customers import RETURN_WINDOW_DAYS

TOP_CATEGORIES = [
    "bed_bath_table", "health_beauty", "sports_leisure", "furniture_decor",
    "computers_accessories", "housewares", "watches_gifts", "telephony",
    "garden_tools", "auto", "toys", "cool_stuff",
]

NUMERIC_FEATURES = [
    "recency_days", "tenure_days", "frequency", "n_orders", "monetary_total", "avg_order_value",
    "avg_items_per_order", "avg_freight_ratio", "share_orders_late", "avg_review_score",
    "share_low_reviews", "share_reviews_with_comment", "n_categories",
    "last_delay_days", "last_delivery_days", "last_estimated_days", "last_review_score",
    "last_max_installments", "last_n_items", "last_freight_ratio",
]
BINARY_FEATURES = ["last_is_late", "last_order_in_transit", "last_used_voucher", "last_review_has_comment"]
CATEGORICAL_FEATURES = ["customer_state", "last_payment_type", "last_category"]
FEATURES = NUMERIC_FEATURES + BINARY_FEATURES + CATEGORICAL_FEATURES
TARGET = "will_return"


def build_snapshot(purchases: pd.DataFrame, cutoff: str | pd.Timestamp,
                   window_days: int = RETURN_WINDOW_DAYS) -> pd.DataFrame:
    """One row per customer who bought anything before `cutoff`.

    `purchases` must come from src.customers.load_purchases().
    """
    cutoff = pd.Timestamp(cutoff)
    window_end = cutoff + pd.Timedelta(days=window_days)
    ts = purchases["order_purchase_timestamp"]

    hist = purchases[ts < cutoff].copy()
    future = purchases[(ts >= cutoff) & (ts < window_end)]

    # --- hide anything not yet known at the cutoff --------------------------
    delivered_known = hist["order_delivered_customer_date"] < cutoff
    for col in ["delivery_days", "delay_days"]:
        hist[col] = hist[col].astype("Float64").where(delivered_known)
    hist["is_late"] = hist["is_late"].where(delivered_known)
    review_known = hist["review_creation_date"] < cutoff
    hist["review_score"] = hist["review_score"].astype("Float64").where(review_known)
    hist["review_has_comment"] = hist["review_has_comment"].where(review_known)
    hist["low_review"] = (hist["review_score"] <= 2).astype("Float64").where(hist["review_score"].notna())
    # bought before the cutoff but not yet delivered by it (still in transit, or never delivered)
    hist["in_transit"] = ~delivered_known.fillna(False)

    hist["purchase_day"] = hist["order_purchase_timestamp"].dt.normalize()
    hist["freight_ratio"] = hist["freight_total"] / hist["items_price_total"].replace(0, np.nan)

    # --- whole-history aggregates (RFM and experience) -----------------------
    g = hist.groupby("customer_unique_id")
    feats = pd.DataFrame({
        "first_purchase": g["order_purchase_timestamp"].min(),
        "last_purchase": g["order_purchase_timestamp"].max(),
        # frequency counts distinct shopping days: Olist splits some baskets into
        # several orders placed seconds apart, which is one visit, not several.
        "frequency": g["purchase_day"].nunique(),
        "n_orders": g.size(),
        "monetary_total": g["payment_total"].sum(),
        "avg_items_per_order": g["n_items"].mean(),
        "avg_freight_ratio": g["freight_ratio"].mean(),
        "share_orders_late": g["is_late"].mean(),
        "avg_review_score": g["review_score"].mean(),
        "share_low_reviews": g["low_review"].mean(),
        "share_reviews_with_comment": g["review_has_comment"].mean(),
        "n_categories": g["main_category"].nunique(),
    })
    feats["recency_days"] = (cutoff - feats["last_purchase"]).dt.days
    feats["tenure_days"] = (cutoff - feats["first_purchase"]).dt.days
    feats["avg_order_value"] = feats["monetary_total"] / feats["n_orders"]

    # --- most recent order: the freshest experience the customer had ---------
    last = (hist.sort_values(["order_purchase_timestamp", "order_id"])
            .drop_duplicates("customer_unique_id", keep="last").set_index("customer_unique_id"))
    feats["last_delay_days"] = last["delay_days"]
    feats["last_delivery_days"] = last["delivery_days"]
    feats["last_estimated_days"] = last["estimated_days"]
    feats["last_review_score"] = last["review_score"]
    feats["last_max_installments"] = last["max_installments"]
    feats["last_n_items"] = last["n_items"]
    feats["last_freight_ratio"] = last["freight_ratio"]
    feats["last_is_late"] = last["is_late"]
    feats["last_order_in_transit"] = last["in_transit"]
    feats["last_used_voucher"] = last["used_voucher"]
    feats["last_review_has_comment"] = last["review_has_comment"]
    feats["customer_state"] = last["customer_state"]
    feats["last_payment_type"] = last["main_payment_type"].fillna("unknown")
    feats["last_category"] = last["main_category"].where(last["main_category"].isin(TOP_CATEGORIES), "other")

    # --- label ----------------------------------------------------------------
    feats[TARGET] = feats.index.isin(set(future["customer_unique_id"])).astype(int)
    feats["cutoff"] = cutoff

    # pandas nullable types -> plain float so every library treats NaN the same way
    for col in NUMERIC_FEATURES + BINARY_FEATURES:
        feats[col] = pd.to_numeric(feats[col].astype("Float64"), errors="coerce").astype(float)
    return feats.drop(columns=["first_purchase", "last_purchase"]).reset_index()


def build_snapshots(purchases: pd.DataFrame, cutoffs: list[str]) -> pd.DataFrame:
    return pd.concat([build_snapshot(purchases, c) for c in cutoffs], ignore_index=True)
