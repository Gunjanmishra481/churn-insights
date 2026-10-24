"""Reusable data-quality checks for PySpark DataFrames.

Each check returns a CheckResult instead of raising immediately, so the pipeline
can run every check, write one complete report, and only then decide whether to
stop. Two severities:
  - "error":   the data is broken (e.g. duplicate primary key) -> pipeline stops.
  - "warning": a known real-world quirk (e.g. orders with no items) -> logged only.
"""
from dataclasses import asdict, dataclass

import pandas as pd
from pyspark.sql import Column, DataFrame
from pyspark.sql import functions as F


class DataValidationError(Exception):
    """Raised when at least one error-severity check fails."""


@dataclass
class CheckResult:
    check: str        # what kind of check, e.g. "unique"
    table: str        # which table it ran on
    target: str       # column(s) checked
    severity: str     # "error" or "warning"
    passed: bool
    n_failing: int    # number of rows (or orders) that failed
    n_total: int
    detail: str = ""


def check_unique(df: DataFrame, cols: list[str], table: str, severity: str = "error") -> CheckResult:
    """Every combination of `cols` should appear only once (primary-key check)."""
    n_total = df.count()
    n_distinct = df.select(*cols).distinct().count()
    n_dupes = n_total - n_distinct
    return CheckResult("unique", table, ",".join(cols), severity, n_dupes == 0, n_dupes, n_total,
                       f"{n_dupes:,} duplicate rows on key")


def check_not_null(df: DataFrame, col: str, table: str, severity: str = "error") -> CheckResult:
    n_total = df.count()
    n_null = df.filter(F.col(col).isNull()).count()
    return CheckResult("not_null", table, col, severity, n_null == 0, n_null, n_total,
                       f"{n_null:,} nulls")


def check_foreign_key(child: DataFrame, child_col: str, parent: DataFrame, parent_col: str,
                      child_table: str, parent_table: str, severity: str = "error") -> CheckResult:
    """Every non-null child key must exist in the parent table (no orphan rows).

    A left_anti join keeps only child rows that have NO match in the parent,
    which is exactly the set of orphans.
    """
    keys = child.select(F.col(child_col).alias("_k")).filter(F.col("_k").isNotNull())
    parent_keys = parent.select(F.col(parent_col).alias("_k"))
    n_orphans = keys.join(parent_keys, on="_k", how="left_anti").count()
    return CheckResult("foreign_key", child_table, f"{child_col} -> {parent_table}.{parent_col}",
                       severity, n_orphans == 0, n_orphans, keys.count(), f"{n_orphans:,} orphan rows")


def check_has_match(parent: DataFrame, parent_col: str, child: DataFrame, child_col: str,
                    parent_table: str, child_table: str, severity: str = "warning") -> CheckResult:
    """Reverse of a foreign key: does every parent row have at least one child row?
    (e.g. does every order have at least one item?)"""
    parent_keys = parent.select(F.col(parent_col).alias("_k"))
    child_keys = child.select(F.col(child_col).alias("_k"))
    n_missing = parent_keys.join(child_keys, on="_k", how="left_anti").count()
    return CheckResult("has_match", parent_table, f"{parent_col} in {child_table}.{child_col}",
                       severity, n_missing == 0, n_missing, parent_keys.count(),
                       f"{n_missing:,} rows with no match")


def check_in_range(df: DataFrame, col: str, table: str, min_value=None, max_value=None,
                   severity: str = "error") -> CheckResult:
    """Non-null values of `col` must lie in [min_value, max_value] (either bound optional)."""
    bad = F.lit(False)
    if min_value is not None:
        bad = bad | (F.col(col) < min_value)
    if max_value is not None:
        bad = bad | (F.col(col) > max_value)
    n_total = df.filter(F.col(col).isNotNull()).count()
    n_bad = df.filter(bad).count()
    return CheckResult("in_range", table, col, severity, n_bad == 0, n_bad, n_total,
                       f"{n_bad:,} values outside [{min_value}, {max_value}]")


def check_condition(df: DataFrame, bad_rows: Column, name: str, table: str, target: str,
                    severity: str = "warning") -> CheckResult:
    """Generic check: count rows matching a 'bad' condition."""
    n_total = df.count()
    n_bad = df.filter(bad_rows).count()
    return CheckResult(name, table, target, severity, n_bad == 0, n_bad, n_total,
                       f"{n_bad:,} rows match")


def check_row_count(df: DataFrame, expected: int, table: str, severity: str = "error") -> CheckResult:
    """Used after joins: if the row count changed, a join multiplied (fanned out) or dropped rows."""
    n = df.count()
    return CheckResult("row_count", table, "*", severity, n == expected, abs(n - expected), expected,
                       f"expected {expected:,}, got {n:,}")


def check_sum_matches(df: DataFrame, col: str, expected_total: float, table: str,
                      tolerance: float = 0.01, severity: str = "error") -> CheckResult:
    """Reconciliation: a column total must equal the same total computed from the raw data.
    This catches values that were double-counted or lost during aggregation and joins."""
    actual = df.agg(F.sum(col)).first()[0] or 0.0
    diff = abs(actual - expected_total)
    return CheckResult("sum_matches", table, col, severity, diff <= tolerance, int(diff > tolerance),
                       1, f"expected {expected_total:,.2f}, got {actual:,.2f}")


# --- Payment vs. items reconciliation --------------------------------------
# Customers should pay roughly (item prices + freight). Exact equality is not
# expected: instalment interest makes card payments a bit higher, and vouchers
# can make them lower. So we only call it a mismatch when the gap is bigger than
# R$1.00 AND bigger than 1% of the expected amount.
PAYMENT_ABS_TOL = 1.00
PAYMENT_REL_TOL = 0.01


def payment_mismatch_col(paid: str = "payment_total", items: str = "items_price_total",
                         freight: str = "freight_total") -> Column:
    """True when payment differs from items+freight beyond tolerance; null if either side is missing."""
    expected = F.col(items) + F.col(freight)
    gap = F.abs(F.col(paid) - expected)
    return (gap > F.lit(PAYMENT_ABS_TOL)) & (gap > expected * F.lit(PAYMENT_REL_TOL))


def check_payment_matches_items(orders: DataFrame, table: str = "orders_enriched",
                                severity: str = "warning") -> CheckResult:
    """Warning-level: how many orders' payments don't roughly match item price + freight?"""
    both = orders.filter(F.col("payment_total").isNotNull() & F.col("items_price_total").isNotNull())
    expected = F.col("items_price_total") + F.col("freight_total")
    flagged = both.filter(payment_mismatch_col())
    n_total = both.count()
    n_bad = flagged.count()
    n_over = flagged.filter(F.col("payment_total") > expected).count()
    pct = 100 * n_bad / n_total if n_total else 0.0
    return CheckResult(
        "payment_vs_items", table, "payment_total ~ items_price_total + freight_total", severity,
        n_bad == 0, n_bad, n_total,
        f"{n_bad:,} of {n_total:,} orders ({pct:.2f}%) mismatch beyond max(R${PAYMENT_ABS_TOL:.2f}, "
        f"{PAYMENT_REL_TOL:.0%}): {n_over:,} paid more, {n_bad - n_over:,} paid less")


# --- Reporting ---------------------------------------------------------------
def results_to_frame(results: list[CheckResult]) -> pd.DataFrame:
    return pd.DataFrame([asdict(r) for r in results])


def print_results(results: list[CheckResult]) -> None:
    for r in results:
        status = "PASS" if r.passed else ("FAIL" if r.severity == "error" else "WARN")
        print(f"  [{status}] {r.table:20s} {r.check:16s} {r.target:45s} {r.detail}")


def raise_on_failures(results: list[CheckResult]) -> None:
    failed = [r for r in results if r.severity == "error" and not r.passed]
    if failed:
        lines = "\n".join(f"  {r.table}.{r.target} ({r.check}): {r.detail}" for r in failed)
        raise DataValidationError(f"{len(failed)} error-level check(s) failed:\n{lines}")
