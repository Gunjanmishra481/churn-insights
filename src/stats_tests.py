"""Small, readable wrappers around SciPy hypothesis tests (Phase 2).

Each function returns a plain dict so results can be collected into one table
(reports/phase2_hypothesis_tests.csv) and quoted in the README.

Why these tests?
- Review scores are 1-5 stars: ordinal and heavily skewed toward 5, not normally
  distributed. So we compare groups with the Mann-Whitney U test (rank-based,
  no normality assumption) instead of a t-test. We still report the means,
  because they are easy to explain to a business audience.
- Yes/no outcomes (e.g. "gave 1-2 stars", "came back to buy again") are compared
  with a chi-square test of independence on a 2x2 table.
- Delivery time across many states is compared with Kruskal-Wallis, the
  rank-based version of one-way ANOVA.

With ~100k rows almost any difference is "statistically significant", so every
test also reports an effect size. The effect size answers "is the difference big
enough to matter?", which is the question a business stakeholder actually has.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats


def mann_whitney(a: pd.Series, b: pd.Series, name: str, label_a: str, label_b: str) -> dict:
    """Two-sided Mann-Whitney U test with rank-biserial correlation as effect size.

    Rank-biserial r ranges from -1 to 1: the probability that a random value from
    group A beats one from group B, minus the reverse. |r| ~0.1 small, ~0.3 medium,
    ~0.5 large.
    """
    a = pd.to_numeric(a, errors="coerce").dropna().astype(float)
    b = pd.to_numeric(b, errors="coerce").dropna().astype(float)
    u, p = stats.mannwhitneyu(a, b, alternative="two-sided")
    r = 2 * u / (len(a) * len(b)) - 1
    return {
        "hypothesis": name,
        "test": "Mann-Whitney U",
        "group_a": label_a,
        "group_b": label_b,
        "n_a": len(a),
        "n_b": len(b),
        "value_a": round(a.mean(), 3),
        "value_b": round(b.mean(), 3),
        "statistic": round(float(u), 1),
        "p_value": float(p),
        "effect_size": round(float(r), 3),
        "effect_size_name": "rank-biserial r",
    }


def chi_square_2x2(group: pd.Series, outcome: pd.Series, name: str,
                   label_a: str, label_b: str) -> dict:
    """Chi-square test of independence between a yes/no group and a yes/no outcome.

    group == True is group A, group == False is group B. Effect sizes:
    - rates in each group (what we'd tell a stakeholder),
    - relative risk = rate_A / rate_B,
    - Cramer's V (0 = no association, 1 = perfect; 0.1 small, 0.3 medium).
    """
    df = pd.DataFrame({"g": group.astype(bool), "y": outcome.astype(bool)}).dropna()
    table = pd.crosstab(df["g"], df["y"]).reindex(index=[True, False], columns=[True, False], fill_value=0)
    chi2, p, dof, _ = stats.chi2_contingency(table.values)
    n = table.values.sum()
    cramers_v = np.sqrt(chi2 / n)
    rate_a = table.loc[True, True] / table.loc[True].sum()
    rate_b = table.loc[False, True] / table.loc[False].sum()
    return {
        "hypothesis": name,
        "test": "Chi-square (2x2)",
        "group_a": label_a,
        "group_b": label_b,
        "n_a": int(table.loc[True].sum()),
        "n_b": int(table.loc[False].sum()),
        "value_a": round(float(rate_a), 4),
        "value_b": round(float(rate_b), 4),
        "statistic": round(float(chi2), 1),
        "p_value": float(p),
        "effect_size": round(float(rate_a / rate_b), 3) if rate_b > 0 else np.nan,
        "effect_size_name": f"relative risk (Cramer's V = {cramers_v:.3f})",
    }


def kruskal_by_group(values: pd.Series, groups: pd.Series, name: str, min_group_size: int = 30) -> dict:
    """Kruskal-Wallis test across groups, with epsilon-squared as effect size.

    Groups smaller than `min_group_size` are dropped so a handful of orders from a
    tiny state can't dominate the result. Epsilon-squared is the share of rank
    variance explained by the grouping (0.01 small, 0.08 medium, 0.26 large).
    """
    df = pd.DataFrame({"v": pd.to_numeric(values, errors="coerce"), "g": groups}).dropna()
    sizes = df["g"].value_counts()
    df = df[df["g"].isin(sizes[sizes >= min_group_size].index)]
    samples = [grp["v"].astype(float).values for _, grp in df.groupby("g")]
    h, p = stats.kruskal(*samples)
    n, k = len(df), len(samples)
    eps2 = (h - k + 1) / (n - k)
    med = df.groupby("g")["v"].median().sort_values()
    return {
        "hypothesis": name,
        "test": "Kruskal-Wallis",
        "group_a": f"fastest: {med.index[0]}",
        "group_b": f"slowest: {med.index[-1]}",
        "n_a": int(sizes[med.index[0]]),
        "n_b": int(sizes[med.index[-1]]),
        "value_a": float(med.iloc[0]),
        "value_b": float(med.iloc[-1]),
        "statistic": round(float(h), 1),
        "p_value": float(p),
        "effect_size": round(float(eps2), 3),
        "effect_size_name": f"epsilon-squared ({k} groups)",
    }


def holm_adjust(p_values: list[float]) -> list[float]:
    """Holm-Bonferroni correction: we test several hypotheses, so we control the
    chance of at least one false positive across the whole family of tests."""
    order = np.argsort(p_values)
    m = len(p_values)
    adjusted = np.empty(m)
    running_max = 0.0
    for rank, idx in enumerate(order):
        val = min(1.0, (m - rank) * p_values[idx])
        running_max = max(running_max, val)
        adjusted[idx] = running_max
    return adjusted.tolist()
