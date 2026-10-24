"""Phase 3: train and evaluate the churn / return model.

Run:  python -m src.train_churn

Time-based evaluation (no shuffling across time)
------------------------------------------------
Model selection uses *rolling-origin validation*: three folds, each fitting on
monthly snapshots up to some month and validating on a later cutoff.

    fold 1: fit 2017-02 .. 2017-05  -> validate 2017-08-01
    fold 2: fit 2017-02 .. 2017-06  -> validate 2017-09-01
    fold 3: fit 2017-02 .. 2017-07  -> validate 2017-10-01

Averaging over three folds is far more stable than a single validation set when
there are only a few hundred positive customers per snapshot.

The final models are refitted on every training cutoff whose 180-day label
window closes before the test cutoff (2017-02-01 .. 2017-09-01), and scored
once on the test cutoff 2018-03-01 (labels 2018-03-01 .. 2018-08-28). The model
never sees anything after 2018-03-01, so the test score is an honest
"deploy it and see" estimate.

Models compared
---------------
1. RFM rule (no ML): rank customers by number of shopping days, then by recency.
   Any model has to beat this to be worth deploying.
2. Logistic regression, class_weight="balanced", all features.
3. XGBoost, all features, with and without class weighting.
4. XGBoost, unweighted, a smaller "core" feature set (RFM + delivery/review
   experience). Fewer features = less room to fit noise with few positives.

Class imbalance (~1.2-1.6% positives) is handled by class weighting
(scale_pos_weight / class_weight) where it helps, by tuning the decision
threshold for F1 instead of using 0.5, and by judging models on PR-AUC and
recall in the top 5% instead of accuracy. SMOTE was not used: with one-hot and
missing-value features, synthetic "in-between" customers aren't meaningful.
"""
from __future__ import annotations

import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, f1_score, precision_recall_curve,
                             precision_score, recall_score, roc_auc_score)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

from src.churn_features import (BINARY_FEATURES, CATEGORICAL_FEATURES, FEATURES, NUMERIC_FEATURES,
                                TARGET, build_snapshots)
from src.customers import PROCESSED, PROJECT_ROOT, load_purchases

SEED = 42
TOP_SHARE = 0.05  # "if marketing can contact the top 5% of customers..."


def months(start: str, end: str) -> list[str]:
    return [str(d.date()) for d in pd.date_range(start, end, freq="MS")]


FOLDS = [
    (months("2017-02-01", "2017-05-01"), ["2017-08-01"]),
    (months("2017-02-01", "2017-06-01"), ["2017-09-01"]),
    (months("2017-02-01", "2017-07-01"), ["2017-10-01"]),
]
TRAIN_CUTOFFS = months("2017-02-01", "2017-09-01")  # 180-day windows close by 2018-02-28
TEST_CUTOFFS = ["2018-03-01"]

CORE_FEATURES = ["recency_days", "frequency", "monetary_total", "tenure_days", "avg_order_value",
                 "avg_review_score", "share_orders_late", "last_review_score", "last_is_late",
                 "last_order_in_transit", "last_max_installments"]

REPORTS = PROJECT_ROOT / "reports"
FIGURES = REPORTS / "figures"
MODELS = PROJECT_ROOT / "models"


# --------------------------------------------------------------------------- models
def make_logreg() -> Pipeline:
    numeric = Pipeline([
        ("impute", SimpleImputer(strategy="median", add_indicator=True)),  # indicator keeps "was missing" as a signal
        ("scale", StandardScaler()),
    ])
    pre = ColumnTransformer([
        ("num", numeric, NUMERIC_FEATURES + BINARY_FEATURES),
        ("cat", OneHotEncoder(handle_unknown="ignore", min_frequency=50), CATEGORICAL_FEATURES),
    ])
    return Pipeline([("pre", pre), ("model", LogisticRegression(
        class_weight="balanced", max_iter=2000, C=0.5, random_state=SEED))])


def make_xgb(weighted: bool, features: str) -> Pipeline:
    # XGBoost handles missing values natively (it learns which branch NaN goes to),
    # so numeric columns pass through untouched; only categories are one-hot encoded.
    if features == "all":
        pre = ColumnTransformer([
            ("num", "passthrough", NUMERIC_FEATURES + BINARY_FEATURES),
            ("cat", OneHotEncoder(handle_unknown="ignore", min_frequency=50), CATEGORICAL_FEATURES),
        ])
    else:
        pre = ColumnTransformer([("num", "passthrough", CORE_FEATURES)])
    # Shallow trees, a high min_child_weight and row/column subsampling keep the
    # model from memorising the few hundred positive customers.
    model = XGBClassifier(n_estimators=300, max_depth=3, learning_rate=0.05, min_child_weight=50,
                          subsample=0.8, colsample_bytree=0.8, random_state=SEED, n_jobs=4,
                          tree_method="hist")
    pipe = Pipeline([("pre", pre), ("model", model)])
    pipe.weighted = weighted  # scale_pos_weight is set from the training labels in fit_model()
    return pipe


CANDIDATES = {
    "Logistic regression (balanced)": make_logreg,
    "XGBoost weighted, all features": lambda: make_xgb(True, "all"),
    "XGBoost unweighted, all features": lambda: make_xgb(False, "all"),
    "XGBoost unweighted, core features": lambda: make_xgb(False, "core"),
}


def fit_model(name: str, df: pd.DataFrame) -> Pipeline:
    model = CANDIDATES[name]()
    y = df[TARGET].values
    if getattr(model, "weighted", False):
        model.set_params(model__scale_pos_weight=float((y == 0).sum() / (y == 1).sum()))
    return model.fit(df[FEATURES], y)


def rfm_rule_score(df: pd.DataFrame) -> np.ndarray:
    """Higher = more likely to return: more shopping days first, then more recent."""
    return df["frequency"].values * 10_000 - df["recency_days"].values


# --------------------------------------------------------------------------- metrics
def best_f1_threshold(y: np.ndarray, p: np.ndarray) -> float:
    precision, recall, thresholds = precision_recall_curve(y, p)
    f1 = 2 * precision * recall / np.clip(precision + recall, 1e-12, None)
    return float(thresholds[np.nanargmax(f1[:-1])])


def evaluate(y: np.ndarray, p: np.ndarray, threshold: float | None = None) -> dict:
    out = {
        "n": int(len(y)),
        "positives": int(y.sum()),
        "base_rate": float(y.mean()),
        "pr_auc": float(average_precision_score(y, p)),
        "roc_auc": float(roc_auc_score(y, p)),
    }
    k = int(np.ceil(TOP_SHARE * len(y)))
    top = np.argsort(-p, kind="stable")[:k]
    out["recall_at_top5pct"] = float(y[top].sum() / y.sum())
    out["precision_at_top5pct"] = float(y[top].mean())
    out["lift_at_top5pct"] = out["precision_at_top5pct"] / out["base_rate"]
    if threshold is not None:
        pred = (p >= threshold).astype(int)
        out.update(threshold=float(threshold),
                   precision=float(precision_score(y, pred, zero_division=0)),
                   recall=float(recall_score(y, pred)),
                   f1=float(f1_score(y, pred)),
                   flagged=int(pred.sum()))
    return out


# --------------------------------------------------------------------------- main
def main() -> None:
    np.random.seed(SEED)
    FIGURES.mkdir(parents=True, exist_ok=True)
    MODELS.mkdir(exist_ok=True)

    purchases = load_purchases()
    cache: dict[str, pd.DataFrame] = {}

    def snaps(cutoffs: list[str]) -> pd.DataFrame:
        for c in cutoffs:
            if c not in cache:
                cache[c] = build_snapshots(purchases, [c])
        return pd.concat([cache[c] for c in cutoffs], ignore_index=True)

    # ---- 1. rolling-origin validation ----------------------------------------
    rows, last_fold_scores = [], {}
    for i, (fit_cutoffs, val_cutoffs) in enumerate(FOLDS, start=1):
        fit, val = snaps(fit_cutoffs), snaps(val_cutoffs)
        y_val = val[TARGET].values
        print(f"fold {i}: fit {len(fit):,} rows ({fit[TARGET].mean():.2%} return) -> "
              f"validate {val_cutoffs[0]}: {len(val):,} rows ({y_val.mean():.2%} return)")
        scores = {"RFM rule (no ML)": rfm_rule_score(val)}
        for name in CANDIDATES:
            scores[name] = fit_model(name, fit).predict_proba(val[FEATURES])[:, 1]
        for name, p in scores.items():
            rows.append({"fold": i, "model": name, **evaluate(y_val, p)})
        last_fold_scores = {k: (y_val, v) for k, v in scores.items()}

    folds = pd.DataFrame(rows)
    summary = (folds.groupby("model")[["pr_auc", "roc_auc", "recall_at_top5pct", "lift_at_top5pct"]]
               .mean().sort_values("pr_auc", ascending=False))
    print("\nValidation (mean of 3 folds):")
    print(summary.to_string(float_format=lambda x: f"{x:.3f}"))

    ml_only = summary.drop(index="RFM rule (no ML)")
    best_ml = ml_only.index[0]
    best_xgb = ml_only[ml_only.index.str.startswith("XGBoost")].index[0]
    to_test = list(dict.fromkeys(["Logistic regression (balanced)", best_xgb, best_ml]))
    # F1-optimal thresholds come from the most recent validation fold
    thresholds = {name: best_f1_threshold(*last_fold_scores[name]) for name in to_test}
    print(f"\nBest ML model on validation: {best_ml} | best XGBoost: {best_xgb}")

    # ---- 2. refit on the full training period, score the test period once -----
    train, test = snaps(TRAIN_CUTOFFS), snaps(TEST_CUTOFFS)
    y_test = test[TARGET].values
    print(f"\ntrain: {len(train):,} rows, {train[TARGET].sum():,} returners ({train[TARGET].mean():.2%})")
    print(f"test : {len(test):,} rows, {y_test.sum():,} returners ({y_test.mean():.2%})")

    final_models = {name: fit_model(name, train) for name in to_test}
    p_test = {"RFM rule (no ML)": rfm_rule_score(test)}
    p_test.update({name: m.predict_proba(test[FEATURES])[:, 1] for name, m in final_models.items()})
    test_results = pd.DataFrame([{"model": k, **evaluate(y_test, v, thresholds.get(k))} for k, v in p_test.items()])
    print("\nTEST (cutoff 2018-03-01):")
    print(test_results[["model", "pr_auc", "roc_auc", "precision", "recall", "f1",
                        "recall_at_top5pct", "lift_at_top5pct"]]
          .to_string(index=False, float_format=lambda x: f"{x:.3f}"))

    # return rate by number of shopping days: the single strongest signal
    by_freq = (test.assign(shopping_days=test["frequency"].clip(upper=3).astype(int).astype(str)
                           .replace("3", "3+"))
               .groupby("shopping_days")[TARGET].agg(customers="size", return_rate="mean"))
    print("\nTest return rate by shopping days before cutoff:")
    print(by_freq.to_string(float_format=lambda x: f"{x:.3f}"))

    # ---- 3. save outputs -------------------------------------------------------
    folds.to_csv(REPORTS / "phase3_validation_folds.csv", index=False)
    summary.reset_index().to_csv(REPORTS / "phase3_validation_summary.csv", index=False)
    test_results.to_csv(REPORTS / "phase3_test_results.csv", index=False)
    by_freq.reset_index().to_csv(REPORTS / "phase3_return_rate_by_frequency.csv", index=False)

    xgb_final = final_models[best_xgb]
    xgb_final.named_steps["model"].save_model(MODELS / "churn_xgb.json")
    feature_names = xgb_final.named_steps["pre"].get_feature_names_out().tolist()
    metadata = {
        "best_ml_model_on_validation": best_ml, "xgboost_model_saved": best_xgb,
        "thresholds": thresholds, "train_cutoffs": TRAIN_CUTOFFS, "test_cutoffs": TEST_CUTOFFS,
        "folds": FOLDS, "features": FEATURES, "core_features": CORE_FEATURES,
        "xgb_encoded_feature_names": feature_names,
        "validation_summary": summary.to_dict(orient="index"),
        "test_metrics": test_results.set_index("model").to_dict(orient="index"),
    }
    (REPORTS / "phase3_metrics.json").write_text(json.dumps(metadata, indent=2, default=str))

    scores = test[["customer_unique_id", "cutoff", "customer_state", "frequency", "recency_days", TARGET]].copy()
    scores["rfm_rule_score"] = p_test["RFM rule (no ML)"]
    scores["p_return_xgb"] = p_test[best_xgb]
    scores["churn_risk_xgb"] = 1 - scores["p_return_xgb"]
    scores.to_parquet(PROCESSED / "churn_scores_test.parquet", index=False)

    # PR curves on test
    fig, ax = plt.subplots(figsize=(7.5, 5))
    for name, p in p_test.items():
        pr, rc, _ = precision_recall_curve(y_test, p)
        ax.plot(rc, pr, label=f"{name} (PR-AUC {average_precision_score(y_test, p):.3f})")
    ax.axhline(y_test.mean(), ls="--", c="grey", label=f"random ({y_test.mean():.3f})")
    ax.set(title="Precision-recall on the test period (cutoff 2018-03-01)",
           xlabel="Recall (share of returners found)", ylabel="Precision", ylim=(0, 0.25))
    ax.legend(fontsize=8); fig.tight_layout(); fig.savefig(FIGURES / "03_pr_curve_test.png", dpi=150); plt.close(fig)

    # Gain-based importance: how much each feature improved the trees' splits.
    # (Phase 5 replaces this with SHAP, which also shows direction and per-customer effects.)
    gain = xgb_final.named_steps["model"].get_booster().get_score(importance_type="gain")
    imp = pd.Series({feature_names[int(k[1:])].split("__", 1)[1]: v for k, v in gain.items()})
    imp = imp.sort_values().tail(15)
    fig, ax = plt.subplots(figsize=(7, 5.5))
    ax.barh(imp.index, imp.values, color="#4C72B0")
    ax.set(title="XGBoost feature importance (gain), top 15", xlabel="Average gain")
    fig.tight_layout(); fig.savefig(FIGURES / "03_feature_importance.png", dpi=150); plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(by_freq.index, by_freq["return_rate"] * 100, color="#55A868")
    for x, (n, r) in enumerate(zip(by_freq["customers"], by_freq["return_rate"])):
        ax.text(x, r * 100, f"n={n:,}", ha="center", va="bottom", fontsize=8)
    ax.set(title="Return rate in next 180 days by past shopping days", xlabel="Shopping days before cutoff",
           ylabel="% who returned")
    fig.tight_layout(); fig.savefig(FIGURES / "03_return_rate_by_frequency.png", dpi=150); plt.close(fig)
    print("\nSaved reports/phase3_*, reports/figures/03_*.png, models/churn_xgb.json, "
          "data/processed/churn_scores_test.parquet")


if __name__ == "__main__":
    main()
