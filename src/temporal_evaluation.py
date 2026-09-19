"""
Phase 4 - Temporal (expanding-window) evaluation for E. coli -> Levofloxacin.

Question: does the model keep its predictive performance on later years?

Design: for each test year t in 2011..2017, train on ALL years < t and test on t
only. No shuffling. Preprocessing (impute + one-hot) is fit on the training years
only, inside each pipeline. Model hyperparameters + class-weighting strategy are
the LOCKED Phase 3 ones (build_models is reused verbatim); nothing is tuned per
year. The class-weighting *rule* (balanced / neg-over-pos) recomputes on each
training window - that is the same rule as Phase 3, not per-year tuning.

Outputs:
    reports/phase4_temporal_metrics.csv       (long format)
    reports/phase4_temporal_predictions.parquet
    reports/figures/temporal_{roc_auc,pr_auc,recall,brier,resistance_prevalence}.png

Run:  python -m src.temporal_evaluation
"""

from __future__ import annotations
import warnings

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from . import config as C
from .modeling import load_modeling_frame, predictor_columns, build_models, evaluate, PLOT_ORDER

FIG_DIR = C.PROJECT_ROOT / "reports" / "figures"
REPORTS_DIR = C.PROJECT_ROOT / "reports"
TEST_YEARS = list(range(2011, 2018))          # 2011..2017
EARLY = [2011, 2012, 2013]
LATE = [2015, 2016, 2017]


# --------------------------------------------------------------------------- #
# Splits
# --------------------------------------------------------------------------- #
def expanding_window_splits(df, test_years=TEST_YEARS):
    """Yield (test_year, train_mask, test_mask): train = all years < t, test = t."""
    for t in test_years:
        yield t, (df[C.TIME_COL] < t), (df[C.TIME_COL] == t)


# --------------------------------------------------------------------------- #
# Core loop
# --------------------------------------------------------------------------- #
def run_temporal_evaluation(save=True, test_years=TEST_YEARS):
    """
    Resumable expanding-window loop. Each test year is computed independently and
    its results written to per-year part files, so a long run can be resumed
    (finished years are skipped). Call finalize() afterwards to assemble.
    """
    df = load_modeling_frame()
    preds = predictor_columns(df)
    PARTS = REPORTS_DIR / "phase4_parts"
    PARTS.mkdir(parents=True, exist_ok=True)

    for t, tr, te in expanding_window_splits(df, test_years):
        part_metrics = PARTS / f"metrics_{t}.csv"
        part_preds = PARTS / f"pred_{t}.parquet"
        if part_metrics.exists() and part_preds.exists():
            print(f"[skip] {t} already done", flush=True)
            continue

        Xtr, ytr = df.loc[tr, preds], df.loc[tr, "target"].values
        Xte, yte = df.loc[te, preds], df.loc[te, "target"].values
        models = build_models(ytr, preds)          # LOCKED Phase 3 configs, refit

        rows, pframes = [], []
        for name, pipe in models.items():
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                pipe.fit(Xtr, ytr)
            m = evaluate(pipe, Xte, yte)
            rows.append({
                "model": name, "test_year": t, "n_test": len(yte),
                "resistance_rate": float(yte.mean()),
                "ROC-AUC": m["ROC-AUC"], "PR-AUC": m["PR-AUC"],
                "precision": m["Precision"], "recall": m["Recall"],
                "F1": m["F1"], "Brier": m["Brier"], "accuracy": m["Accuracy"],
            })
            pframes.append(pd.DataFrame({
                "model": name, "test_year": t, "true_target": yte,
                "predicted_class": m["_pred"], "resistance_probability": m["_proba"]}))
        pd.DataFrame(rows).to_csv(part_metrics, index=False)
        pd.concat(pframes, ignore_index=True).to_parquet(part_preds, index=False)
        print(f"[done] {t}: {rows[-1]['n_test']} test rows", flush=True)

    return finalize(save=save)


def finalize(save=True):
    """Assemble per-year parts into the final metrics CSV + predictions parquet."""
    PARTS = REPORTS_DIR / "phase4_parts"
    mfiles = sorted(PARTS.glob("metrics_*.csv"))
    pfiles = sorted(PARTS.glob("pred_*.parquet"))
    metrics = pd.concat([pd.read_csv(f) for f in mfiles], ignore_index=True)
    predictions = pd.concat([pd.read_parquet(f) for f in pfiles], ignore_index=True)
    metrics = metrics.sort_values(["model", "test_year"]).reset_index(drop=True)
    if save:
        metrics.to_csv(REPORTS_DIR / "phase4_temporal_metrics.csv", index=False)
        predictions.to_parquet(REPORTS_DIR / "phase4_temporal_predictions.parquet", index=False)
    return metrics, predictions


# --------------------------------------------------------------------------- #
# Figures
# --------------------------------------------------------------------------- #
def _plot_metric(metrics, col, title, path, lower_better=False):
    fig, ax = plt.subplots(figsize=(8, 4.6))
    for name in PLOT_ORDER:
        s = metrics[metrics["model"] == name].sort_values("test_year")
        ax.plot(s["test_year"], s[col], "-o", lw=2, label=name)
    ax.set_xlabel("Test year"); ax.set_ylabel(col)
    ax.set_title(title + ("  (lower is better)" if lower_better else ""))
    ax.set_xticks(TEST_YEARS); ax.grid(alpha=0.3); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(path, dpi=120); plt.close(fig)


def _plot_prevalence(metrics, path):
    prev = (metrics[["test_year", "resistance_rate", "n_test"]]
            .drop_duplicates("test_year").sort_values("test_year"))
    fig, ax = plt.subplots(figsize=(8, 4.6))
    ax.plot(prev["test_year"], prev["resistance_rate"] * 100, "-o", color="#C4453B", lw=2)
    for _, r in prev.iterrows():
        ax.annotate(f"n={int(r['n_test']):,}", (r["test_year"], r["resistance_rate"] * 100),
                    textcoords="offset points", xytext=(0, 7), ha="center", fontsize=7)
    ax.set_xlabel("Test year"); ax.set_ylabel("Resistance prevalence %")
    ax.set_title("Resistance prevalence by test year (E. coli - Levofloxacin)")
    ax.set_xticks(TEST_YEARS); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(path, dpi=120); plt.close(fig)


def make_figures(metrics):
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    _plot_metric(metrics, "ROC-AUC", "Temporal Model Performance - ROC-AUC", FIG_DIR / "temporal_roc_auc.png")
    _plot_metric(metrics, "PR-AUC", "Temporal Model Performance - PR-AUC", FIG_DIR / "temporal_pr_auc.png")
    _plot_metric(metrics, "recall", "Temporal Model Performance - Recall", FIG_DIR / "temporal_recall.png")
    _plot_metric(metrics, "Brier", "Temporal Probability Performance - Brier Score",
                 FIG_DIR / "temporal_brier.png", lower_better=True)
    _plot_prevalence(metrics, FIG_DIR / "temporal_resistance_prevalence.png")


# --------------------------------------------------------------------------- #
# Summaries: early vs late, and performance-vs-prevalence association
# --------------------------------------------------------------------------- #
def early_vs_late(metrics):
    rows = []
    for name in PLOT_ORDER:
        s = metrics[metrics["model"] == name]
        e = s[s["test_year"].isin(EARLY)].mean(numeric_only=True)
        l = s[s["test_year"].isin(LATE)].mean(numeric_only=True)
        rows.append({"model": name, **{f"{k}_early": e[k] for k in ["ROC-AUC", "PR-AUC", "recall", "Brier"]},
                     **{f"{k}_late": l[k] for k in ["ROC-AUC", "PR-AUC", "recall", "Brier"]},
                     **{f"{k}_delta": l[k] - e[k] for k in ["ROC-AUC", "PR-AUC", "recall", "Brier"]}})
    return pd.DataFrame(rows)


def prevalence_association(metrics):
    """Pearson r between yearly resistance prevalence and each metric (association only)."""
    rows = []
    for name in PLOT_ORDER:
        s = metrics[metrics["model"] == name].sort_values("test_year")
        for col in ["ROC-AUC", "PR-AUC", "Brier"]:
            r = np.corrcoef(s["resistance_rate"], s[col])[0, 1]
            rows.append({"model": name, "vs_metric": col, "pearson_r": round(r, 3)})
    return pd.DataFrame(rows)


def print_report(metrics):
    pd.set_option("display.width", 160)
    for name in PLOT_ORDER:
        s = metrics[metrics["model"] == name].sort_values("test_year")
        print(f"\n=== {name} ===")
        print(s[["test_year", "n_test", "resistance_rate", "ROC-AUC", "PR-AUC",
                 "recall", "F1", "Brier"]]
              .assign(resistance_rate=lambda x: (x["resistance_rate"] * 100).round(1))
              .round(3).to_string(index=False))
    print("\n=== EARLY (2011-13) vs LATE (2015-17): late - early ===")
    print(early_vs_late(metrics).round(3).to_string(index=False))
    print("\n=== Prevalence vs performance (Pearson r, association only) ===")
    print(prevalence_association(metrics).to_string(index=False))


if __name__ == "__main__":
    metrics, _ = run_temporal_evaluation(save=True)
    make_figures(metrics)
    print_report(metrics)
    print(f"\nSaved: reports/phase4_temporal_metrics.csv, "
          f"reports/phase4_temporal_predictions.parquet, 5 figures in reports/figures/")
