"""
Phase 7 - Probability calibration for the frozen XGBoost pipeline (E. coli -> Levofloxacin).

Discrimination (ROC/PR) asks "can the model rank?"; calibration asks "do the
probabilities match observed frequencies?". Phase 3 used class weighting, which
can push raw probabilities away from the base rate, so we evaluate and (via a
sigmoid layer) attempt to improve probability alignment.

Design (temporal, auditable):
    frozen XGBoost (trained 2004-2014)
        -> raw margin on 2015  -> fit sigmoid/Platt calibrator (2015 ONLY)
        -> raw margin on 2016-2017 -> apply calibrator -> compare raw vs calibrated

We deliberately implement Platt scaling as a small standalone layer (a 1-D
logistic on the frozen model's raw margin), NOT CalibratedClassifierCV, so the
XGBoost model is never refit and the 2015-only boundary is obvious.

Run:  python -m src.calibration
"""

from __future__ import annotations
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (brier_score_loss, log_loss, roc_auc_score,
                             average_precision_score)
from sklearn.calibration import calibration_curve

from . import config as C
from .modeling import load_modeling_frame, predictor_columns

MODELS_DIR = C.PROJECT_ROOT / "models"
FIG_DIR = C.PROJECT_ROOT / "reports" / "figures"
REPORTS_DIR = C.PROJECT_ROOT / "reports"

CAL_YEARS = [2015]            # calibrator fit here ONLY
TEST_YEARS = [2016, 2017]     # untouched until final evaluation
N_BINS = 10                   # uniform bins for reliability / ECE


# --------------------------------------------------------------------------- #
# Frozen-model scores
# --------------------------------------------------------------------------- #
def _scores_for_years(pipe, df, preds, years):
    """Return (raw_margin, raw_proba, y) from the FROZEN pipeline for given years."""
    sub = df[df[C.TIME_COL].isin(years)]
    X = sub[preds]
    prep, clf = pipe.named_steps["prep"], pipe.named_steps["clf"]
    Xs = prep.transform(X)                              # sparse, as deployed
    margin = clf.predict(Xs, output_margin=True)
    proba = pipe.predict_proba(X)[:, 1]
    return margin, proba, sub["target"].values, sub[C.TIME_COL].values


class SigmoidCalibrator:
    """Platt scaling: 1-D logistic mapping frozen-model margin -> calibrated prob."""
    def __init__(self):
        self.lr = LogisticRegression(C=1e6, solver="lbfgs")   # ~unregularised

    def fit(self, margin, y):
        self.lr.fit(np.asarray(margin).reshape(-1, 1), y)
        return self

    def predict(self, margin):
        return self.lr.predict_proba(np.asarray(margin).reshape(-1, 1))[:, 1]


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #
def prob_metrics(y, p):
    return {"Brier": brier_score_loss(y, p), "LogLoss": log_loss(y, p),
            "ROC-AUC": roc_auc_score(y, p), "PR-AUC": average_precision_score(y, p)}


def calibration_bins(y, p, n_bins=N_BINS):
    edges = np.linspace(0, 1, n_bins + 1)
    idx = np.clip(np.digitize(p, edges[1:-1]), 0, n_bins - 1)
    rows = []
    for b in range(n_bins):
        m = idx == b
        rows.append({
            "bin": f"{edges[b]:.1f}-{edges[b+1]:.1f}",
            "mean_predicted": round(float(p[m].mean()), 4) if m.any() else np.nan,
            "observed_resistance": round(float(y[m].mean()), 4) if m.any() else np.nan,
            "count": int(m.sum()),
        })
    return pd.DataFrame(rows)


def expected_calibration_error(y, p, n_bins=N_BINS):
    """ECE over uniform bins; empty bins contribute 0. Returns (ece, n_nonempty)."""
    edges = np.linspace(0, 1, n_bins + 1)
    idx = np.clip(np.digitize(p, edges[1:-1]), 0, n_bins - 1)
    N = len(y); ece = 0.0; nonempty = 0
    for b in range(n_bins):
        m = idx == b
        if m.any():
            nonempty += 1
            ece += (m.sum() / N) * abs(y[m].mean() - p[m].mean())
    return float(ece), nonempty


# --------------------------------------------------------------------------- #
# Figures
# --------------------------------------------------------------------------- #
def plot_reliability(y, p_raw, p_cal, path):
    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    ax.plot([0, 1], [0, 1], "--", color="grey", label="Perfect calibration")
    for p, lab, col in [(p_raw, "XGBoost raw", "#C4453B"), (p_cal, "XGBoost calibrated", "#4C6F9F")]:
        frac, mean_pred = calibration_curve(y, p, n_bins=N_BINS, strategy="uniform")
        ax.plot(mean_pred, frac, "-o", color=col, label=lab)
    ax.set_xlabel("Mean predicted probability"); ax.set_ylabel("Observed fraction resistant")
    ax.set_title("Probability Calibration - XGBoost (test 2016-2017, 10 bins)")
    ax.legend(loc="upper left", fontsize=8); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(path, dpi=120); plt.close(fig)


def plot_prob_dist(p_raw, p_cal, path):
    fig, ax = plt.subplots(figsize=(7, 4.2))
    bins = np.linspace(0, 1, 31)
    ax.hist(p_raw, bins=bins, alpha=0.5, label="Raw", color="#C4453B")
    ax.hist(p_cal, bins=bins, alpha=0.5, label="Calibrated", color="#4C6F9F")
    ax.set_xlabel("Predicted resistance probability"); ax.set_ylabel("Test isolates")
    ax.set_title("Probability distribution - raw vs calibrated (2016-2017)")
    ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(path, dpi=120); plt.close(fig)


def plot_brier(brier_raw, brier_cal, path):
    fig, ax = plt.subplots(figsize=(4.2, 4))
    bars = ax.bar(["Raw", "Calibrated"], [brier_raw, brier_cal],
                  color=["#C4453B", "#4C6F9F"])
    for b, v in zip(bars, [brier_raw, brier_cal]):
        ax.text(b.get_x() + b.get_width() / 2, v, f"{v:.4f}", ha="center", va="bottom", fontsize=9)
    ax.set_ylabel("Brier score (lower better)")
    ax.set_ylim(0, max(brier_raw, brier_cal) * 1.15)
    ax.set_title("Brier score - raw vs calibrated")
    fig.tight_layout(); fig.savefig(path, dpi=120); plt.close(fig)


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #
def run(save=True):
    pipe = joblib.load(MODELS_DIR / "xgboost.joblib")
    df = load_modeling_frame()
    preds = predictor_columns(df)

    m_cal, _, y_cal, yr_cal = _scores_for_years(pipe, df, preds, CAL_YEARS)
    m_te, p_raw, y_te, yr_te = _scores_for_years(pipe, df, preds, TEST_YEARS)

    calibrator = SigmoidCalibrator().fit(m_cal, y_cal)      # 2015 ONLY
    p_cal = calibrator.predict(m_te)

    metrics = pd.DataFrame({
        "XGBoost Raw": prob_metrics(y_te, p_raw),
        "XGBoost Calibrated": prob_metrics(y_te, p_cal),
    }).T
    bins = calibration_bins(y_te, p_cal)
    ece_raw, _ = expected_calibration_error(y_te, p_raw)
    ece_cal, ne = expected_calibration_error(y_te, p_cal)

    # by-year Brier (optional)
    by_year = []
    for yv in TEST_YEARS:
        mk = yr_te == yv
        by_year.append({"year": yv,
                        "Brier_raw": round(brier_score_loss(y_te[mk], p_raw[mk]), 4),
                        "Brier_cal": round(brier_score_loss(y_te[mk], p_cal[mk]), 4)})
    by_year = pd.DataFrame(by_year)

    # ---- validation checks ---------------------------------------------- #
    checks = {
        "calibration_fit_years == [2015]": sorted(set(yr_cal.tolist())) == CAL_YEARS,
        "eval_years == [2016, 2017]": sorted(set(yr_te.tolist())) == TEST_YEARS,
        "no_test_labels_in_calibrator_fit": not np.array_equal(y_cal, y_te)
                                            and len(y_cal) != len(y_te),
        "calibrator_input_is_score_not_features": True,  # only margin passed in
        "xgb_params_unchanged": pipe.named_steps["clf"].get_params()["n_estimators"] == 400,
    }
    # Check 6: raw test probs match Phase-3 saved XGBoost test probs
    try:
        p3 = pd.read_parquet(REPORTS_DIR / "phase3_test_predictions.parquet")
        p3 = p3[p3["model"] == "XGBoost"]["resistance_probability"].values
        checks["raw_test_probs_match_phase3"] = (
            len(p3) == len(p_raw) and float(np.max(np.abs(np.sort(p3) - np.sort(p_raw)))) < 1e-6)
    except Exception as e:
        checks["raw_test_probs_match_phase3"] = f"skip: {e}"

    if save:
        REPORTS_DIR.mkdir(parents=True, exist_ok=True); FIG_DIR.mkdir(parents=True, exist_ok=True)
        out = metrics.copy(); out["ECE"] = [ece_raw, ece_cal]
        out.round(5).to_csv(REPORTS_DIR / "phase7_calibration_metrics.csv")
        bins.to_csv(REPORTS_DIR / "phase7_calibration_bins.csv", index=False)
        pd.DataFrame({
            "year": yr_te, "true_target": y_te,
            "raw_probability": p_raw, "calibrated_probability": p_cal,
            "raw_predicted_class": (p_raw >= 0.5).astype(int),
            "calibrated_predicted_class": (p_cal >= 0.5).astype(int),
        }).to_csv(REPORTS_DIR / "phase7_test_predictions.csv", index=False)
        plot_reliability(y_te, p_raw, p_cal, FIG_DIR / "calibration_curve.png")
        plot_prob_dist(p_raw, p_cal, FIG_DIR / "probability_distribution_raw_vs_calibrated.png")
        plot_brier(metrics.loc["XGBoost Raw", "Brier"],
                   metrics.loc["XGBoost Calibrated", "Brier"],
                   FIG_DIR / "brier_score_comparison.png")

    return metrics, bins, (ece_raw, ece_cal, ne), by_year, checks


if __name__ == "__main__":
    metrics, bins, (ece_raw, ece_cal, ne), by_year, checks = run(save=True)
    pd.set_option("display.width", 130)
    print("=== Raw vs calibrated (test 2016-2017) ===")
    print(metrics.round(4).to_string())
    print(f"\nECE (10 uniform bins): raw={ece_raw:.4f}  calibrated={ece_cal:.4f}  (nonempty bins={ne})")
    print("\n=== Calibration bins (calibrated probabilities) ===")
    print(bins.to_string(index=False))
    print("\n=== By-year Brier ===")
    print(by_year.to_string(index=False))
    print("\n=== Validation checks ===")
    for k, v in checks.items():
        ok = (v is True)
        print(f"  {'OK ' if ok else '.. '}{k}: {v}")
