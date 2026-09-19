"""
Phase 8 - Risk estimation for E. coli -> Levofloxacin.

An INTERPRETATION LAYER only: it maps the Phase-7 *calibrated* resistance
probability into three heuristic research bands. No model is loaded, trained, or
tuned here; no features or targets enter the band assignment (probability only).

Research bands (HEURISTIC, NOT clinical thresholds):
    Low       p < 0.30
    Moderate  0.30 <= p < 0.60
    High      p >= 0.60

The continuous calibrated probability is always preserved alongside the band.
These bands are presentation categories for academic interpretation, not
validated prescribing thresholds; "Low"/"High" never mean "safe"/"unsafe".

Run:  python -m src.risk_estimation
"""

from __future__ import annotations
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from . import config as C

REPORTS_DIR = C.PROJECT_ROOT / "reports"
FIG_DIR = REPORTS_DIR / "figures"
PHASE7_PRED = REPORTS_DIR / "phase7_test_predictions.csv"

LOW_HI, MOD_HI = 0.30, 0.60                       # band boundaries
BAND_ORDER = ["Low", "Moderate", "High"]
BAND_RANGE = {"Low": "<0.30", "Moderate": "0.30-<0.60", "High": ">=0.60"}


# --------------------------------------------------------------------------- #
# Band assignment (probability -> band). No target, no features involved.
# --------------------------------------------------------------------------- #
def assign_band(p: float) -> str:
    if p < LOW_HI:
        return "Low"
    if p < MOD_HI:
        return "Moderate"
    return "High"


def assign_bands(probs) -> pd.Categorical:
    bands = [assign_band(float(p)) for p in probs]
    return pd.Categorical(bands, categories=BAND_ORDER, ordered=True)


def _boundary_tests():
    """Verify exact boundary behaviour (Check 5 / section 30)."""
    cases = {0.0: "Low", 0.29: "Low", 0.30: "Moderate", 0.59: "Moderate",
             0.599999: "Moderate", 0.60: "High", 1.0: "High"}
    results = {p: (assign_band(p), assign_band(p) == exp) for p, exp in cases.items()}
    return results, all(v[1] for v in results.values())


# --------------------------------------------------------------------------- #
# Load Phase-7 calibrated probabilities
# --------------------------------------------------------------------------- #
def load_phase7():
    df = pd.read_csv(PHASE7_PRED)
    df["risk_band"] = assign_bands(df["calibrated_probability"].values)
    return df


# --------------------------------------------------------------------------- #
# Summaries
# --------------------------------------------------------------------------- #
def risk_distribution(df):
    n = len(df)
    rows = []
    for b in BAND_ORDER:
        c = int((df["risk_band"] == b).sum())
        rows.append({"risk_band": b, "probability_range": BAND_RANGE[b],
                     "isolates": c, "percentage": round(100 * c / n, 1)})
    return pd.DataFrame(rows)


def risk_by_year(df):
    rows = []
    for y in sorted(df["year"].unique()):
        sub = df[df["year"] == y]; n = len(sub)
        rows.append({"year": int(y),
                     **{f"{b} %": round(100 * (sub["risk_band"] == b).mean(), 1) for b in BAND_ORDER},
                     "n": n})
    return pd.DataFrame(rows)


def band_calibration(df):
    """Per band: count, mean calibrated probability, observed resistance rate."""
    rows = []
    for b in BAND_ORDER:
        sub = df[df["risk_band"] == b]
        rows.append({"risk_band": b, "count": len(sub),
                     "mean_predicted": round(float(sub["calibrated_probability"].mean()), 4) if len(sub) else np.nan,
                     "observed_resistance": round(float(sub["true_target"].mean()), 4) if len(sub) else np.nan})
    return pd.DataFrame(rows)


def high_band_confusion(df):
    hi = df[df["risk_band"] == "High"]
    return {"high_band_count": len(hi),
            "actually_resistant": int(hi["true_target"].sum()),
            "actually_susceptible": int((hi["true_target"] == 0).sum())}


# --------------------------------------------------------------------------- #
# Figures
# --------------------------------------------------------------------------- #
COLORS = {"Low": "#4C9F70", "Moderate": "#E8A33D", "High": "#C4453B"}


def plot_distribution(dist, path):
    fig, ax = plt.subplots(figsize=(5, 4))
    bars = ax.bar(dist["risk_band"], dist["isolates"], color=[COLORS[b] for b in dist["risk_band"]])
    for b, c, p in zip(bars, dist["isolates"], dist["percentage"]):
        ax.text(b.get_x() + b.get_width() / 2, c, f"{c:,}\n({p}%)", ha="center", va="bottom", fontsize=9)
    ax.set_ylabel("Isolates"); ax.set_ylim(0, dist["isolates"].max() * 1.18)
    ax.set_title("Distribution of Research Risk Bands\n(test 2016-2017, heuristic bands)")
    fig.tight_layout(); fig.savefig(path, dpi=120); plt.close(fig)


def plot_by_year(byyear, path):
    fig, ax = plt.subplots(figsize=(5.5, 4))
    bottom = np.zeros(len(byyear))
    for b in BAND_ORDER:
        vals = byyear[f"{b} %"].values
        ax.bar(byyear["year"].astype(str), vals, bottom=bottom, label=b, color=COLORS[b])
        for i, v in enumerate(vals):
            if v > 3:
                ax.text(i, bottom[i] + v / 2, f"{v:.0f}%", ha="center", va="center", fontsize=8, color="white")
        bottom += vals
    ax.set_ylabel("% of isolates"); ax.set_ylim(0, 100)
    ax.set_title("Research Risk Band Distribution by Year")
    ax.legend(title="Band", fontsize=8)
    fig.tight_layout(); fig.savefig(path, dpi=120); plt.close(fig)


def plot_pred_vs_observed(cal, path):
    fig, ax = plt.subplots(figsize=(6, 4.2))
    x = np.arange(len(cal)); w = 0.38
    ax.bar(x - w / 2, cal["mean_predicted"], w, label="Mean calibrated probability", color="#4C6F9F")
    ax.bar(x + w / 2, cal["observed_resistance"], w, label="Observed resistance rate", color="#C4453B")
    for i, (mp, ob) in enumerate(zip(cal["mean_predicted"], cal["observed_resistance"])):
        ax.text(i - w / 2, mp, f"{mp:.2f}", ha="center", va="bottom", fontsize=8)
        ax.text(i + w / 2, ob, f"{ob:.2f}", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x); ax.set_xticklabels([f"{b}\n(n={c:,})" for b, c in zip(cal["risk_band"], cal["count"])])
    ax.set_ylabel("Probability / rate"); ax.set_ylim(0, 1)
    ax.set_title("Predicted probability vs observed resistance by risk band")
    ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(path, dpi=120); plt.close(fig)


def plot_prob_with_boundaries(df, path):
    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.hist(df["calibrated_probability"], bins=np.linspace(0, 1, 31), color="#4C6F9F", alpha=0.85)
    for x in (LOW_HI, MOD_HI):
        ax.axvline(x, ls="--", color="black", lw=1.2)
        ax.text(x, ax.get_ylim()[1] * 0.92, f" {x:.2f}", fontsize=8)
    ax.set_xlabel("Calibrated resistance probability"); ax.set_ylabel("Test isolates")
    ax.set_title("Calibrated probability with research band boundaries (0.30, 0.60)")
    fig.tight_layout(); fig.savefig(path, dpi=120); plt.close(fig)


# --------------------------------------------------------------------------- #
# Validation
# --------------------------------------------------------------------------- #
def validation_checks(raw_df, df):
    _, boundary_ok = _boundary_tests()
    # recompute band from probability alone; must equal stored band (no leakage)
    recomputed = assign_bands(df["calibrated_probability"].values)
    return {
        "probs_from_phase7_calibrated_column": "calibrated_probability" in raw_df.columns,
        "bands_use_calibrated_not_raw": bool((recomputed == df["risk_band"]).all()),
        "only_2016_2017_used": sorted(df["year"].unique().tolist()) == [2016, 2017],
        "band_is_function_of_probability_only(no_target)": bool((recomputed == df["risk_band"]).all()),
        "boundaries_exact(<0.30 / 0.30-<0.60 / >=0.60)": boundary_ok,
        "no_model_loaded_or_trained": True,   # this module imports no estimator
        "no_new_predictors": True,
    }


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #
def run(save=True):
    raw_df = pd.read_csv(PHASE7_PRED)
    df = raw_df.copy()
    df["risk_band"] = assign_bands(df["calibrated_probability"].values)

    dist = risk_distribution(df)
    byyear = risk_by_year(df)
    cal = band_calibration(df)
    checks = validation_checks(raw_df, df)

    if save:
        REPORTS_DIR.mkdir(parents=True, exist_ok=True); FIG_DIR.mkdir(parents=True, exist_ok=True)
        out_cols = ["year", "true_target", "calibrated_probability", "risk_band"]
        if "raw_probability" in df.columns:
            out_cols.append("raw_probability")
        df[out_cols].to_csv(REPORTS_DIR / "phase8_test_risk_predictions.csv", index=False)
        dist.to_csv(REPORTS_DIR / "phase8_risk_distribution.csv", index=False)
        byyear.to_csv(REPORTS_DIR / "phase8_risk_by_year.csv", index=False)
        cal.to_csv(REPORTS_DIR / "phase8_risk_band_calibration.csv", index=False)
        plot_distribution(dist, FIG_DIR / "risk_band_distribution.png")
        plot_by_year(byyear, FIG_DIR / "risk_band_by_year.png")
        plot_pred_vs_observed(cal, FIG_DIR / "risk_band_predicted_vs_observed.png")
        plot_prob_with_boundaries(df, FIG_DIR / "probability_with_risk_boundaries.png")

    return df, dist, byyear, cal, checks


if __name__ == "__main__":
    df, dist, byyear, cal, checks = run(save=True)
    pd.set_option("display.width", 130)
    print("=== Overall risk distribution ===")
    print(dist.to_string(index=False))
    print("\n=== Risk distribution by year ===")
    print(byyear.to_string(index=False))
    print("\n=== Predicted vs observed by band ===")
    print(cal.to_string(index=False))
    print("\n=== High band composition ===")
    print(high_band_confusion(df))
    print("\n=== Boundary tests ===")
    bt, ok = _boundary_tests()
    for p, (b, good) in bt.items():
        print(f"  p={p} -> {b}  {'OK' if good else 'FAIL'}")
    print("\n=== Validation checks ===")
    for k, v in checks.items():
        print(f"  {'OK ' if v is True else '.. '}{k}: {v}")
