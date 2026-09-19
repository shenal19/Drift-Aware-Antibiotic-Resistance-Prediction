"""
Phase 5 - Population Stability Index (PSI) drift detection for E. coli -> Levofloxacin.

Measures whether the INPUT feature distributions shift over time (data drift),
independent of model performance (Phase 4). Real ATLAS data only; nothing is
synthesised or manipulated.

PSI (categorical):
    PSI = Σ_bins (actual% - expected%) * ln(actual% / expected%)
    expected% = reference-period proportion, actual% = later-year proportion.
    Zero proportions are floored at EPS (documented) to avoid log(0).

Category handling (consistent with the modelling pipeline's OHE min_frequency):
    - Missing values -> an explicit "Missing" category (as the imputer does).
    - Categories with reference count < MIN_COUNT are collapsed into "Other".
    - Any future-year category not seen (or rare) in the reference maps to
      "Other" too, so future-only categories are REPRESENTED, never discarded.

Heuristic thresholds (NOT clinical): PSI<0.10 little shift, 0.10-0.25 moderate,
>=0.25 large.

Run:  python -m src.psi_drift
"""

from __future__ import annotations
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from . import config as C
from .modeling import load_modeling_frame, predictor_columns

FIG_DIR = C.PROJECT_ROOT / "reports" / "figures"
REPORTS_DIR = C.PROJECT_ROOT / "reports"

REFERENCE_YEARS = [2011, 2012, 2013]
FUTURE_YEARS = [2014, 2015, 2016, 2017]
MIN_COUNT = C.OHE_MIN_FREQUENCY          # 20 - same rare-category cutoff as modelling
EPS = 1e-6                               # zero-proportion floor (documented)


# --------------------------------------------------------------------------- #
# Category preparation
# --------------------------------------------------------------------------- #
def _prep(series: pd.Series) -> pd.Series:
    """Missing -> 'Missing'; everything as string (matches the imputer)."""
    return series.astype(object).where(pd.notna(series), "Missing").astype(str)


def _kept_categories(ref_series: pd.Series) -> set:
    """Categories frequent enough in the REFERENCE period to keep as their own bin."""
    vc = ref_series.value_counts()
    return set(vc[vc >= MIN_COUNT].index)


def _binned_proportions(series: pd.Series, kept: set, bins: list) -> pd.Series:
    """Collapse anything not in `kept` to 'Other' and return proportions over `bins`."""
    mapped = series.where(series.isin(kept), "Other")
    props = mapped.value_counts(normalize=True)
    return props.reindex(bins, fill_value=0.0)


# --------------------------------------------------------------------------- #
# PSI
# --------------------------------------------------------------------------- #
def calculate_categorical_psi(reference: pd.Series, current: pd.Series):
    """PSI between a reference and a current categorical series. Returns (psi, detail_df)."""
    ref, cur = _prep(reference), _prep(current)
    kept = _kept_categories(ref)
    bins = sorted(kept) + ["Other"]
    exp = _binned_proportions(ref, kept, bins)
    act = _binned_proportions(cur, kept, bins)
    exp_f = exp.clip(lower=EPS)
    act_f = act.clip(lower=EPS)
    contrib = (act_f - exp_f) * np.log(act_f / exp_f)
    detail = pd.DataFrame({"expected%": exp * 100, "actual%": act * 100,
                           "psi_contrib": contrib})
    return float(contrib.sum()), detail


def calculate_all_psi(ref_df: pd.DataFrame, cur_df: pd.DataFrame, features: list) -> dict:
    return {f: calculate_categorical_psi(ref_df[f], cur_df[f])[0] for f in features}


def run_temporal_psi(df, reference_years=REFERENCE_YEARS, future_years=FUTURE_YEARS, features=None):
    """PSI table: rows=features, cols=future years, comparing each year to the reference."""
    features = features or predictor_columns(df)
    ref = df[df[C.TIME_COL].isin(reference_years)]
    out = {}
    for y in future_years:
        cur = df[df[C.TIME_COL] == y]
        out[y] = calculate_all_psi(ref, cur, features)
    return pd.DataFrame(out).reindex(features)   # feature x year


# --------------------------------------------------------------------------- #
# Summaries
# --------------------------------------------------------------------------- #
def summarize_drift(psi_df: pd.DataFrame) -> pd.DataFrame:
    """Per-year summary indicators (mean is a summary only, NOT a validated score)."""
    rows = []
    for y in psi_df.columns:
        col = psi_df[y]
        rows.append({
            "Year": y,
            "Mean PSI": round(col.mean(), 3),
            "Median PSI": round(col.median(), 3),
            "Max PSI": round(col.max(), 3),
            "Features >=0.10": int((col >= C.PSI_THRESHOLDS["low"]).sum()),
            "Features >=0.25": int((col >= C.PSI_THRESHOLDS["moderate"]).sum()),
        })
    return pd.DataFrame(rows)


def country_shift(df, reference_years=REFERENCE_YEARS, future_year=2017, top=10):
    """Reference vs future-year proportions for Country, biggest movers first."""
    ref = _prep(df[df[C.TIME_COL].isin(reference_years)]["Country"])
    cur = _prep(df[df[C.TIME_COL] == future_year]["Country"])
    kept = _kept_categories(ref)
    bins = sorted(kept) + ["Other"]
    tbl = pd.DataFrame({
        "Reference %": (_binned_proportions(ref, kept, bins) * 100).round(1),
        f"{future_year} %": (_binned_proportions(cur, kept, bins) * 100).round(1),
    })
    tbl["Change (pp)"] = (tbl[f"{future_year} %"] - tbl["Reference %"]).round(1)
    return tbl.reindex(tbl["Change (pp)"].abs().sort_values(ascending=False).index).head(top)


def link_to_phase4(psi_df, model="XGBoost"):
    """Combine mean PSI with Phase-4 performance (association view, no causality)."""
    p4 = pd.read_csv(REPORTS_DIR / "phase4_temporal_metrics.csv")
    p4 = p4[(p4["model"] == model) & (p4["test_year"].isin(psi_df.columns))]
    mean_psi = psi_df.mean(axis=0)
    rows = []
    for y in psi_df.columns:
        r = p4[p4["test_year"] == y].iloc[0]
        rows.append({"Year": y, "Mean PSI": round(mean_psi[y], 3),
                     f"{model} ROC-AUC": round(r["ROC-AUC"], 3),
                     f"{model} PR-AUC": round(r["PR-AUC"], 3),
                     f"{model} Recall": round(r["recall"], 3),
                     f"{model} Brier": round(r["Brier"], 3)})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Figures
# --------------------------------------------------------------------------- #
def plot_psi_heatmap(psi_df, path):
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    im = ax.imshow(psi_df.values, cmap="OrRd", vmin=0, vmax=max(0.25, psi_df.values.max()))
    ax.set_xticks(range(len(psi_df.columns))); ax.set_xticklabels(psi_df.columns)
    ax.set_yticks(range(len(psi_df.index))); ax.set_yticklabels(psi_df.index)
    for i in range(psi_df.shape[0]):
        for j in range(psi_df.shape[1]):
            v = psi_df.values[i, j]
            ax.text(j, i, f"{v:.3f}", ha="center", va="center",
                    color="white" if v > 0.18 else "black", fontsize=8)
    ax.set_title("Temporal Feature Drift - PSI (reference 2011-2013)")
    fig.colorbar(im, fraction=0.046, pad=0.04, label="PSI")
    fig.tight_layout(); fig.savefig(path, dpi=120); plt.close(fig)


def plot_mean_psi(psi_df, path):
    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.plot(psi_df.columns, psi_df.mean(axis=0), "-o", label="Mean PSI", lw=2)
    ax.plot(psi_df.columns, psi_df.median(axis=0), "--s", label="Median PSI", lw=1.5)
    for thr, lab in [(0.10, "0.10 moderate"), (0.25, "0.25 large")]:
        ax.axhline(thr, ls=":", color="grey", lw=1)
        ax.text(psi_df.columns[0], thr + 0.005, lab, fontsize=7, color="grey")
    ax.set_xlabel("Test year"); ax.set_ylabel("PSI (summary across 7 features)")
    ax.set_title("Aggregate feature drift over time (summary indicator only)")
    ax.set_xticks(list(psi_df.columns)); ax.legend(fontsize=8); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(path, dpi=120); plt.close(fig)


def plot_psi_by_feature(psi_df, year, path):
    s = psi_df[year].sort_values()
    fig, ax = plt.subplots(figsize=(7, 4.2))
    colors = ["#C4453B" if v >= 0.25 else "#E8A33D" if v >= 0.10 else "#4C9F70" for v in s]
    ax.barh(s.index, s.values, color=colors)
    for thr in (0.10, 0.25):
        ax.axvline(thr, ls=":", color="grey", lw=1)
    for i, v in enumerate(s.values):
        ax.text(v + 0.003, i, f"{v:.3f}", va="center", fontsize=8)
    ax.set_xlabel("PSI"); ax.set_title(f"Feature-level drift in {year} (vs 2011-2013 reference)")
    fig.tight_layout(); fig.savefig(path, dpi=120); plt.close(fig)


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #
def run(save=True):
    df = load_modeling_frame()
    features = predictor_columns(df)
    psi_df = run_temporal_psi(df, features=features)
    summary = summarize_drift(psi_df)
    if save:
        REPORTS_DIR.mkdir(parents=True, exist_ok=True)
        FIG_DIR.mkdir(parents=True, exist_ok=True)
        psi_df.round(4).to_csv(REPORTS_DIR / "phase5_psi.csv")
        summary.to_csv(REPORTS_DIR / "phase5_drift_summary.csv", index=False)
        plot_psi_heatmap(psi_df, FIG_DIR / "psi_heatmap.png")
        plot_mean_psi(psi_df, FIG_DIR / "mean_psi_over_time.png")
        plot_psi_by_feature(psi_df, 2017, FIG_DIR / "psi_by_feature_2017.png")
    return df, psi_df, summary


if __name__ == "__main__":
    df, psi_df, summary = run(save=True)
    pd.set_option("display.width", 140)
    print("=== PSI table (feature x year) ===")
    print(psi_df.round(3).to_string())
    print("\n=== Drift summary ===")
    print(summary.to_string(index=False))
    print("\n=== Country: reference vs 2017 (biggest movers) ===")
    print(country_shift(df).to_string())
    print("\n=== Mean PSI vs Phase-4 XGBoost performance (association only) ===")
    print(link_to_phase4(psi_df).to_string(index=False))
