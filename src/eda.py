"""
Phase 2 - Exploratory Data Analysis for E. coli -> Levofloxacin.

Deliberately small: four figures, each tied to the research question
(temporal reliability of a resistance model). No modelling here.

Figures written to reports/figures/:
    1. class_balance.png          - overall Susceptible vs Resistant
    2. temporal_resistance.png    - resistance % by year (95% CI) + isolate count/year
    3. resistance_by_subgroup.png - resistance % by gender / age / country / source
    4. input_shift_over_time.png  - do the INPUT distributions move across years?
                                    (the "data drift" half of the research question)

Run:  python -m src.eda
"""

from __future__ import annotations
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from . import config as C

FIG_DIR = C.PROJECT_ROOT / "reports" / "figures"
AGE_ORDER = ["0 to 2 Years", "3 to 12 Years", "13 to 18 Years",
             "19 to 64 Years", "65 to 84 Years", "85 and Over", "Unknown"]

plt.rcParams.update({"figure.dpi": 120, "axes.grid": True,
                     "grid.alpha": 0.3, "axes.spines.top": False,
                     "axes.spines.right": False, "font.size": 10})


def load_processed() -> pd.DataFrame:
    path = C.PROCESSED_DIR / f"{C.SPECIES.split()[0].lower()}_{C.ANTIBIOTIC.lower()}.parquet"
    return pd.read_parquet(path).astype({"Year": int})


def _wilson_ci(k, n, z=1.96):
    """95% Wilson interval for a proportion (better than normal approx at small n)."""
    if n == 0:
        return (np.nan, np.nan)
    p = k / n
    denom = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denom
    half = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return (centre - half, centre + half)


# --------------------------------------------------------------------------- #
# Summary tables (printed / returned)
# --------------------------------------------------------------------------- #
def summaries(d: pd.DataFrame) -> dict:
    out = {}
    out["overall"] = d["target"].value_counts().rename({0: "Susceptible", 1: "Resistant"})

    per_year = d.groupby("Year")["target"].agg(["count", "sum", "mean"])
    per_year.columns = ["n", "n_resistant", "resist_rate"]
    ci = per_year.apply(lambda r: _wilson_ci(r["n_resistant"], r["n"]), axis=1)
    per_year["ci_low"] = [c[0] for c in ci]
    per_year["ci_high"] = [c[1] for c in ci]
    out["per_year"] = per_year

    def grp(col, min_n=1):
        g = d.groupby(col)["target"].agg(["count", "mean"])
        g.columns = ["n", "resist_rate"]
        return g[g["n"] >= min_n].sort_values("resist_rate", ascending=False)

    out["by_gender"] = grp("Gender")
    out["by_age"] = grp("Age Group").reindex(AGE_ORDER)
    out["by_country"] = grp("Country", min_n=500).sort_values("n", ascending=False).head(10)
    out["by_source"] = grp("Source", min_n=500).sort_values("n", ascending=False).head(10)
    return out


# --------------------------------------------------------------------------- #
# Figures
# --------------------------------------------------------------------------- #
def fig_class_balance(s, path):
    fig, ax = plt.subplots(figsize=(4.2, 3.4))
    vals = [s["overall"].get("Susceptible", 0), s["overall"].get("Resistant", 0)]
    bars = ax.bar(["Susceptible (0)", "Resistant (1)"], vals,
                  color=["#4C9F70", "#C4453B"])
    total = sum(vals)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v, f"{v:,}\n({v/total:.1%})",
                ha="center", va="bottom", fontsize=9)
    ax.set_ylabel("Isolates")
    ax.set_title("Overall class balance\nE. coli - Levofloxacin")
    ax.set_ylim(0, max(vals) * 1.18)
    fig.tight_layout(); fig.savefig(path); plt.close(fig)


def fig_temporal(s, path):
    py = s["per_year"]
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6), sharex=True,
                                   gridspec_kw={"height_ratios": [2, 1]})
    yrs = py.index.values
    ax1.plot(yrs, py["resist_rate"] * 100, "-o", color="#C4453B", lw=2)
    ax1.fill_between(yrs, py["ci_low"] * 100, py["ci_high"] * 100,
                     color="#C4453B", alpha=0.15, label="95% CI")
    ax1.set_ylabel("Resistance %")
    ax1.set_title("Levofloxacin resistance in E. coli over time (2004-2017)")
    ax1.legend(loc="lower right")
    ax2.bar(yrs, py["n"], color="#4C6F9F")
    ax2.set_ylabel("Isolates"); ax2.set_xlabel("Year")
    ax2.set_xticks(yrs); ax2.tick_params(axis="x", rotation=45)
    fig.tight_layout(); fig.savefig(path); plt.close(fig)


def _barh(ax, g, title):
    ax.barh(g.index.astype(str), g["resist_rate"] * 100, color="#7A5FA6")
    for i, (rate, n) in enumerate(zip(g["resist_rate"], g["n"])):
        ax.text(rate * 100 + 0.5, i, f"{rate:.0%} (n={int(n):,})", va="center", fontsize=8)
    ax.set_xlabel("Resistance %"); ax.set_title(title); ax.invert_yaxis()
    ax.set_xlim(0, min(100, g["resist_rate"].max() * 100 + 18))


def fig_subgroup(s, path):
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    _barh(axes[0, 0], s["by_gender"], "By gender")
    _barh(axes[0, 1], s["by_age"].dropna(), "By age group")
    _barh(axes[1, 0], s["by_country"], "By country (top 10 by n, n>=500)")
    _barh(axes[1, 1], s["by_source"], "By specimen source (top 10 by n, n>=500)")
    fig.suptitle("Levofloxacin resistance by context variable (E. coli)", y=1.01)
    fig.tight_layout(); fig.savefig(path, bbox_inches="tight"); plt.close(fig)


def fig_input_shift(d, path):
    """Do the INPUT distributions shift across years? (data-drift preview)"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.6))
    year_tot = d.groupby("Year").size()
    # (a) top-5 country share by year (denominator = all isolates that year)
    top5 = d["Country"].value_counts().head(5).index
    cnt = d.pivot_table(index="Year", columns="Country", aggfunc="size", fill_value=0)
    share = cnt[list(top5)].div(year_tot, axis=0)
    for c in top5:
        ax1.plot(share.index, share[c] * 100, "-o", ms=3, label=c)
    ax1.set_ylabel("% of that year's isolates"); ax1.set_xlabel("Year")
    ax1.set_title("Country mix over time (top 5)")
    ax1.legend(fontsize=8)
    # (b) age-group share by year (stacked)
    agecnt = (d.pivot_table(index="Year", columns="Age Group", aggfunc="size", fill_value=0)
              .reindex(columns=AGE_ORDER, fill_value=0))
    ages = agecnt.div(year_tot, axis=0)
    ax2.stackplot(ages.index, [ages[a].values * 100 for a in ages.columns],
                  labels=ages.columns, alpha=0.85)
    ax2.set_ylabel("% of that year's isolates"); ax2.set_xlabel("Year")
    ax2.set_title("Age-group mix over time"); ax2.set_ylim(0, 100)
    ax2.legend(fontsize=7, loc="upper left", ncol=2)
    fig.suptitle("Input-distribution shift across years (bridges to Phase 5 drift)", y=1.02)
    fig.tight_layout(); fig.savefig(path, bbox_inches="tight"); plt.close(fig)


def run():
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    d = load_processed()
    s = summaries(d)
    fig_class_balance(s, FIG_DIR / "class_balance.png")
    fig_temporal(s, FIG_DIR / "temporal_resistance.png")
    fig_subgroup(s, FIG_DIR / "resistance_by_subgroup.png")
    fig_input_shift(d, FIG_DIR / "input_shift_over_time.png")
    return d, s


if __name__ == "__main__":
    d, s = run()
    pd.set_option("display.width", 120)
    print("=== Overall ===")
    print(s["overall"], f"\nResistant rate: {d['target'].mean():.1%}")
    print("\n=== Per year ===")
    print((s["per_year"][["n", "resist_rate"]]
           .assign(resist_pct=lambda x: (x["resist_rate"] * 100).round(1))
           [["n", "resist_pct"]]).to_string())
    print("\n=== By gender ===\n", s["by_gender"].round(3).to_string())
    print("\n=== By age group ===\n", s["by_age"].round(3).to_string())
    print("\n=== By country (top 10) ===\n", s["by_country"].round(3).to_string())
    print("\n=== By source (top 10) ===\n", s["by_source"].round(3).to_string())
    print(f"\nFigures written to {FIG_DIR}")
