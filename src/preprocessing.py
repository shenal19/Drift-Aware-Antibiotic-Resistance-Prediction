"""
Phase 1 - Data preprocessing for the Drift-Aware Antibiotic Resistance project.

Design principles
-----------------
1. Parameterized by (species, antibiotic) via config.py so the SAME code runs
   the primary and any secondary experiment.
2. Leakage-safe: all MIC columns, all S/I/R (*_I) columns, Phenotype, and Year
   are removed from the predictor set. Year is retained ONLY as a split key.
3. No encoding is frozen here. Preprocessing outputs a *tidy* frame of raw
   categorical predictors + Year + target. The categorical encoder is BUILT
   here (build_feature_encoder) but returned UNFITTED, to be fit on the
   training fold only during the temporal experiment. Fitting an encoder on
   all years before the split would leak future information.

Run directly:  python -m src.preprocessing
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder

from . import config as C


# --------------------------------------------------------------------------- #
# Small report container so run() can hand a structured summary to callers
# (a notebook, a test, or the next phase) instead of only printing.
# --------------------------------------------------------------------------- #
@dataclass
class PreprocessReport:
    species: str
    antibiotic: str
    n_total_rows: int = 0
    n_species_rows: int = 0
    target_counts_raw: dict = field(default_factory=dict)
    n_dropped_intermediate: int = 0
    n_dropped_missing_target: int = 0
    n_dropped_duplicate_isolates: int = 0
    n_model_rows: int = 0
    target_distribution: dict = field(default_factory=dict)
    resistant_rate: float = 0.0
    per_year: pd.DataFrame = None
    predictors: list = field(default_factory=list)
    dropped_predictors: dict = field(default_factory=dict)
    leakage_removed: list = field(default_factory=list)
    missing_summary: pd.Series = None


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #
def _find_header_row(path, sheet, scan_rows: int = 15) -> int:
    """
    Locate the real header row programmatically instead of hard-coding it.

    The workbook opens with a 'Micron Research' banner (rows 1-3) and blank
    rows, so the true header ('Isolate Id', 'Study', ...) is not row 1. We scan
    the first few rows and return the 0-based index of the row whose first cell
    is the ID column.
    """
    probe = pd.read_excel(path, sheet_name=sheet, header=None, nrows=scan_rows,
                          engine="openpyxl")
    for idx in range(len(probe)):
        if str(probe.iloc[idx, 0]).strip() == C.ID_COL:
            return idx
    raise ValueError(f"Could not locate header row containing '{C.ID_COL}'.")


def load_atlas(path=None, sheet=None) -> pd.DataFrame:
    """Load the workbook with the correct header row and trim string whitespace."""
    path = path or C.RAW_XLSX
    sheet = sheet or C.SHEET_NAME
    header_row = _find_header_row(path, sheet)
    df = pd.read_excel(path, sheet_name=sheet, header=header_row, engine="openpyxl")
    # Standardize categoricals: strip surrounding whitespace on object columns.
    obj_cols = df.select_dtypes(include="object").columns
    for c in obj_cols:
        df[c] = df[c].astype("string").str.strip()
    return df


# --------------------------------------------------------------------------- #
# Column auditing (which columns are metadata vs MIC vs interpretation)
# --------------------------------------------------------------------------- #
def audit_columns(df: pd.DataFrame) -> dict:
    """Classify columns into metadata / MIC / interpretation groups."""
    interp = [c for c in df.columns if c.endswith("_I")]
    mic = [c[:-2] for c in interp if c[:-2] in df.columns]
    metadata = [c for c in df.columns if c not in interp and c not in mic]
    return {"metadata": metadata, "mic": mic, "interpretation": interp}


# --------------------------------------------------------------------------- #
# Feature selection with leakage rules applied
# --------------------------------------------------------------------------- #
def select_predictors(df: pd.DataFrame, cols: dict) -> tuple[list, dict, list]:
    """
    Return (predictors, dropped_predictors, leakage_removed).

    Leakage removed  : every MIC + every *_I column + configured metadata
                       (Phenotype). These are measured-outcome / label info.
    dropped_predictors: candidate predictors we drop with a reason
                        (constant columns, Study time-confound, Year split key).
    """
    leakage_removed = cols["mic"] + cols["interpretation"] + list(C.LEAKAGE_METADATA)

    dropped = {}
    predictors = []
    for col in C.CANDIDATE_PREDICTORS:
        if col not in df.columns:
            dropped[col] = "absent from file"
            continue
        if col == "Study" and C.DROP_STUDY:
            dropped[col] = "time-confounded (INFORM only exists 2012+); excluded from features"
            continue
        if df[col].nunique(dropna=True) <= 1:
            dropped[col] = "constant in this single-species subset; no signal"
            continue
        predictors.append(col)

    # Year is never a feature.
    dropped[C.TIME_COL] = "temporal split key, not a predictive feature"
    return predictors, dropped, leakage_removed


# --------------------------------------------------------------------------- #
# Build the modelling subset for one (species, antibiotic)
# --------------------------------------------------------------------------- #
def build_subset(df: pd.DataFrame, species=None, antibiotic=None):
    """Filter to the species, build the binary target, apply leakage rules."""
    species = species or C.SPECIES
    antibiotic = antibiotic or C.ANTIBIOTIC
    target_col = f"{antibiotic}_I"

    rep = PreprocessReport(species=species, antibiotic=antibiotic,
                           n_total_rows=len(df))

    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found.")

    ec = df[df["Species"] == species].copy()
    rep.n_species_rows = len(ec)
    rep.target_counts_raw = ec[target_col].value_counts(dropna=True).to_dict()

    # ---- Build binary target -------------------------------------------- #
    rep.n_dropped_missing_target = int(ec[target_col].isna().sum())
    if C.DROP_INTERMEDIATE:
        keep = ec[target_col].isin(["Resistant", "Susceptible"])
        rep.n_dropped_intermediate = int((ec[target_col] == "Intermediate").sum())
        model = ec[keep].copy()
        model["target"] = (model[target_col] == "Resistant").astype(int)
    else:
        keep = ec[target_col].isin(["Resistant", "Susceptible", "Intermediate"])
        model = ec[keep].copy()
        model["target"] = model[target_col].map(
            {"Susceptible": 0, "Resistant": 1, "Intermediate": C.INTERMEDIATE_AS}
        ).astype(int)

    # ---- One row per isolate -------------------------------------------- #
    before = len(model)
    model = model.drop_duplicates(subset=C.ID_COL, keep="first")
    rep.n_dropped_duplicate_isolates = before - len(model)

    # ---- Feature selection (leakage rules) ------------------------------ #
    cols = audit_columns(df)
    predictors, dropped, leakage_removed = select_predictors(model, cols)
    rep.predictors = predictors
    rep.dropped_predictors = dropped
    rep.leakage_removed = leakage_removed

    # ---- Assemble tidy modelling frame ---------------------------------- #
    tidy = model[[C.ID_COL, C.TIME_COL] + predictors + ["target"]].reset_index(drop=True)

    # ---- Reports -------------------------------------------------------- #
    rep.n_model_rows = len(tidy)
    rep.target_distribution = tidy["target"].value_counts().to_dict()
    rep.resistant_rate = float(tidy["target"].mean())
    per_year = tidy.groupby(C.TIME_COL)["target"].agg(["count", "mean"])
    per_year.columns = ["isolates", "resistant_rate"]
    per_year["resistant_rate"] = (per_year["resistant_rate"] * 100).round(1)
    rep.per_year = per_year
    rep.missing_summary = (tidy[predictors].isna().mean() * 100).round(2)

    return tidy, rep


# --------------------------------------------------------------------------- #
# Encoder builder (UNFITTED — fit on train fold only, later phases)
# --------------------------------------------------------------------------- #
def build_feature_encoder(predictors: list) -> ColumnTransformer:
    """
    Return an UNFITTED ColumnTransformer that one-hot-encodes the categorical
    predictors (all our predictors are categorical). Missing values become an
    explicit 'Missing' category (important for State, ~79% missing).

    IMPORTANT: do not call .fit() here. The temporal experiment fits this on the
    training years only, so no future-year categories leak into training.
    """
    cat_pipe = Pipeline(steps=[
        ("impute", SimpleImputer(strategy="constant", fill_value="Missing")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", min_frequency=20,
                                 sparse_output=True)),
    ])
    return ColumnTransformer(
        transformers=[("cat", cat_pipe, predictors)],
        remainder="drop",
    )


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #
def run(save: bool = True, path=None):
    """Full Phase 1: load -> subset -> report -> save tidy frame."""
    df = load_atlas(path)
    tidy, rep = build_subset(df)

    if save:
        C.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
        out = C.PROCESSED_DIR / f"{rep.species.split()[0].lower()}_{rep.antibiotic.lower()}.parquet"
        tidy.to_parquet(out, index=False)
        rep._saved_to = str(out)
    return tidy, rep


def print_report(rep: PreprocessReport):
    """Human-readable Phase 1 summary (the 11 requested items)."""
    print("=" * 70)
    print(f"PHASE 1 REPORT  |  {rep.species}  ->  {rep.antibiotic}")
    print("=" * 70)
    print(f"1. Rows in file                 : {rep.n_total_rows:,}")
    print(f"2. {rep.species} rows           : {rep.n_species_rows:,}")
    print(f"3. Raw target (S/I/R) counts    : {rep.target_counts_raw}")
    print(f"   - dropped Intermediate       : {rep.n_dropped_intermediate:,}")
    print(f"   - dropped missing target     : {rep.n_dropped_missing_target:,}")
    print(f"   - dropped duplicate isolates : {rep.n_dropped_duplicate_isolates:,}")
    print(f"4. Target distribution          : "
          f"S(0)={rep.target_distribution.get(0,0):,}  R(1)={rep.target_distribution.get(1,0):,}")
    print(f"   Resistant rate               : {rep.resistant_rate*100:.1f}%")
    print(f"5. Predictors ({len(rep.predictors)})            : {rep.predictors}")
    print(f"6. Dropped predictors           :")
    for k, v in rep.dropped_predictors.items():
        print(f"      - {k}: {v}")
    print(f"7. Leakage columns removed      : {len(rep.leakage_removed)} "
          f"(all MIC + all *_I + {C.LEAKAGE_METADATA})")
    print(f"8. Final modelling dimensions   : {rep.n_model_rows:,} rows x "
          f"{len(rep.predictors)} predictors (+ Year split key + target)")
    print("\n9. Missing % per predictor:")
    print(rep.missing_summary.to_string())
    print("\n10. Per-year (split feasibility):")
    print(rep.per_year.to_string())
    if hasattr(rep, "_saved_to"):
        print(f"\n11. Saved tidy dataset          : {rep._saved_to}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--xlsx", default=None, help="Path to the ATLAS xlsx")
    ap.add_argument("--no-save", action="store_true")
    args = ap.parse_args()
    _, report = run(save=not args.no_save, path=args.xlsx)
    print_report(report)
