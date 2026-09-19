"""
Central configuration for the Drift-Aware Antibiotic Resistance Prediction project.

Everything that a later phase might need to know about *what* we are modelling
and *which columns are forbidden* lives here, so no magic strings are scattered
across the pipeline. Change the combination in one place to re-run for a
secondary experiment (e.g. S. aureus -> Levofloxacin).
"""

from pathlib import Path

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
# Resolve relative to this file so the project runs from any working directory.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_XLSX = PROJECT_ROOT / "data" / "raw" / "Open_Atlas_Reuse_Data.xlsx"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
SHEET_NAME = "TEST and Inform Data"

# --------------------------------------------------------------------------- #
# The modelling combination (primary experiment)
# --------------------------------------------------------------------------- #
SPECIES = "Escherichia coli"
ANTIBIOTIC = "Levofloxacin"          # target column is f"{ANTIBIOTIC}_I"

# Target encoding. Intermediate is dropped for the primary experiment (your
# spec, Module 5). Flip DROP_INTERMEDIATE to False + set INTERMEDIATE_AS to
# 0 or 1 if you ever want the "non-susceptible" convention instead.
DROP_INTERMEDIATE = True
INTERMEDIATE_AS = 1                  # only used if DROP_INTERMEDIATE is False

TIME_COL = "Year"                    # split key ONLY, never a feature
ID_COL = "Isolate Id"

# --------------------------------------------------------------------------- #
# Temporal split + reproducibility
# --------------------------------------------------------------------------- #
# NOTE: these four constants were absent from the uploaded config.py but are
# required by src/modeling.py (temporal_split, build_models) and asserted by
# phase10_audit.py. They are restored here with the project's locked values so
# the pipeline imports and runs. Data is ordered by Year and never shuffled.
TRAIN_YEARS = list(range(2004, 2015))   # 2004-2014 (train)
VAL_YEARS = [2015]                       # 2015 (validate + calibrator fit)
TEST_YEARS = [2016, 2017]                # 2016-2017 (held-out test)
RANDOM_STATE = 42                        # seed for model init / sampling only

# --------------------------------------------------------------------------- #
# Feature policy  (leakage rules — NON-NEGOTIABLE)
# --------------------------------------------------------------------------- #
# Legitimate context predictors we are allowed to use.
CANDIDATE_PREDICTORS = [
    "Organism Group",   # dropped automatically if constant (it is, for E. coli)
    "Country",
    "State",
    "Gender",
    "Age Group",
    "Speciality",
    "Source",
    "In / Out Patient",
    "Study",            # dropped by default: near-perfect proxy for year>=2012
]

# Study is time-confounded (INFORM only exists 2012+). Excluded by default so it
# cannot act as an "era" shortcut in the temporal experiment.
DROP_STUDY = True

# Columns that must NEVER be predictors (measured-outcome / label leakage).
# Every antibiotic base column (MIC) and every *_I column is outcome info.
# Phenotype (ESBL/MRSA/...) is a resistance-mechanism label -> also leakage.
LEAKAGE_METADATA = ["Phenotype"]

# High-missingness threshold above which we warn (State is ~79% missing).
HIGH_MISSING_WARN = 0.5

# Rare one-hot categories with fewer than this many rows are grouped as
# "infrequent" by the encoder. Referenced by src/psi_drift.py so drift uses the
# same rare-category cutoff as modelling. Matches OneHotEncoder(min_frequency=20)
# in src/preprocessing.py. (Absent from the uploaded config.py; restored here.)
OHE_MIN_FREQUENCY = 20

# --------------------------------------------------------------------------- #
# Drift / risk thresholds (documented, configurable — used in later phases)
# --------------------------------------------------------------------------- #
PSI_THRESHOLDS = {"low": 0.10, "moderate": 0.25}      # <0.10 low, <0.25 moderate, else high
RISK_BANDS = {"low": 0.30, "moderate": 0.70}          # visualization only, NOT clinical
