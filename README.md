# Drift-Aware Antibiotic Resistance Prediction

Estimating the probability that an *Escherichia coli* isolate is resistant to
**Levofloxacin** from pre-test contextual metadata, and studying temporal
generalization, feature drift, explainability, calibration and risk interpretation.

> **Decision-support research prototype — not a clinical or prescribing tool. Not clinically validated.**

This bundle reorganizes the per-phase `files (1..10)` upload into one coherent,
runnable Python project and repairs two gaps that stopped it from importing (see
[What was repaired](#what-was-repaired)). Verified with `verify_setup.py`
(8/8 runnable checks pass; the one model-dependent check is skipped only because
the frozen model isn't in the bundle — see [models/README.md](models/README.md)).

---

## Directory structure

```
HEALTHCARE_PROJECT/
├── README.md                     ← you are here
├── README_original.md            ← the original Phase-9 project README (kept as-is)
├── PHASE9_VIVA.md                ← viva Q&A
├── PHASE10_FINAL_AUDIT.md        ← final audit write-up (20 sections)
├── requirements.txt
├── verify_setup.py               ← one-shot integrity check (run this first)
├── app.py                        ← Streamlit dashboard entry point (Phase 9)
├── phase10_audit.py              ← re-runnable integrity audit (Phase 10)
├── .vscode/                      ← settings + launch configs (imports resolve, one-click run)
│   ├── settings.json
│   └── launch.json
├── src/                          ← the importable package (all logic lives here)
│   ├── __init__.py               ← (added) makes `src` a package
│   ├── config.py                 ← (repaired) locked constants & leakage policy
│   ├── preprocessing.py          ← Phase 1: leakage-safe preprocessing + encoder
│   ├── eda.py                    ← Phase 2: exploratory figures
│   ├── modeling.py               ← Phase 3: LR / RF / XGBoost, temporal split
│   ├── temporal_evaluation.py    ← Phase 4: expanding-window evaluation
│   ├── psi_drift.py              ← Phase 5: PSI drift
│   ├── shap_explainability.py    ← Phase 6: SHAP (sparse, log-odds)
│   ├── calibration.py            ← Phase 7: Platt/sigmoid calibration
│   ├── risk_estimation.py        ← Phase 8: research risk bands
│   └── dashboard.py              ← Phase 9: dashboard core logic (no Streamlit)
├── data/
│   └── processed/
│       └── escherichia_levofloxacin.parquet   ← 78,297 × (7 predictors + Year + target)
├── models/
│   ├── calibrator.joblib         ← frozen Phase-7 calibrator
│   ├── xgboost.joblib            ← ⚠ NOT in bundle — add it (see models/README.md)
│   └── README.md
├── notebooks/                    ← thin drivers over src/ (each does sys.path.insert('..'))
│   ├── 01_data_preprocessing.ipynb … 08_risk_estimation.ipynb
└── reports/                      ← saved outputs (numbers + figures)
    ├── phase3_test_metrics.csv, phase4_temporal_metrics.csv, phase5_psi.csv,
    │   phase5_drift_summary.csv, phase6_shap_global.csv, phase6_shap_local.csv,
    │   phase7_calibration_bins.csv, phase7_calibration_metrics.csv,
    │   phase7_test_predictions.csv, phase8_risk_band_calibration.csv,
    │   phase8_test_risk_predictions.csv, phase10_*.csv
    └── figures/                  ← 26 PNGs (EDA, temporal, PSI, SHAP, calibration, risk)
```

**How the wiring works:** `src/config.py` sets `PROJECT_ROOT` to its own
grandparent (i.e. the folder above `src/`), so every path (`data/`, `models/`,
`reports/`) resolves no matter your working directory. Modules import each other
with `from . import ...`; the root scripts use `from src import ...`; notebooks
add the project root to `sys.path` before importing `src`.

---

## Quick start (VS Code)

1. **Open the `HEALTHCARE_PROJECT` folder** as the workspace root in VS Code.
2. Create and select a Python 3.10+ interpreter, then install deps:
   ```bash
   python -m venv .venv
   # Windows: .venv\Scripts\activate   |   macOS/Linux: source .venv/bin/activate
   pip install -r requirements.txt
   ```
3. **Verify everything is linked and consistent:**
   ```bash
   python verify_setup.py
   ```
   Expect `8 passed, 0 failed, 1 skipped` (the skip is the frozen model — add it
   to turn that into a pass).
4. Use **Run and Debug** (the `.vscode/launch.json` configs): *Verify setup*,
   *Phase 10 audit*, *Run a phase module*, or *Streamlit dashboard*.

`.vscode/settings.json` adds the project root to the Pylance analysis path, so
`from src import ...` resolves with no red squiggles, and points the Jupyter
working directory at each notebook's own folder so their `sys.path.insert('..')`
lands on the project root.

## Running the pipeline (command line)

Run modules as packages **from the project root**:

```bash
python -m src.modeling               # Phase 3 (also writes models/xgboost.joblib)
python -m src.temporal_evaluation    # Phase 4  (long; chunk if >300s)
python -m src.psi_drift              # Phase 5
python -m src.shap_explainability    # Phase 6  (needs models/xgboost.joblib)
python -m src.calibration            # Phase 7
python -m src.risk_estimation        # Phase 8
python phase10_audit.py              # Phase 10 (needs models/xgboost.joblib + phase3 parquet)
streamlit run app.py                 # Phase 9 dashboard
```

---

## What was repaired

The uploaded folders were correct but **not runnable as shipped** — the
`config.py` in them was a stale copy missing constants the rest of the code
imports. Two minimal, documented fixes (nothing else in the logic was touched):

1. **`src/__init__.py` added** — the modules use package-relative imports
   (`from . import config`), which require `src` to be a package.
2. **`src/config.py`: restored 5 constants** that `modeling.py`, `psi_drift.py`
   and `phase10_audit.py` reference but the uploaded file omitted (the code would
   otherwise raise `AttributeError` on import):
   - `TRAIN_YEARS = 2004–2014`, `VAL_YEARS = [2015]`, `TEST_YEARS = [2016, 2017]`
     (the locked temporal split, matching the audit's asserted values),
   - `RANDOM_STATE = 42` (seeding only),
   - `OHE_MIN_FREQUENCY = 20` (rare-category cutoff, matching
     `OneHotEncoder(min_frequency=20)` in `preprocessing.py`).

Each restored line is commented in `config.py` explaining why it's there.

Note: `config.RISK_BANDS` in the original reads `{"low":0.30,"moderate":0.70}`,
but the **authoritative** band boundaries are in `src/risk_estimation.py`
(`LOW_HI, MOD_HI = 0.30, 0.60`). `RISK_BANDS` is legacy/unused by the code; the
0.60 boundary is the real one. Left unchanged to avoid altering original files.

## Not in this bundle (needed for full end-to-end run)

These are **Phase-3 outputs**, not source, and weren't in the upload:

- `models/xgboost.joblib` — the frozen primary model (see `models/README.md`).
- `reports/phase3_test_predictions.parquet` — needed by `phase10_audit.py`.
- `reports/phase4_temporal_predictions.parquet` — Phase-4 output.

Add the original `xgboost.joblib`, or regenerate with `python -m src.modeling`
(⚠ retrains — see `models/README.md` for the fidelity caveat).

## Headline results (from the saved reports)

- Baseline test 2016–2017: **XGBoost ROC-AUC 0.694**, PR-AUC 0.547, Brier 0.227.
- Temporal: **stable**, no degradation across 2011–2017.
- Drift (PSI by 2017): large in Country / State / In-Out Patient; median ≈ 0.07.
- SHAP: **Country dominates** (attribution, not causation).
- Calibration: **ECE 0.151 → 0.011** (ranking unchanged).
- Risk bands: Low 39.2% / Moderate 53.0% / High 7.8%; observed resistance
  0.21 / 0.41 / 0.66.
