# Drift-Aware Antibiotic Resistance Prediction

Academic **decision-support research prototype** (30-mark internal project).
Predicts the probability that an *E. coli* isolate is resistant to **Levofloxacin**
from pre-test contextual metadata, and studies temporal generalization, feature
drift, explainability, probability calibration and risk interpretation.

**Not a clinical prescribing system.** Laboratory susceptibility testing and
clinical judgment remain necessary.

## Data / experiment
- Dataset: Pfizer ATLAS TEST + INFORM re-use extract (2004-2017).
- Species/antibiotic: *E. coli* -> Levofloxacin. Target: `Levofloxacin_I` (R=1, S=0, Intermediate dropped).
- Predictors (7, leakage-safe): Country, State, Gender, Age Group, Speciality, Source, In / Out Patient.
- Excluded (leakage/constant/proxy): all MIC + all `*_I` + Phenotype + Species + Organism Group + Study + Year(as feature) + Isolate Id.

## Pipeline (phases)
1. Leakage-safe preprocessing -> `data/processed/escherichia_levofloxacin.parquet`
2. EDA · 3. Baseline ML (LR/RF/XGBoost, temporal split) · 4. Temporal (expanding-window) evaluation
5. PSI drift · 6. SHAP explainability · 7. Probability calibration · 8. Research risk bands · 9. Streamlit dashboard

## Setup & run
```bash
pip install -r requirements.txt
# place the raw file at data/raw/Open_Atlas_Reuse_Data.xlsx (only needed to regenerate phase 1)
python -m src.preprocessing        # phase 1 (regenerates processed parquet)
python -m src.modeling             # phase 3 (saves models/*.joblib)
python -m src.calibration          # phase 7 (metrics + saves models/calibrator.joblib via fit_and_save_calibrator)
python -m src.risk_estimation      # phase 8
streamlit run app.py               # phase 9 dashboard
```

## Phase 9 - dashboard

### How to run
`streamlit run app.py` (from the project root).

### Architecture
`app.py` is a thin Streamlit UI. All correctness-critical logic lives in
`src/dashboard.py` (loading, prediction flow, KPIs) so it is unit-testable
without launching Streamlit. Sidebar navigation switches between: Model summary,
Resistance over time, Temporal drift (PSI), Explainability (SHAP), Calibrated
probability, Research risk bands, Interactive estimation, and Limitations.
Sections 2-6 display the figures/CSVs already produced by Phases 2-8.

### Data flow (interactive estimation)
```
user input (7 predictors)
  -> saved Phase-3 pipeline preprocessing (sparse one-hot, handle_unknown="ignore")
  -> XGBoost raw margin
  -> saved Phase-7 sigmoid calibrator (models/calibrator.joblib)
  -> calibrated resistance probability
  -> Phase-8 heuristic research band (Low <0.30 / Moderate 0.30-<0.60 / High >=0.60)
```

### Why the model is not retrained
The saved XGBoost (`models/xgboost.joblib`) was trained on the **sparse** one-hot
representation, where XGBoost treats structural zeros as missing. Densifying or
re-encoding changes its predictions materially (found in Phase 6). The dashboard
therefore loads the exact saved pipeline and calibrator and only applies them.

### Why calibrated probability is used
Phase 7 showed the raw class-weighted probabilities were overconfident
(ECE 0.151); a 2015-fitted sigmoid layer improved alignment (ECE 0.011, Brier
0.227 -> 0.204) while preserving ranking. The dashboard's user-facing probability
is the **calibrated** one.

### Why risk bands are heuristic
The 0.30 / 0.60 boundaries are presentation categories to make the probability
easier to read. They are **not** clinically validated thresholds and never imply
"safe"/"unsafe" or any prescribing action.

### Limitations
Metadata-only predictors; test period 2016-2017; drift in Country/State/In-Out
Patient; PSI is not evidence of biological evolution; SHAP is not causal; bands
are heuristic; calibration fit on 2015; surveillance isolates only; not clinically
validated; laboratory testing remains necessary.
