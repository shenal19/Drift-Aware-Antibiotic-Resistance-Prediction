# Phase 9 - Dashboard viva notes

## Dashboard architecture
`app.py` (Streamlit UI, sidebar navigation, 8 sections) is a thin layer over
`src/dashboard.py`, which holds all correctness-critical logic (loading the frozen
model + saved calibrator, the prediction flow, KPI loading, category options,
optional local SHAP). Keeping logic in `src/dashboard.py` lets it be tested
without launching Streamlit. Sections 2-6 render figures/CSVs produced by
Phases 2-8; Section 7 does live estimation; Section 9 lists limitations.

## Data flow (live estimation)
input (7 predictors) -> saved Phase-3 pipeline preprocessing (sparse one-hot)
-> XGBoost raw margin -> saved Phase-7 sigmoid calibrator -> calibrated
probability -> Phase-8 band. Verified: dashboard calibrated probabilities equal
the Phase-7 test probabilities to ~1e-16.

## Why Streamlit
Lightweight pure-Python way to build an interactive analytics UI with no
front-end code; runs locally with `streamlit run app.py`; ideal for a small
academic prototype.

## How prediction works
The saved `models/xgboost.joblib` pipeline transforms the input row with its
already-fitted encoder (no refit) and produces a raw margin/probability. We never
recreate the one-hot encoding by hand.

## How calibration is applied
The saved `models/calibrator.joblib` (a sigmoid/Platt layer fitted on 2015 in
Phase 7) maps the frozen model's raw margin to a calibrated probability. It is
loaded, never refitted, in the app.

## How risk bands are assigned
A pure function of the calibrated probability (Phase-8 `assign_band`):
Low <0.30, Moderate 0.30-<0.60, High >=0.60. No target, no features, no tuning.

## How SHAP is displayed
Global SHAP figures from Phase 6 are shown as images. Optional local SHAP for a
live input uses `shap.TreeExplainer` on the **same sparse** representation as the
saved model (matching the Phase-6 correction), shown as top log-odds contributions.

## How PSI is displayed
Phase-5 PSI heatmap and 2017 feature-level PSI figures are shown as images, with
text noting the 2011-2013 reference and that Country/State/In-Out Patient drift
most. No new PSI is computed in the app.

## Leakage safeguards
Only the 7 approved predictors are inputs; Year/target/Isolate Id/MIC/`*_I`/
Phenotype/Study/Species/Organism Group are never used; the encoder and calibrator
are loaded (not refit); `handle_unknown="ignore"` handles unseen categories.

## 10 likely viva questions
1. **Is the dashboard another model?** No - a UI layer over saved Phase 1-8 outputs.
2. **Does it retrain XGBoost?** No; it loads `models/xgboost.joblib` frozen.
3. **Why not densify the input?** XGBoost was trained on sparse input (zeros=missing); densifying changes predictions.
4. **Which probability is shown?** The Phase-7 **calibrated** probability.
5. **Where does the calibrator come from?** Saved `models/calibrator.joblib`, fit on 2015 only; loaded, not refit.
6. **How is the band decided?** `assign_band(calibrated_probability)` with fixed 0.30/0.60 cutoffs.
7. **Are the bands clinical?** No - heuristic research categories; not safe/unsafe, not prescribing.
8. **What if a user picks an unseen category?** `handle_unknown="ignore"` encodes it as all-zeros; no crash, no leakage.
9. **Is SHAP causal?** No - model attribution only.
10. **Does PSI mean the model is unsafe or bacteria evolved?** No - it only measures observed input-distribution change.
