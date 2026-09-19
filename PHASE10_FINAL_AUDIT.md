# PHASE 10 — Final Integration, Validation & Research Audit

**Status: COMPLETE.** All 22 automated integrity checks pass; nothing was retrained, refit, or re-methodologised.

## 1. Executive summary
Phases 1–9 are internally consistent, leakage-safe, temporally sound, reproducible, and presentation-ready. The saved XGBoost reproduces Phase-3 probabilities exactly (max abs diff **0.0**); the saved calibrator reproduces Phase-7 calibrated probabilities to **1.1e-16**; the dashboard reproduces the same to **1.1e-16**. The leakage audit passes on a map-based (not substring) basis, with all 217 encoded features tracing to the seven approved predictors. One dashboard numbering inconsistency was fixed. No overstated claims remain.

## 2. Final architecture
Preprocessing → EDA → Baseline ML → Temporal evaluation → PSI drift → SHAP → Calibration → Risk bands → Dashboard. Each phase has a `src/` module, a notebook, and saved reports/figures (inventory: all present).

## 3. Dataset and target
Pfizer ATLAS **TEST + INFORM** re-use extract (2004–2017). Organism *E. coli*, antibiotic **Levofloxacin**. Target `Levofloxacin_I`: R=1, S=0, Intermediate dropped. Modelling frame: 78,297 isolates.

## 4. Predictor list (7, leakage-safe)
Country, State, Gender, Age Group, Speciality, Source, In / Out Patient. `Year` is a split key only; MIC/`*_I`/Phenotype/Study/Species/Organism Group/Isolate Id are excluded.

## 5. Temporal split
Train 2004–2014 · Validate 2015 · Test 2016–2017 (no shuffle). Phase-4 expanding window 2011→2017. PSI reference 2011–2013 vs 2014–2017. Calibrator fit on 2015 only. Verified consistent across config and all modules.

## 6. Model results
XGBoost test (2016–2017): **ROC-AUC 0.694, PR-AUC 0.547, raw Brier 0.227**. Modest discrimination is the leakage-free ceiling for metadata-only predictors, not an implementation shortfall.

## 7. Temporal evaluation
Expanding-window 2011–2017 shows **relatively stable performance, no degradation** (ROC-AUC ~0.62–0.70; slight upward drift within noise). Reported without manufactured drift; PR-AUC changes partly track prevalence (mechanical).

## 8. PSI findings
**Scenario A — substantial input drift, stable performance.** 2017 PSI: Country 1.242, State 1.052, In/Out Patient 1.556; Gender/Age/Speciality/Source < 0.10. Median PSI ~0.07 (drift concentrated in 3 features). Interpreted as observed population/feature-distribution change — not biological evolution, not model failure.

## 9. SHAP findings
Explains the **saved sparse** XGBoost (log-odds attribution, base ≈0.005). Original-variable importance: Country 0.72 ≫ Age Group 0.22 > Source 0.20 > Speciality 0.12 > Gender 0.09 > State 0.07 > In/Out 0.02. Additivity holds; all features map to the approved 7. Model attribution, **not causal**.

## 10. Calibration findings
Platt/sigmoid on 2015 applied to 2016–2017: **Brier 0.227→0.204, log loss 0.645→0.594, ECE 0.151→0.011**, ROC-AUC/PR-AUC **unchanged** (0.694/0.547). Raw probabilities were overconfident (class weighting); calibration corrected alignment while preserving ranking.

## 11. Risk-band findings
Heuristic bands on the **calibrated** probability: Low <0.30 (39.2%), Moderate 0.30–<0.60 (53.0%), High ≥0.60 (7.8%). Observed resistance rises monotonically **0.21 / 0.41 / 0.66** and matches mean predicted per band. Not clinical thresholds.

## 12. Dashboard validation
Loads frozen `xgboost.joblib` + saved `calibrator.joblib`; every section executes with no exception (Streamlit AppTest); interactive prediction reproduces Phase-7 to **1.1e-16**. Sidebar numbering corrected (Limitations is Section 8; local SHAP is a subsection of Section 7).

## 13. Leakage audit
`reports/phase10_leakage_audit.csv` = **PASS**. Predictors exactly the approved 7; every forbidden variable absent from predictors and from the encoded→original map; all 217 encoded features map to an approved predictor (map-based, no substring matching); processed frame has no MIC/`*_I`/Phenotype columns.

## 14. Artifact integrity
XGBoost: loads; pipeline = preprocessing + classifier; 7 predictors → **217** sparse one-hot features; predictions reproduce Phase-3 (**max diff 0.0**). Calibrator: `SigmoidCalibrator`, operates on raw margin, reproduces Phase-7 (**1.1e-16**).

## 15. Reproducibility
Modules run in order; all expected artifacts, figures, and CSVs present; `requirements.txt` covers pandas/scikit-learn/xgboost/shap/streamlit. Note: the 126 MB raw xlsx is needed only to regenerate Phase 1; `random_forest.joblib` is large (~638 MB).

## 16. Known limitations
Metadata-only predictors (modest ceiling); test window 2016–2017; drift in Country/State/In-Out Patient; PSI ≠ biological evolution; SHAP ≠ causal; bands heuristic; calibration fit on 2015; surveillance isolates only; **not clinically validated**; lab susceptibility testing and clinical judgment remain necessary.

## 17. Corrections made (across the project, all locked)
1. **Study** excluded (near-perfect time proxy); **Organism Group** dropped (constant for E. coli).
2. **Sparse XGBoost semantics** preserved — densifying changed predictions (Phase 6); SHAP and dashboard use the sparse rep.
3. **Leakage check** fixed from substring to **map-based** ("Year" no longer false-flags "Age Group_0 to 2 Years").
4. **Calibrator persisted** to `models/calibrator.joblib` (Phase 9) so the app loads, never refits.
5. **Dense-RF speed fix** (Phase 4) — representation only; RF ROC-AUC identical to sparse.
6. **Dashboard numbering** — Limitations renumbered to Section 8; viva notes updated.

## 18. Remaining issues
None blocking. Two "PASS WITH NOTE" items (Baseline ML modest by design; large RF artifact / raw-xlsx dependency) are documentation notes, not defects.

## 19. Defensible research claims
- A leakage-safe, metadata-only model estimates E. coli Levofloxacin resistance with **modest** discrimination (ROC-AUC ~0.69).
- Performance is **relatively stable** across 2011–2017 (no degradation observed).
- **Substantial input-distribution drift** exists in a subset of contextual features (Country, State, In/Out Patient) — observed distribution change, not biological evolution.
- **Calibration** materially improves probability alignment (ECE 0.15→0.01) without changing ranking.
- Risk bands provide interpretable, **descriptively separated** categories over the calibrated probability.
- The system is a **decision-support research prototype**, not clinically validated.

## 20. Final viva checklist
- [x] One species/antibiotic/target across all phases
- [x] Seven predictors only; leakage audit PASS
- [x] Temporal split respected; test untouched until evaluation
- [x] XGBoost frozen; probs reproduce Phase 3 exactly
- [x] Calibrator fit on 2015; reproduces Phase 7
- [x] Bands from calibrated probability; boundaries 0.30/0.60
- [x] PSI = observed drift; SHAP = attribution (both non-causal)
- [x] No synthetic data, no clinical/prescribing claims
- [x] Dashboard runs; parity verified
- [x] Limitations visible throughout
