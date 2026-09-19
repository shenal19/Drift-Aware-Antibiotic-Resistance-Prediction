# models/

| File | In this bundle? | Purpose |
|---|---|---|
| `calibrator.joblib` | ✅ yes | Frozen Phase-7 sigmoid/Platt calibrator (fit on 2015). ~1 KB. |
| `xgboost.joblib` | ❌ **not included** | Frozen Phase-3 primary model (sklearn Pipeline: sparse one-hot → XGBoost). ~628 KB. |
| `random_forest.joblib` | ❌ not included | RF baseline. ~638 MB — normally not needed. |
| `logistic.joblib` | ❌ not included | LR baseline. |

## About the missing `xgboost.joblib`

The primary model file was **not part of the uploaded zip** (only `calibrator.joblib` was).
It is required to run:

- the Streamlit **dashboard** (`app.py`) — prediction + local SHAP,
- **`src/shap_explainability.py`** (Phase 6),
- **`phase10_audit.py`** — its model-parity, SHAP and dashboard-parity checks.

Everything else — data, config, preprocessing, the saved report CSVs and all
figures — is verified and runnable without it (`python verify_setup.py` passes
8/8 and cleanly SKIPs the model-dependent check).

### Preferred fix
Drop the **original** frozen `xgboost.joblib` (the ~628 KB file from your build
session) into this folder. `verify_setup.py` will then also pass the parity
check, and the dashboard / audit will run exactly against the saved numbers.

### If you no longer have it — regenerate (⚠ retrains)
From the project root:

```bash
python -m src.modeling
```

This re-fits LR / RF / XGBoost on the temporal split and writes
`models/xgboost.joblib` (+ `logistic.joblib`, `random_forest.joblib`) and
`reports/phase3_test_predictions.parquet`.

**Caveat (honest):** a regenerated model may not be byte-identical to the
original. The saved `calibrator.joblib` and `reports/phase7_test_predictions.csv`
were produced against the *original* model's margins, so after regenerating you
should also re-run Phase 7 (`python -m src.calibration`) if you want the
calibrated numbers to line up again. Prefer the original artifact when possible.
