"""
verify_setup.py — one-shot integrity check for the wired project.

Run from the project ROOT:
    python verify_setup.py

It confirms that the folder structure, imports, data, config, saved reports and
figures all line up, WITHOUT retraining anything. Checks that need the frozen
primary model (models/xgboost.joblib) run only if that file is present; if it is
missing they are reported as SKIPPED (not failed), so you can verify everything
else and drop the model in later.

Exit code 0 = all runnable checks passed (model-dependent ones may be skipped).
Exit code 1 = at least one check failed.
"""
from __future__ import annotations
import sys
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PASS, FAIL, SKIP = "PASS", "FAIL", "SKIP"
rows = []


def check(name, fn):
    try:
        ok, detail = fn()
        rows.append((PASS if ok else FAIL, name, detail))
    except Exception as e:  # noqa: BLE001
        rows.append((FAIL, name, f"{type(e).__name__}: {e}"))


def skip(name, detail):
    rows.append((SKIP, name, detail))


# --------------------------------------------------------------------------- #
# 1. Structure
# --------------------------------------------------------------------------- #
def _structure():
    needed = [
        "src/__init__.py", "src/config.py", "src/preprocessing.py", "src/modeling.py",
        "src/temporal_evaluation.py", "src/psi_drift.py", "src/shap_explainability.py",
        "src/calibration.py", "src/risk_estimation.py", "src/dashboard.py", "src/eda.py",
        "app.py", "phase10_audit.py", "requirements.txt",
        "data/processed/escherichia_levofloxacin.parquet",
        "models/calibrator.joblib",
    ]
    missing = [p for p in needed if not (ROOT / p).exists()]
    return (not missing), ("all present" if not missing else f"missing: {missing}")


check("structure: required files present", _structure)

# --------------------------------------------------------------------------- #
# 2. Imports (this is what proves the package is 'linked' for VS Code)
# --------------------------------------------------------------------------- #
sys.path.insert(0, str(ROOT))  # so `from src import ...` resolves from anywhere


def _imports():
    import importlib
    mods = ["src.config", "src.preprocessing", "src.eda", "src.modeling",
            "src.temporal_evaluation", "src.psi_drift", "src.calibration",
            "src.risk_estimation", "src.dashboard"]
    imported = []
    for m in mods:
        importlib.import_module(m)
        imported.append(m.split(".")[-1])
    # shap_explainability imports `shap` at top level — import separately so a
    # missing optional dep is reported clearly rather than failing the batch.
    try:
        importlib.import_module("src.shap_explainability")
        imported.append("shap_explainability")
    except Exception as e:  # noqa: BLE001
        return False, f"core modules OK {imported}; shap_explainability needs: {e}"
    return True, f"imported {imported}"


check("imports: all src modules import cleanly", _imports)

# --------------------------------------------------------------------------- #
# 3. Config constants
# --------------------------------------------------------------------------- #
def _config():
    from src import config as C
    assert C.TRAIN_YEARS == list(range(2004, 2015)), "TRAIN_YEARS wrong"
    assert C.VAL_YEARS == [2015], "VAL_YEARS wrong"
    assert C.TEST_YEARS == [2016, 2017], "TEST_YEARS wrong"
    assert isinstance(C.RANDOM_STATE, int), "RANDOM_STATE missing"
    assert C.SPECIES == "Escherichia coli" and C.ANTIBIOTIC == "Levofloxacin"
    return True, f"species={C.SPECIES}, drug={C.ANTIBIOTIC}, split=2004-14/15/16-17"


check("config: locked constants present & correct", _config)

# --------------------------------------------------------------------------- #
# 4. Data + predictors
# --------------------------------------------------------------------------- #
APPROVED = ["Country", "State", "Gender", "Age Group", "Speciality", "Source", "In / Out Patient"]


def _data():
    from src.modeling import load_modeling_frame, predictor_columns
    df = load_modeling_frame()
    preds = predictor_columns(df)
    assert preds == APPROVED, f"predictors != approved 7: {preds}"
    assert "target" in df.columns, "no target column"
    yrs = sorted(df["Year"].unique().tolist())
    return True, f"{df.shape[0]:,} rows x {len(preds)} predictors; years {yrs[0]}-{yrs[-1]}"


check("data: parquet loads, 7 approved predictors, target present", _data)


def _encoder():
    from src.preprocessing import build_feature_encoder
    from src.modeling import load_modeling_frame, predictor_columns
    df = load_modeling_frame()
    enc = build_feature_encoder(predictor_columns(df))
    return (enc is not None), f"unfitted encoder built ({type(enc).__name__})"


check("preprocessing: feature encoder builds", _encoder)

# --------------------------------------------------------------------------- #
# 5. Calibrator (frozen) unpickles
# --------------------------------------------------------------------------- #
def _calibrator():
    import joblib
    from src.calibration import SigmoidCalibrator  # noqa: F401  (needed to unpickle)
    cal = joblib.load(ROOT / "models" / "calibrator.joblib")
    assert cal.__class__.__name__ == "SigmoidCalibrator", cal.__class__.__name__
    return True, "calibrator.joblib loads as SigmoidCalibrator"


check("models: calibrator.joblib unpickles", _calibrator)

# --------------------------------------------------------------------------- #
# 6. Risk-band logic
# --------------------------------------------------------------------------- #
def _bands():
    from src.risk_estimation import assign_band, LOW_HI, MOD_HI
    cases = {0.29: "Low", 0.30: "Moderate", 0.59: "Moderate", 0.60: "High"}
    bad = {p: assign_band(p) for p, b in cases.items() if assign_band(p) != b}
    assert not bad, f"band mismatch: {bad}"
    return True, f"boundaries {LOW_HI}/{MOD_HI}; Low<0.30 / Mod / High>=0.60"


check("risk_estimation: band boundaries correct", _bands)

# --------------------------------------------------------------------------- #
# 7. Saved reports + figures readable
# --------------------------------------------------------------------------- #
def _reports():
    import pandas as pd
    csvs = ["phase3_test_metrics.csv", "phase4_temporal_metrics.csv", "phase5_psi.csv",
            "phase6_shap_global.csv", "phase7_calibration_metrics.csv",
            "phase7_test_predictions.csv", "phase8_test_risk_predictions.csv"]
    for c in csvs:
        pd.read_csv(ROOT / "reports" / c)
    figs = list((ROOT / "reports" / "figures").glob("*.png"))
    assert len(figs) >= 20, f"only {len(figs)} figures"
    return True, f"{len(csvs)} report CSVs readable; {len(figs)} figures present"


check("reports: CSVs parse & figures present", _reports)

# --------------------------------------------------------------------------- #
# 8. Model-dependent parity (only if xgboost.joblib is present)
# --------------------------------------------------------------------------- #
XGB = ROOT / "models" / "xgboost.joblib"
if XGB.exists():
    def _parity():
        import numpy as np, pandas as pd, joblib
        from src import config as C
        from src.modeling import load_modeling_frame, predictor_columns
        model = joblib.load(XGB)
        df = load_modeling_frame()
        preds = predictor_columns(df)
        test = df[df[C.TIME_COL].isin(C.TEST_YEARS)]
        p_now = model.predict_proba(test[preds])[:, 1]
        p7 = pd.read_csv(ROOT / "reports" / "phase7_test_predictions.csv")
        prep, clf = model.named_steps["prep"], model.named_steps["clf"]
        cal = joblib.load(ROOT / "models" / "calibrator.joblib")
        p_cal = cal.predict(clf.predict(prep.transform(test[preds]), output_margin=True))
        d = float(np.max(np.abs(p_cal - p7["calibrated_probability"].values)))
        return (d < 1e-6), f"calibrated-prob parity vs phase7 maxdiff={d:.2e}"
    check("model: frozen XGBoost parity with saved reports", _parity)
else:
    skip("model: frozen XGBoost parity", "models/xgboost.joblib not in this bundle "
         "-> dashboard/SHAP/full audit need it (see models/README.md)")

# --------------------------------------------------------------------------- #
# Summary
# --------------------------------------------------------------------------- #
print("\n" + "=" * 78)
print("  SETUP VERIFICATION — Drift-Aware Antibiotic Resistance Prediction")
print("=" * 78)
w = max(len(n) for _, n, _ in rows)
for status, name, detail in rows:
    mark = {PASS: "OK  ", FAIL: "FAIL", SKIP: "SKIP"}[status]
    print(f"  [{mark}] {name:<{w}}  {detail}")
print("=" * 78)
n_pass = sum(s == PASS for s, _, _ in rows)
n_fail = sum(s == FAIL for s, _, _ in rows)
n_skip = sum(s == SKIP for s, _, _ in rows)
print(f"  {n_pass} passed, {n_fail} failed, {n_skip} skipped")
if n_fail == 0 and n_skip:
    print("  -> Structure & code verified. Add models/xgboost.joblib to enable the "
          "skipped model checks, the dashboard, SHAP and the full audit.")
elif n_fail == 0:
    print("  -> Fully verified end to end.")
print("=" * 78)
sys.exit(1 if n_fail else 0)
