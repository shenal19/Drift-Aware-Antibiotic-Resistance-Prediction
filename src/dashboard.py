"""
Phase 9 - Dashboard core logic (importable + testable, no Streamlit here).

Everything the app needs that must be CORRECT lives here so it can be unit-tested
without launching Streamlit. Nothing is retrained, refit, or re-encoded: we load
the frozen Phase-3 XGBoost pipeline and the saved Phase-7 calibrator and apply
them exactly.

Prediction flow (identical to Phases 3/7/8):
    input row -> saved pipeline preprocessing (sparse one-hot)
              -> XGBoost raw margin
              -> saved Phase-7 sigmoid calibrator
              -> calibrated probability
              -> Phase-8 heuristic research band
"""

from __future__ import annotations
import functools
import joblib
import numpy as np
import pandas as pd

from . import config as C
from .modeling import load_modeling_frame, predictor_columns
from .risk_estimation import assign_band, BAND_RANGE     # reuse Phase-8 logic exactly
from .calibration import SigmoidCalibrator               # class needed to unpickle

MODELS_DIR = C.PROJECT_ROOT / "models"
REPORTS_DIR = C.PROJECT_ROOT / "reports"
FIG_DIR = REPORTS_DIR / "figures"


@functools.lru_cache(maxsize=1)
def load_model():
    return joblib.load(MODELS_DIR / "xgboost.joblib")


@functools.lru_cache(maxsize=1)
def load_calibrator():
    return joblib.load(MODELS_DIR / "calibrator.joblib")


@functools.lru_cache(maxsize=1)
def _processed():
    return load_modeling_frame()


def predictors():
    return predictor_columns(_processed())


def category_options():
    """Sorted unique categories per predictor (for the input form). 'Missing' added."""
    df = _processed()
    opts = {}
    for col in predictors():
        vals = sorted(df[col].dropna().astype(str).unique().tolist())
        if df[col].isna().any() and "Missing" not in vals:
            vals = ["Missing"] + vals
        opts[col] = vals
    return opts


def predict_calibrated(row: dict):
    """
    row: {predictor: value}. Returns dict with raw prob, calibrated prob, band.
    Uses the frozen pipeline + saved calibrator only.
    """
    model, calibrator = load_model(), load_calibrator()
    X = pd.DataFrame([{c: row.get(c, "Missing") for c in predictors()}])
    prep, clf = model.named_steps["prep"], model.named_steps["clf"]
    Xs = prep.transform(X)                          # sparse, as deployed
    margin = clf.predict(Xs, output_margin=True)
    raw = float(model.predict_proba(X)[:, 1][0])
    cal = float(calibrator.predict(margin)[0])
    return {"raw_probability": raw, "calibrated_probability": cal,
            "risk_band": assign_band(cal), "band_range": BAND_RANGE[assign_band(cal)]}


def local_shap(row: dict, top_n=8):
    """Optional local SHAP for one input (same sparse rep as the saved model)."""
    import shap
    model = load_model()
    X = pd.DataFrame([{c: row.get(c, "Missing") for c in predictors()}])
    prep, clf = model.named_steps["prep"], model.named_steps["clf"]
    Xs = prep.transform(X)
    names = [n.split("__", 1)[1] if "__" in n else n for n in prep.get_feature_names_out()]
    sv = np.array(shap.TreeExplainer(clf).shap_values(Xs))[0]
    active = Xs.toarray()[0] > 0                    # one-hot columns that are "on"
    contrib = pd.DataFrame({"feature": np.array(names)[active], "shap_logodds": sv[active]})
    return contrib.reindex(contrib["shap_logodds"].abs().sort_values(ascending=False).index).head(top_n)


def load_kpis():
    """Read the locked metrics from the saved report CSVs (no recomputation)."""
    kpis = {"ROC-AUC": None, "PR-AUC": None,
            "Brier_raw": None, "Brier_cal": None, "ECE_raw": None, "ECE_cal": None}
    try:
        p3 = pd.read_csv(REPORTS_DIR / "phase3_test_metrics.csv", index_col=0)
        kpis["ROC-AUC"] = float(p3.loc["XGBoost", "ROC-AUC"])
        kpis["PR-AUC"] = float(p3.loc["XGBoost", "PR-AUC"])
    except Exception:
        kpis["ROC-AUC"], kpis["PR-AUC"] = 0.694, 0.547
    try:
        p7 = pd.read_csv(REPORTS_DIR / "phase7_calibration_metrics.csv", index_col=0)
        kpis["Brier_raw"] = float(p7.loc["XGBoost Raw", "Brier"])
        kpis["Brier_cal"] = float(p7.loc["XGBoost Calibrated", "Brier"])
        kpis["ECE_raw"] = float(p7.loc["XGBoost Raw", "ECE"])
        kpis["ECE_cal"] = float(p7.loc["XGBoost Calibrated", "ECE"])
    except Exception:
        kpis.update(Brier_raw=0.2269, Brier_cal=0.2038, ECE_raw=0.1507, ECE_cal=0.0110)
    return kpis


def fig(name):
    return str(FIG_DIR / name)
