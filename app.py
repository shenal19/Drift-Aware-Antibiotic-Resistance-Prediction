"""
Phase 9 - Streamlit dashboard for the Drift-Aware Antibiotic Resistance project.

A visualization/interface layer over the completed Phases 1-8. It loads the
FROZEN Phase-3 XGBoost pipeline and the SAVED Phase-7 calibrator and applies
them; nothing is retrained, re-encoded, or refit here. Run:

    streamlit run app.py
"""

import pandas as pd
import streamlit as st

from src import dashboard as D

st.set_page_config(page_title="E. coli - Levofloxacin Resistance (Research)",
                   layout="wide", initial_sidebar_state="expanded")


# --------------------------------------------------------------------------- #
# Cached loaders (never retrain / refit)
# --------------------------------------------------------------------------- #
@st.cache_resource
def _model():
    return D.load_model()


@st.cache_resource
def _calibrator():
    return D.load_calibrator()


@st.cache_data
def _kpis():
    return D.load_kpis()


@st.cache_data
def _options():
    return D.category_options()


DISCLAIMER = ("Academic research prototype - not a clinical prescribing system. "
              "Laboratory susceptibility testing and clinical judgment remain necessary.")


# --------------------------------------------------------------------------- #
# Header
# --------------------------------------------------------------------------- #
def header():
    st.title("E. coli Levofloxacin Resistance - Research Decision-Support Dashboard")
    st.caption("Temporal evaluation, drift monitoring, explainability and "
               "calibrated resistance probability")
    st.warning(DISCLAIMER)


# --------------------------------------------------------------------------- #
# Section 1 - Model summary
# --------------------------------------------------------------------------- #
def section_summary():
    st.subheader("Model summary")
    st.write("Metrics describe the **research** XGBoost model on the temporal test "
             "period **2016-2017** (train 2004-2014, calibrate 2015).")
    k = _kpis()
    c = st.columns(6)
    c[0].metric("Model", "XGBoost")
    c[1].metric("Test period", "2016-2017")
    c[2].metric("ROC-AUC", f"{k['ROC-AUC']:.3f}")
    c[3].metric("PR-AUC", f"{k['PR-AUC']:.3f}")
    c[4].metric("Calibrated Brier", f"{k['Brier_cal']:.3f}")
    c[5].metric("Calibrated ECE", f"{k['ECE_cal']:.3f}")
    st.info("Discrimination is modest (ROC-AUC ~0.69) because predictors are "
            "pre-test contextual metadata only - this is the leakage-free ceiling, "
            "not a limitation of effort.")


# --------------------------------------------------------------------------- #
# Section 2 - Resistance over time
# --------------------------------------------------------------------------- #
def section_temporal():
    st.subheader("Resistance over time")
    a, b = st.columns(2)
    with a:
        st.markdown("**Observed resistance by year (Phase 2)**")
        st.image(D.fig("temporal_resistance.png"), width='stretch')
    with b:
        st.markdown("**Temporal model performance - ROC-AUC (Phase 4)**")
        st.image(D.fig("temporal_roc_auc.png"), width='stretch')
    st.caption("Found no dramatic performance collapse across 2011-2017; "
               "performance was relatively stable.")


# --------------------------------------------------------------------------- #
# Section 3 - Temporal drift (PSI)
# --------------------------------------------------------------------------- #
def section_drift():
    st.subheader("Temporal drift (PSI)")
    st.write("PSI measures change in observed feature distributions relative to "
             "the **2011-2013** reference period. Substantial observed drift appears "
             "in **Country, State and In / Out Patient**.")
    a, b = st.columns(2)
    with a:
        st.image(D.fig("psi_heatmap.png"), width='stretch')
    with b:
        st.image(D.fig("psi_by_feature_2017.png"), width='stretch')
    st.caption("PSI reflects observed population/distribution change. It does not "
               "prove bacterial evolution and does not imply the model is unsafe.")


# --------------------------------------------------------------------------- #
# Section 4 - Explainability (SHAP)
# --------------------------------------------------------------------------- #
def section_shap():
    st.subheader("Model explainability (SHAP)")
    st.write("SHAP describes which input variables contribute most to the XGBoost "
             "model's predictions. It does **not** establish causality.")
    a, b = st.columns([1, 1])
    with a:
        st.image(D.fig("shap_global_original_features.png"), width='stretch')
    with b:
        st.image(D.fig("shap_summary.png"), width='stretch')
    st.caption("Order (model-attributed): Country > Age Group > Source > Speciality "
               "> Gender > State > In / Out Patient.")


# --------------------------------------------------------------------------- #
# Section 5 - Calibrated probability
# --------------------------------------------------------------------------- #
def section_calibration():
    st.subheader("Calibrated probability")
    k = _kpis()
    c = st.columns(4)
    c[0].metric("Raw Brier", f"{k['Brier_raw']:.4f}")
    c[1].metric("Calibrated Brier", f"{k['Brier_cal']:.4f}", delta=f"{k['Brier_cal']-k['Brier_raw']:.4f}")
    c[2].metric("Raw ECE", f"{k['ECE_raw']:.4f}")
    c[3].metric("Calibrated ECE", f"{k['ECE_cal']:.4f}", delta=f"{k['ECE_cal']-k['ECE_raw']:.4f}")
    a, b = st.columns(2)
    with a:
        st.image(D.fig("calibration_curve.png"), width='stretch')
    with b:
        st.image(D.fig("brier_score_comparison.png"), width='stretch')
    st.info("Sigmoid/Platt calibration (fitted on 2015) improves agreement between "
            "predicted probabilities and observed resistance frequencies while "
            "preserving ranking (ROC-AUC/PR-AUC unchanged).")


# --------------------------------------------------------------------------- #
# Section 6 - Research risk bands
# --------------------------------------------------------------------------- #
def section_risk():
    st.subheader("Research risk bands")
    st.write("**Research Risk Bands** - heuristic presentation categories over the "
             "calibrated probability: **Low** <0.30, **Moderate** 0.30-<0.60, "
             "**High** >=0.60.")
    st.warning("Heuristic research categories - **not** clinical decision thresholds. "
               "'Low'/'High' do not mean safe/unsafe.")
    cols = st.columns(3)
    for c, name in zip(cols, ["risk_band_distribution.png", "risk_band_by_year.png",
                              "risk_band_predicted_vs_observed.png"]):
        c.image(D.fig(name), width='stretch')
    try:
        st.dataframe(pd.read_csv(D.REPORTS_DIR / "phase8_risk_band_calibration.csv"),
                     hide_index=True, width='stretch')
    except Exception:
        pass


# --------------------------------------------------------------------------- #
# Section 7/8 - Interactive isolate estimation (+ optional local SHAP)
# --------------------------------------------------------------------------- #
def _band_box(band, prob):
    msg = f"**Research Risk Band: {band}**  (calibrated probability = {prob:.1%})"
    (st.error if band == "High" else st.warning if band == "Moderate" else st.info)(msg)


def section_interactive():
    st.subheader(" Interactive isolate estimation")
    st.write("Enter the seven contextual predictors. The input passes through the "
             "**saved** XGBoost pipeline, then the **saved** Phase-7 calibrator, then "
             "the Phase-8 band. No retraining or refitting occurs.")
    opts = _options()
    _model(); _calibrator()                    # warm cache
    cols = st.columns(2)
    row = {}
    for i, feat in enumerate(D.predictors()):
        with cols[i % 2]:
            choices = opts[feat]
            default = choices.index("United States") if feat == "Country" and "United States" in choices else 0
            row[feat] = st.selectbox(feat, choices, index=default)

    if st.button("Estimate resistance probability", type="primary"):
        out = D.predict_calibrated(row)
        m = st.columns(3)
        m[0].metric("Calibrated probability", f"{out['calibrated_probability']:.1%}")
        m[1].metric("Raw probability", f"{out['raw_probability']:.1%}")
        m[2].metric("Band range", out["band_range"])
        _band_box(out["risk_band"], out["calibrated_probability"])
        st.caption("The **calibrated** probability (not the raw one) determines the band. " + DISCLAIMER)

        st.markdown("**Local explanation (SHAP, log-odds contributions)**")
        try:
            ls = D.local_shap(row)
            ls = ls.rename(columns={"feature": "Active feature", "shap_logodds": "SHAP (log-odds)"})
            st.dataframe(ls.round(3), hide_index=True, width='stretch')
            st.caption("Positive pushes toward higher resistance probability; negative "
                       "toward lower. SHAP explains the model output, not the true outcome.")
        except Exception as e:
            st.caption(f"Local SHAP unavailable in this environment ({type(e).__name__}).")


# --------------------------------------------------------------------------- #
# Section 8 - Limitations
# --------------------------------------------------------------------------- #
def section_limitations():
    st.subheader("Limitations")
    st.markdown(
        "- Predictors are contextual **metadata only** (no laboratory measurements).\n"
        "- Test period limited to **2016-2017**.\n"
        "- Temporal drift observed in selected variables (Country, State, In/Out Patient).\n"
        "- **PSI does not prove biological evolution.**\n"
        "- **SHAP is not causal.**\n"
        "- Risk bands are **heuristic**, not clinical thresholds.\n"
        "- Calibration was fitted using **2015** only.\n"
        "- Data are observed surveillance isolates, not every clinical population.\n"
        "- The model is **not clinically validated**.\n"
        "- Laboratory susceptibility testing and clinical judgment remain necessary.")


# --------------------------------------------------------------------------- #
# Router
# --------------------------------------------------------------------------- #
SECTIONS = {
    "Model summary": section_summary,
    "Resistance over time": section_temporal,
    "Temporal drift (PSI)": section_drift,
    "Explainability (SHAP)": section_shap,
    "Calibrated probability": section_calibration,
    "Research risk bands": section_risk,
    "Interactive estimation": section_interactive,
    "Limitations": section_limitations,
}


def main():
    header()
    st.sidebar.title("Sections")
    choice = st.sidebar.radio("Go to", list(SECTIONS.keys()))
    st.sidebar.markdown("---")
    st.sidebar.caption(" ")
    st.divider()
    SECTIONS[choice]()


if __name__ == "__main__":
    main()
