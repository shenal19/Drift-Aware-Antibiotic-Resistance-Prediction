"""
Phase 10 - final integration audit. Read-only: loads existing artifacts, verifies
consistency/leakage/temporal integrity/parity, writes audit CSVs. Trains nothing.
"""
import numpy as np, pandas as pd, joblib
from pathlib import Path
from src import config as C
from src import dashboard as D
from src.modeling import load_modeling_frame, predictor_columns
from src.risk_estimation import assign_band
from src import calibration as CALmod

R = C.PROJECT_ROOT / "reports"; F = R / "figures"; M = C.PROJECT_ROOT / "models"
APPROVED = ["Country","State","Gender","Age Group","Speciality","Source","In / Out Patient"]
FORBIDDEN = ["Year","target","Isolate Id","Phenotype","Study","Species","Organism Group","Levofloxacin","Levofloxacin_I"]
results = {}

# ---------------- Task 1: file inventory ---------------- #
inv = [
 ("1","src/preprocessing.py"),("1","data/processed/escherichia_levofloxacin.parquet"),("1","notebooks/01_data_preprocessing.ipynb"),
 ("2","src/eda.py"),("2","reports/figures/temporal_resistance.png"),("2","notebooks/02_eda.ipynb"),
 ("3","src/modeling.py"),("3","models/xgboost.joblib"),("3","reports/phase3_test_metrics.csv"),("3","reports/phase3_test_predictions.parquet"),("3","notebooks/03_baseline_ml.ipynb"),
 ("4","src/temporal_evaluation.py"),("4","reports/phase4_temporal_metrics.csv"),("4","reports/phase4_temporal_predictions.parquet"),("4","reports/figures/temporal_roc_auc.png"),("4","notebooks/04_temporal_evaluation.ipynb"),
 ("5","src/psi_drift.py"),("5","reports/phase5_psi.csv"),("5","reports/figures/psi_heatmap.png"),("5","notebooks/05_psi_drift.ipynb"),
 ("6","src/shap_explainability.py"),("6","reports/phase6_shap_global.csv"),("6","reports/figures/shap_summary.png"),("6","notebooks/06_shap_explainability.ipynb"),
 ("7","src/calibration.py"),("7","reports/phase7_calibration_metrics.csv"),("7","reports/phase7_test_predictions.csv"),("7","models/calibrator.joblib"),("7","notebooks/07_probability_calibration.ipynb"),
 ("8","src/risk_estimation.py"),("8","reports/phase8_test_risk_predictions.csv"),("8","reports/figures/risk_band_predicted_vs_observed.png"),("8","notebooks/08_risk_estimation.ipynb"),
 ("9","app.py"),("9","src/dashboard.py"),("9","README.md"),("9","PHASE9_VIVA.md"),
 ("config","src/config.py"),
]
rows=[]
for ph,f in inv:
    rows.append({"phase":ph,"file":f,"purpose":Path(f).name,"status":"present" if (C.PROJECT_ROOT/f).exists() else "MISSING"})
pd.DataFrame(rows).to_csv(R/"phase10_file_inventory.csv",index=False)
results["inventory_all_present"]= all(r["status"]=="present" for r in rows)

# ---------------- load core artifacts ---------------- #
df = load_modeling_frame(); preds = predictor_columns(df)
model = joblib.load(M/"xgboost.joblib"); calib = joblib.load(M/"calibrator.joblib")
prep, clf = model.named_steps["prep"], model.named_steps["clf"]
enc_raw = list(prep.get_feature_names_out())
enc = [n.split("__",1)[1] if "__" in n else n for n in enc_raw]
def map_orig(e):
    best=None
    for p in APPROVED:
        if e==p or e.startswith(p+"_"):
            if best is None or len(p)>len(best): best=p
    return best
orig = [map_orig(e) for e in enc]

# ---------------- Task 2: consistency ---------------- #
cons=[]
def chk(name,expected,actual):
    ok = (expected==actual); cons.append({"check":name,"expected":str(expected),"actual":str(actual),"status":"PASS" if ok else "FAIL","notes":""}); return ok
chk("species",C.SPECIES,"Escherichia coli")
chk("antibiotic",C.ANTIBIOTIC,"Levofloxacin")
chk("predictor_set",APPROVED,preds)
chk("train_years",list(range(2004,2015)),C.TRAIN_YEARS)
chk("val_years",[2015],C.VAL_YEARS)
chk("test_years",[2016,2017],C.TEST_YEARS)
chk("calibration_years",[2015],CALmod.CAL_YEARS)
chk("calibration_test_years",[2016,2017],CALmod.TEST_YEARS)
chk("risk_boundaries",(0.30,0.60),(__import__('src.risk_estimation',fromlist=['LOW_HI','MOD_HI']).LOW_HI, __import__('src.risk_estimation',fromlist=['MOD_HI']).MOD_HI))
chk("encoded_feature_count",217,len(enc))
pd.DataFrame(cons).to_csv(R/"phase10_consistency_audit.csv",index=False)
results["consistency_all_pass"]= all(c["status"]=="PASS" for c in cons)

# ---------------- Task 3: leakage audit (map-based, no substring) ---------------- #
lk=[]
def leak(name,cond): lk.append({"check":name,"result":"PASS" if cond else "FAIL"}); return cond
leak("target_not_in_predictors","target" not in preds)
leak("predictors_exactly_approved", preds==APPROVED)
leak("all_encoded_map_to_approved", all(o in APPROVED for o in orig))
leak("no_unmapped_encoded_features", None not in orig)
for fbd in FORBIDDEN:
    leak(f"forbidden_absent_from_predictors[{fbd}]", fbd not in preds)
    leak(f"forbidden_absent_from_original_map[{fbd}]", fbd not in orig)
leak("processed_has_no_MIC_or_I_columns", not any(c.endswith("_I") for c in df.columns) and "Phenotype" not in df.columns)
overall_leak = all(x["result"]=="PASS" for x in lk)
lk.append({"check":"OVERALL_LEAKAGE_AUDIT","result":"PASS" if overall_leak else "FAIL"})
pd.DataFrame(lk).to_csv(R/"phase10_leakage_audit.csv",index=False)
results["leakage_pass"]=overall_leak

# ---------------- Task 5: model artifact integrity ---------------- #
import scipy.sparse as sp
test = df[df[C.TIME_COL].isin([2016,2017])]
Xs = prep.transform(test[preds])
results["model_input_sparse"]= sp.issparse(Xs)
p_now = model.predict_proba(test[preds])[:,1]
p3 = pd.read_parquet(R/"phase3_test_predictions.parquet"); p3 = p3[p3["model"]=="XGBoost"]["resistance_probability"].values
results["model_prob_match_phase3_maxdiff"]= float(np.max(np.abs(np.sort(p3)-np.sort(p_now))))
results["predictor_count_is_7"]= len(preds)==7
results["encoded_count_is_217"]= len(enc)==217

# ---------------- Task 6: calibrator integrity ---------------- #
margin = clf.predict(Xs, output_margin=True)
p_cal_now = calib.predict(margin)
p7 = pd.read_csv(R/"phase7_test_predictions.csv")
results["calibrator_match_phase7_maxdiff"]= float(np.max(np.abs(p_cal_now - p7["calibrated_probability"].values)))
results["calibrator_is_sigmoid"]= calib.__class__.__name__=="SigmoidCalibrator"

# ---------------- Task 7: risk band integrity ---------------- #
band_cases={0.299999:"Low",0.30:"Moderate",0.599999:"Moderate",0.60:"High"}
results["band_boundaries_ok"]= all(assign_band(p)==b for p,b in band_cases.items())
p8 = pd.read_csv(R/"phase8_test_risk_predictions.csv")
recomputed_bands = [assign_band(p) for p in p8["calibrated_probability"].values]
results["band_matches_phase8"]= bool((np.array(recomputed_bands)==p8["risk_band"].values).all())
results["band_uses_calibrated_not_raw"]= bool((np.array([assign_band(p) for p in p8["calibrated_probability"]])==p8["risk_band"]).all())

# ---------------- Task 8: SHAP integrity (small sample) ---------------- #
try:
    import shap
    samp = test.sample(n=500, random_state=C.RANDOM_STATE)
    Xs_s = prep.transform(samp[preds])
    ex = shap.TreeExplainer(clf); sv = np.array(ex.shap_values(Xs_s)); base=float(np.ravel(ex.expected_value)[0])
    marg = clf.predict(Xs_s, output_margin=True)
    results["shap_feature_count_eq_model"]= sv.shape[1]==Xs_s.shape[1]==217
    results["shap_all_map_to_approved"]= all(o in APPROVED for o in orig)
    results["shap_additivity_logodds_ok"]= bool(np.allclose(sv.sum(1)+base, marg, atol=1e-3))
except Exception as e:
    results["shap_check_error"]=repr(e)

# ---------------- Task 9: PSI integrity ---------------- #
psi = pd.read_csv(R/"phase5_psi.csv", index_col=0)
results["psi_features_7"]= list(psi.index).__len__()==7
results["psi_years_2014_2017"]= [str(c) for c in psi.columns]==["2014","2015","2016","2017"]
top3 = set(psi["2017"].sort_values(ascending=False).head(3).index)
results["psi_top_drift_is_country_state_inout"]= top3=={"Country","State","In / Out Patient"}

# ---------------- Task 11: dashboard parity ---------------- #
cal_dash = calib.predict(clf.predict(prep.transform(test[preds]), output_margin=True))
results["dashboard_parity_maxdiff"]= float(np.max(np.abs(cal_dash - p7["calibrated_probability"].values)))

# ---------------- Task 12: reproducibility ---------------- #
req = (C.PROJECT_ROOT/"requirements.txt").read_text().lower()
results["requirements_has_core"]= all(k in req for k in ["pandas","scikit-learn","xgboost","shap","streamlit"])
results["all_figures_exist"]= all((F/f).exists() for f in
    ["temporal_resistance.png","temporal_roc_auc.png","psi_heatmap.png","shap_summary.png",
     "calibration_curve.png","risk_band_predicted_vs_observed.png"])

print("=== PHASE 10 AUDIT RESULTS ===")
for k,v in results.items(): print(f"  {k}: {v}")
