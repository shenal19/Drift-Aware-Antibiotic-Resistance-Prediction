"""
Phase 6 - SHAP explainability for the saved XGBoost pipeline (E. coli -> Levofloxacin).

Explains WHY the XGBoost model predicts higher/lower Levofloxacin-resistance
probability, using the already-trained Phase-3 pipeline. Nothing is retrained,
tuned, or re-encoded; SHAP is an explanation layer only.

Output space: for an XGBoost binary classifier, shap.TreeExplainer explains the
model's RAW MARGIN (log-odds), not probability. Positive SHAP -> higher
resistance log-odds (-> higher resistance probability); negative -> lower. Base
value = mean model log-odds over the background. We report probabilities from
the pipeline separately and never relabel log-odds SHAP as probability changes.

Run:  python -m src.shap_explainability
"""

from __future__ import annotations
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import shap

from . import config as C
from .modeling import load_modeling_frame, predictor_columns

MODELS_DIR = C.PROJECT_ROOT / "models"
FIG_DIR = C.PROJECT_ROOT / "reports" / "figures"
REPORTS_DIR = C.PROJECT_ROOT / "reports"

SHAP_YEARS = [2016, 2017]     # Phase-3 final test period
SAMPLE_N = 5000
FORBIDDEN = {"Year", "target", "Isolate Id", "Species", "Organism Group", "Study",
             "Phenotype", "Levofloxacin", "Levofloxacin_I"}


# --------------------------------------------------------------------------- #
# Load + prepare
# --------------------------------------------------------------------------- #
def load_pipeline():
    path = MODELS_DIR / "xgboost.joblib"
    if not path.exists():
        raise FileNotFoundError(f"Saved XGBoost pipeline not found at {path}. "
                                "Run Phase 3 first; do not retrain here.")
    return joblib.load(path)


def get_test_sample(n=SAMPLE_N, random_state=C.RANDOM_STATE):
    """Reproducible sample from the 2016-2017 test period (no target-based selection)."""
    df = load_modeling_frame()
    preds = predictor_columns(df)
    test = df[df[C.TIME_COL].isin(SHAP_YEARS)]
    if len(test) > n:
        test = test.sample(n=n, random_state=random_state)
    return test.reset_index(drop=True), preds


def _clean_names(raw_names):
    """Strip the ColumnTransformer 'cat__' prefix for readability."""
    return [n.split("__", 1)[1] if "__" in n else n for n in raw_names]


def _map_to_original(encoded_names, predictors):
    """Map each encoded column to its original variable by longest matching prefix."""
    out = []
    for enc in encoded_names:
        match = None
        for p in predictors:
            if enc == p or enc.startswith(p + "_"):
                if match is None or len(p) > len(match):
                    match = p
        out.append(match if match else enc)
    return out


# --------------------------------------------------------------------------- #
# SHAP computation + validation
# --------------------------------------------------------------------------- #
def compute_shap(pipe, sample, predictors):
    prep = pipe.named_steps["prep"]
    clf = pipe.named_steps["clf"]

    X = sample[predictors]
    # The saved Phase-3 encoder outputs a SPARSE matrix, and XGBoost was trained
    # on it treating structural zeros as MISSING. We must explain the model on
    # that same sparse representation - densifying makes zeros explicit and
    # changes XGBoost's predictions drastically. Xd (dense) is used only for
    # displaying 0/1 feature values in the plots.
    import scipy.sparse as sp
    Xs = prep.transform(X)
    if not sp.issparse(Xs):
        Xs = sp.csr_matrix(Xs)
    Xd = np.asarray(Xs.toarray(), dtype=np.float64)

    enc_raw = list(prep.get_feature_names_out())
    enc = _clean_names(enc_raw)
    orig = _map_to_original(enc, predictors)

    explainer = shap.TreeExplainer(clf)
    shap_vals = np.array(explainer.shap_values(Xs))     # sparse -> correct semantics
    base = float(np.ravel(explainer.expected_value)[0])
    # Explanation for waterfall plots (values in log-odds; data = dense 0/1 view)
    expl = shap.Explanation(values=shap_vals, base_values=np.full(len(Xd), base),
                            data=Xd, feature_names=enc)

    proba = pipe.predict_proba(X)[:, 1]
    pred = (proba >= 0.5).astype(int)

    # ---- validation checks ---------------------------------------------- #
    margin = clf.predict(Xs, output_margin=True)
    checks = {
        "n_shap_features == n_model_features": shap_vals.shape[1] == Xs.shape[1] == len(enc),
        "feature_names_all_map_to_original": all(o in predictors for o in orig),
        "no_forbidden_leakage_feature": not any(o in FORBIDDEN for o in orig),
        "estimator_matches_pipeline_proba": float(np.max(np.abs(
            clf.predict_proba(Xs)[:, 1] - proba))) < 1e-6,
        "shap_additivity_ok(logodds)": bool(np.allclose(
            shap_vals.sum(1) + base, margin, atol=1e-3)),
    }

    return {"Xt": Xd, "enc": enc, "orig": orig, "shap": shap_vals, "base": base,
            "proba": proba, "pred": pred, "explanation": expl, "checks": checks}


# --------------------------------------------------------------------------- #
# Global
# --------------------------------------------------------------------------- #
def global_importance(res):
    mean_abs = np.abs(res["shap"]).mean(axis=0)
    g = pd.DataFrame({"encoded_feature": res["enc"], "original_feature": res["orig"],
                      "mean_abs_shap": mean_abs}).sort_values("mean_abs_shap", ascending=False)
    by_orig = (g.groupby("original_feature")["mean_abs_shap"].sum()
               .sort_values(ascending=False).rename("mean_abs_shap").reset_index())
    return g.reset_index(drop=True), by_orig


def plot_summary(res, path, max_display=15):
    plt.figure()
    shap.summary_plot(res["shap"], res["Xt"], feature_names=res["enc"],
                      max_display=max_display, show=False)
    plt.title("SHAP Summary - XGBoost Resistance Prediction")
    plt.tight_layout(); plt.savefig(path, dpi=120, bbox_inches="tight"); plt.close()


def plot_global_original(by_orig, path):
    fig, ax = plt.subplots(figsize=(7, 4.2))
    s = by_orig.sort_values("mean_abs_shap")
    ax.barh(s["original_feature"], s["mean_abs_shap"], color="#4C6F9F")
    for i, v in enumerate(s["mean_abs_shap"]):
        ax.text(v, i, f" {v:.3f}", va="center", fontsize=8)
    ax.set_xlabel("Mean |SHAP| (log-odds, summed over categories)")
    ax.set_title("Global Feature Importance - SHAP")
    fig.tight_layout(); fig.savefig(path, dpi=120); plt.close(fig)


# --------------------------------------------------------------------------- #
# Local (reproducible: highest and lowest predicted probability in the sample)
# --------------------------------------------------------------------------- #
def local_examples(res, sample, predictors):
    idx_high = int(np.argmax(res["proba"]))
    idx_low = int(np.argmin(res["proba"]))
    rows = []
    per_example_expl = {}
    for label, idx in [("high_probability", idx_high), ("low_probability", idx_low)]:
        # grouped SHAP per original variable for this observation
        sv = res["shap"][idx]
        grouped = pd.Series(sv, index=res["orig"]).groupby(level=0).sum()
        for feat in predictors:
            raw_val = sample.iloc[idx][feat]
            rows.append({
                "example": label, "original_feature": feat,
                "feature_value": "Missing" if pd.isna(raw_val) else str(raw_val),
                "grouped_shap_logodds": round(float(grouped.get(feat, 0.0)), 4),
                "predicted_probability": round(float(res["proba"][idx]), 4),
                "predicted_class": int(res["pred"][idx]),
                "true_target": int(sample.iloc[idx]["target"]),
            })
        per_example_expl[label] = (idx, res["explanation"][idx])
    return pd.DataFrame(rows), per_example_expl


def plot_waterfall(expl_row, path, title):
    plt.figure()
    shap.plots.waterfall(expl_row, max_display=12, show=False)
    plt.title(title)
    plt.tight_layout(); plt.savefig(path, dpi=120, bbox_inches="tight"); plt.close()


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #
def run(save=True):
    pipe = load_pipeline()
    sample, predictors = get_test_sample()
    res = compute_shap(pipe, sample, predictors)
    g_enc, g_orig = global_importance(res)
    local_df, expl_map = local_examples(res, sample, predictors)

    if save:
        REPORTS_DIR.mkdir(parents=True, exist_ok=True); FIG_DIR.mkdir(parents=True, exist_ok=True)
        g_enc.to_csv(REPORTS_DIR / "phase6_shap_global.csv", index=False)
        local_df.to_csv(REPORTS_DIR / "phase6_shap_local.csv", index=False)
        plot_summary(res, FIG_DIR / "shap_summary.png")
        plot_global_original(g_orig, FIG_DIR / "shap_global_original_features.png")
        plot_waterfall(expl_map["high_probability"][1],
                       FIG_DIR / "shap_high_probability_waterfall.png",
                       "Local SHAP - highest predicted resistance (log-odds)")
        plot_waterfall(expl_map["low_probability"][1],
                       FIG_DIR / "shap_low_probability_waterfall.png",
                       "Local SHAP - lowest predicted resistance (log-odds)")
    return res, g_enc, g_orig, local_df


if __name__ == "__main__":
    res, g_enc, g_orig, local_df = run(save=True)
    pd.set_option("display.width", 150)
    print(f"SHAP output space: log-odds (raw margin). Base value = {res['base']:.4f}")
    print(f"Explained {res['shap'].shape[0]} test isolates x {res['shap'].shape[1]} encoded features")
    print("\n=== Validation checks ===")
    for k, v in res["checks"].items():
        print(f"  {'OK ' if v else 'XX '}{k}: {v}")
    print("\n=== Top 12 encoded features by mean|SHAP| ===")
    print(g_enc.head(12).to_string(index=False))
    print("\n=== Original-variable importance (mean|SHAP| summed) ===")
    print(g_orig.to_string(index=False))
    print("\n=== Local explanations (grouped per original variable, log-odds) ===")
    print(local_df.to_string(index=False))
