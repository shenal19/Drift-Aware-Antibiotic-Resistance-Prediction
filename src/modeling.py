"""
Phase 3 - Baseline ML for E. coli -> Levofloxacin.

Fixed temporal benchmark (config): train 2004-2014, validate 2015, test 2016-2017.
Three models, each a full sklearn Pipeline so preprocessing (impute + one-hot) is
fit on TRAINING data only and travels with the model. No random split, no leakage,
no calibration/SHAP/drift (later phases).

Run:  python -m src.modeling
"""

from __future__ import annotations
import warnings

import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (roc_auc_score, average_precision_score, brier_score_loss,
                             precision_recall_fscore_support, accuracy_score,
                             confusion_matrix, roc_curve, precision_recall_curve)

from . import config as C
from .preprocessing import build_feature_encoder

MODELS_DIR = C.PROJECT_ROOT / "models"
FIG_DIR = C.PROJECT_ROOT / "reports" / "figures"
REPORTS_DIR = C.PROJECT_ROOT / "reports"

# XGBoost with documented fallback.
try:
    from xgboost import XGBClassifier
    _HAS_XGB = True
except ImportError:                                   # pragma: no cover
    from sklearn.ensemble import HistGradientBoostingClassifier
    _HAS_XGB = False

PLOT_ORDER = ["Logistic Regression", "Random Forest", "XGBoost"]
FILE_KEY = {"Logistic Regression": "logistic",
            "Random Forest": "random_forest",
            "XGBoost": "xgboost"}


# --------------------------------------------------------------------------- #
# Data + temporal split
# --------------------------------------------------------------------------- #
def load_modeling_frame() -> pd.DataFrame:
    path = C.PROCESSED_DIR / f"{C.SPECIES.split()[0].lower()}_{C.ANTIBIOTIC.lower()}.parquet"
    df = pd.read_parquet(path)
    df[C.TIME_COL] = df[C.TIME_COL].astype(int)
    # Parquet stores categoricals as pandas 'string' (arrow) dtype whose missing
    # value is pd.NA; sklearn's SimpleImputer can't mask pd.NA. Convert predictor
    # columns to plain object with np.nan so imputation works.
    for c in df.columns:
        if c in (C.ID_COL, C.TIME_COL, "target"):
            continue
        col = df[c].astype(object)
        df[c] = col.where(pd.notna(col), np.nan)
    return df


def predictor_columns(df: pd.DataFrame) -> list:
    """Predictors = everything that isn't the id, the split key, or the target."""
    return [c for c in df.columns if c not in (C.ID_COL, C.TIME_COL, "target")]


def temporal_split(df: pd.DataFrame):
    """Split strictly by Year into train / val / test. Nothing is shuffled."""
    preds = predictor_columns(df)
    parts = {}
    for name, years in [("train", C.TRAIN_YEARS), ("val", C.VAL_YEARS), ("test", C.TEST_YEARS)]:
        sub = df[df[C.TIME_COL].isin(years)]
        parts[name] = {"X": sub[preds], "y": sub["target"].values,
                       "year": sub[C.TIME_COL].values}
    return parts, preds


# --------------------------------------------------------------------------- #
# Models
# --------------------------------------------------------------------------- #
def build_models(y_train: np.ndarray, predictors: list) -> dict:
    """
    One pipeline per model = (train-fitted encoder) + classifier.

    Class imbalance (~34% resistant) is moderate, so we use class weighting
    rather than resampling (no SMOTE/oversampling, per spec). Weighting lifts
    recall on the resistant (minority) class, which is the costly error to miss;
    the trade-off is that weighted probabilities are less calibrated (Brier
    looks worse) - that is expected and is exactly what Phase 7 will repair.
    """
    def pipe(clf):
        return Pipeline([("prep", build_feature_encoder(predictors)), ("clf", clf)])

    models = {
        "Logistic Regression": pipe(LogisticRegression(
            solver="lbfgs", max_iter=2000, class_weight="balanced",
            random_state=C.RANDOM_STATE)),
        "Random Forest": pipe(RandomForestClassifier(
            n_estimators=300, class_weight="balanced", n_jobs=-1,
            random_state=C.RANDOM_STATE)),
    }

    if _HAS_XGB:
        # scale_pos_weight = negatives/positives balances the classes for XGB.
        pos = int(y_train.sum()); neg = int(len(y_train) - pos)
        spw = neg / max(pos, 1)
        models["XGBoost"] = pipe(XGBClassifier(
            n_estimators=400, max_depth=4, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, scale_pos_weight=spw,
            eval_metric="logloss", random_state=C.RANDOM_STATE, n_jobs=-1,
            tree_method="hist"))
    else:                                             # documented fallback
        models["XGBoost"] = pipe(HistGradientBoostingClassifier(
            max_iter=400, max_depth=4, learning_rate=0.05,
            random_state=C.RANDOM_STATE))
    return models


# --------------------------------------------------------------------------- #
# Evaluation
# --------------------------------------------------------------------------- #
def evaluate(pipe, X, y, threshold=0.5) -> dict:
    """Threshold-independent (AUCs, Brier) + operating-point (P/R/F1 at 0.5)."""
    proba = pipe.predict_proba(X)[:, 1]
    pred = (proba >= threshold).astype(int)
    p, r, f1, _ = precision_recall_fscore_support(
        y, pred, average="binary", zero_division=0)
    return {
        "ROC-AUC": roc_auc_score(y, proba),
        "PR-AUC": average_precision_score(y, proba),
        "Precision": p, "Recall": r, "F1": f1,
        "Brier": brier_score_loss(y, proba),
        "Accuracy": accuracy_score(y, pred),
        "_proba": proba, "_pred": pred,
    }


def _metrics_row(m):
    return {k: m[k] for k in ["ROC-AUC", "PR-AUC", "Precision", "Recall", "F1", "Brier", "Accuracy"]}


# --------------------------------------------------------------------------- #
# Figures
# --------------------------------------------------------------------------- #
def plot_roc(fitted, parts, path):
    fig, ax = plt.subplots(figsize=(5.5, 5))
    for name in PLOT_ORDER:
        proba = fitted[name].predict_proba(parts["test"]["X"])[:, 1]
        fpr, tpr, _ = roc_curve(parts["test"]["y"], proba)
        ax.plot(fpr, tpr, lw=2, label=f"{name} (AUC={roc_auc_score(parts['test']['y'], proba):.3f})")
    ax.plot([0, 1], [0, 1], "--", color="grey", lw=1)
    ax.set_xlabel("False positive rate"); ax.set_ylabel("True positive rate")
    ax.set_title("ROC - test 2016-2017 (E. coli - Levofloxacin)")
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout(); fig.savefig(path, dpi=120); plt.close(fig)


def plot_pr(fitted, parts, path):
    fig, ax = plt.subplots(figsize=(5.5, 5))
    base = parts["test"]["y"].mean()
    for name in PLOT_ORDER:
        proba = fitted[name].predict_proba(parts["test"]["X"])[:, 1]
        prec, rec, _ = precision_recall_curve(parts["test"]["y"], proba)
        ap = average_precision_score(parts["test"]["y"], proba)
        ax.plot(rec, prec, lw=2, label=f"{name} (PR-AUC={ap:.3f})")
    ax.axhline(base, ls="--", color="grey", lw=1, label=f"No-skill ({base:.2f})")
    ax.set_xlabel("Recall"); ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall - test 2016-2017")
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout(); fig.savefig(path, dpi=120); plt.close(fig)


def plot_confusion(name, y_true, y_pred, path):
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    fig, ax = plt.subplots(figsize=(4, 3.6))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1]); ax.set_xticklabels(["S (0)", "R (1)"])
    ax.set_yticks([0, 1]); ax.set_yticklabels(["S (0)", "R (1)"])
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
    ax.set_title(f"Confusion - {name}\n(test 2016-2017)")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{cm[i, j]:,}", ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    fig.colorbar(im, fraction=0.046, pad=0.04)
    fig.tight_layout(); fig.savefig(path, dpi=120); plt.close(fig)


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #
def run(save=True):
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    df = load_modeling_frame()
    parts, predictors = temporal_split(df)
    models = build_models(parts["train"]["y"], predictors)

    fitted, results, pred_rows = {}, {"val": {}, "test": {}}, []
    for name, pipe in models.items():
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            pipe.fit(parts["train"]["X"], parts["train"]["y"])
        fitted[name] = pipe
        for split in ("val", "test"):
            m = evaluate(pipe, parts[split]["X"], parts[split]["y"])
            results[split][name] = _metrics_row(m)
            if split == "test":
                pred_rows.append(pd.DataFrame({
                    "year": parts["test"]["year"],
                    "true_target": parts["test"]["y"],
                    "predicted_class": m["_pred"],
                    "resistance_probability": m["_proba"],
                    "model": name,
                }))
        if save:
            joblib.dump(pipe, MODELS_DIR / f"{FILE_KEY[name]}.joblib")

    if save:
        plot_roc(fitted, parts, FIG_DIR / "roc_curves.png")
        plot_pr(fitted, parts, FIG_DIR / "precision_recall_curves.png")
        for name in PLOT_ORDER:
            proba = fitted[name].predict_proba(parts["test"]["X"])[:, 1]
            pred = (proba >= 0.5).astype(int)
            plot_confusion(name, parts["test"]["y"], pred,
                           FIG_DIR / f"confusion_matrix_{FILE_KEY[name]}.png")
        preds = pd.concat(pred_rows, ignore_index=True)
        REPORTS_DIR.mkdir(parents=True, exist_ok=True)
        preds.to_parquet(REPORTS_DIR / "phase3_test_predictions.parquet", index=False)
        pd.DataFrame(results["test"]).T.to_csv(REPORTS_DIR / "phase3_test_metrics.csv")

    return fitted, parts, results, predictors


def preliminary_importance(fitted, predictors):
    """
    Simple, aggregated importance by ORIGINAL feature (not per one-hot column).
    LR: sum of |coef| over a feature's dummies. RF: sum of impurity importances.
    This is a preliminary view only - SHAP is Phase 6.
    """
    out = {}
    ohe = fitted["Logistic Regression"].named_steps["prep"]
    names = ohe.get_feature_names_out()
    base = [n.split("__", 1)[1].rsplit("_", 1)[0] for n in names]  # strip 'cat__' + dummy tail

    def agg(values):
        s = pd.Series(values, index=base).groupby(level=0).sum()
        return (s / s.sum()).sort_values(ascending=False).round(3)

    out["LogReg |coef| share"] = agg(np.abs(
        fitted["Logistic Regression"].named_steps["clf"].coef_.ravel()))
    out["RF importance share"] = agg(
        fitted["Random Forest"].named_steps["clf"].feature_importances_)
    return pd.DataFrame(out)


def print_report(parts, results, predictors, fitted):
    n_feat = fitted["Logistic Regression"].named_steps["prep"].get_feature_names_out().shape[0]
    print("=" * 74)
    print(f"PHASE 3 REPORT  |  {C.SPECIES} -> {C.ANTIBIOTIC}  (XGBoost={'real' if _HAS_XGB else 'FALLBACK HistGB'})")
    print("=" * 74)
    for split in ("train", "val", "test"):
        y = parts[split]["y"]
        yrs = f"{min(parts[split]['year'])}-{max(parts[split]['year'])}"
        print(f"{split:5s}: {len(y):>6,} rows  years {yrs}  resistant={y.mean():.1%}")
    print(f"Features after one-hot encoding: {n_feat}")
    for split in ("val", "test"):
        print(f"\n--- {split.upper()} metrics ---")
        print(pd.DataFrame(results[split]).T.round(3).to_string())
    print("\n--- Preliminary aggregated importance (NOT SHAP) ---")
    print(preliminary_importance(fitted, predictors).to_string())


if __name__ == "__main__":
    fitted, parts, results, predictors = run(save=True)
    print_report(parts, results, predictors, fitted)
