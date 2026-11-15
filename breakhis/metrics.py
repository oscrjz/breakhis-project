"""Image-level and patient-level metrics."""
import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, f1_score,
                             recall_score, roc_auc_score)


def compute_metrics(y_true, prob_malignant, patients, threshold=0.5):
    y_true = np.asarray(y_true)
    prob = np.asarray(prob_malignant)
    y_pred = (prob >= threshold).astype(int)
    out = {
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, average="macro"),
        "benign_recall": recall_score(y_true, y_pred, pos_label=0),
        "malignant_recall": recall_score(y_true, y_pred, pos_label=1),
        "auroc": roc_auc_score(y_true, prob) if len(set(y_true)) > 1 else float("nan"),
    }
    df = pd.DataFrame({"patient": patients, "y": y_true, "pred": y_pred, "prob": prob})
    # Patient recognition rate as in Spanhol et al. (2016): mean over patients of the
    # share of that patient's images classified correctly.
    out["patient_recognition_rate"] = df.assign(ok=df.y == df.pred).groupby("patient")["ok"].mean().mean()
    # Patient-level diagnosis: average each patient's probabilities into one call.
    pp = df.groupby("patient").agg(y=("y", "first"), prob=("prob", "mean"))
    out["patient_accuracy"] = float(((pp.prob >= threshold).astype(int) == pp.y).mean())
    return {k: float(v) for k, v in out.items()}
