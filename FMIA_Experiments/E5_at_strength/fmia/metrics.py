"""MIA metrics: AUC, TPR at fixed low FPR, best balanced accuracy, ROC points."""
import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve

FPR_GRID = np.logspace(-4, 0, 41)


def _clean(s):
    s = np.asarray(s, dtype=np.float64)
    if not np.isfinite(s).all():
        fin = s[np.isfinite(s)]
        lo = fin.min() if fin.size else 0.0
        hi = fin.max() if fin.size else 0.0
        s = np.nan_to_num(s, nan=lo, posinf=hi, neginf=lo)
    return s


def auc(y, s):
    y = np.asarray(y, bool)
    if y.all() or (~y).all():
        return float("nan")
    return float(roc_auc_score(y, _clean(s)))


def tpr_at(fpr, tpr, t):
    ok = fpr <= t
    return float(tpr[ok].max()) if ok.any() else 0.0


def summarize(y, s, with_roc=False):
    y = np.asarray(y, bool)
    s = _clean(s)
    fpr, tpr, _ = roc_curve(y, s)
    out = {"auc": float(roc_auc_score(y, s)),
           "tpr@1%": tpr_at(fpr, tpr, 0.01),
           "tpr@0.1%": tpr_at(fpr, tpr, 0.001),
           "acc": float(np.max(1 - (fpr + (1 - tpr)) / 2))}
    if with_roc:
        out["roc_tpr"] = [tpr_at(fpr, tpr, t) for t in FPR_GRID]
    return out
