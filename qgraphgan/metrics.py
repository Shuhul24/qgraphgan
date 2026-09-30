"""Link-prediction and node-classification metrics."""

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import cross_val_score


def y_for_edges(pos, neg):
    pos_set = {tuple(sorted(e)) for e in pos}
    return [1 if tuple(sorted(e)) in pos_set else 0 for e in pos + neg]


def safe_auc_ap(y, s):
    if len(set(y)) < 2:
        return np.nan, np.nan
    return roc_auc_score(y, s), average_precision_score(y, s)


def precision_at_k(y, s, k):
    top = sorted(zip(s, y), reverse=True)[:k]
    return sum(v for _, v in top) / k if top else np.nan


def metric_row(method, y, s):
    """AUC, AP and Precision@{5,10,20} (P@k only when k <= number of test pairs)."""
    auc, ap = safe_auc_ap(y, s)
    row = {"method": method, "auc": auc, "ap": ap}
    for k in [5, 10, 20]:
        if k <= len(y):
            row[f"p@{k}"] = precision_at_k(y, s, k)
    return row


def node_f1(emb, labels, seed):
    """5-fold macro-F1 of logistic regression on embeddings (NaN if not applicable)."""
    if labels is None or len(set(labels)) < 2 or len(labels) < 20:
        return np.nan
    try:
        clf = LogisticRegression(max_iter=1000, random_state=seed)
        return float(cross_val_score(clf, emb, labels, cv=5, scoring="f1_macro").mean())
    except Exception:
        return np.nan
