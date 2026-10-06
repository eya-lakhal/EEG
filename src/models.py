"""Modèles, split et évaluation (Parties VIII-IX)."""
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score)
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier


def get_models(balanced=True, seed=42):
    cw = "balanced" if balanced else None
    return {
        "Decision Tree": make_pipeline(StandardScaler(), DecisionTreeClassifier(max_depth=8, min_samples_leaf=5, class_weight=cw, random_state=seed)),
        "Random Forest": make_pipeline(StandardScaler(), RandomForestClassifier(n_estimators=200, min_samples_leaf=2, class_weight=cw, n_jobs=-1, random_state=seed)),
        "KNN": make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=7, weights="distance", n_jobs=-1)),
        "SVM": make_pipeline(StandardScaler(), SVC(kernel="rbf", C=1.0, class_weight=cw, probability=True, random_state=seed)),
    }


def split_data(df, feats, target="Class", mode="stratifie", test_size=0.2, seed=42):
    """mode 'stratifie' : 80/20 stratifié sur les fenêtres (demandé par le sujet).
    mode 'patient' : 80/20 par enregistrement (aucune fuite entre fenêtres chevauchantes)."""
    if mode == "stratifie":
        tr, te = train_test_split(df.index, test_size=test_size, stratify=df[target], random_state=seed)
    else:
        # on tente plusieurs graines jusqu'à avoir des crises dans train ET test
        for s in range(seed, seed + 50):
            tr, te = next(GroupShuffleSplit(1, test_size=test_size, random_state=s).split(df, groups=df["rec_id"]))
            tr, te = df.index[tr], df.index[te]
            if df.loc[te, target].sum() > 20 and df.loc[tr, target].sum() > 20:
                break
    return df.loc[tr], df.loc[te]


def metrics(y, p, avg="binary"):
    kw = dict(zero_division=0) if avg == "binary" else dict(zero_division=0, average=avg)
    return {"Accuracy": accuracy_score(y, p), "Precision": precision_score(y, p, **kw),
            "Recall": recall_score(y, p, **kw), "F1-score": f1_score(y, p, **kw)}


def evaluate_all(models, Xtr, ytr, Xte, yte):
    rows, cms, preds, fitted = [], {}, {}, {}
    for name, m in models.items():
        m.fit(Xtr, ytr)
        p = m.predict(Xte)
        rows.append({"Modèle": name, **metrics(yte, p)})
        cms[name] = confusion_matrix(yte, p, labels=[0, 1])
        preds[name] = p
        fitted[name] = m
    return pd.DataFrame(rows).set_index("Modèle"), cms, preds, fitted


def loo_proba(ds, feats, model="Random Forest", seed=42):
    """Probabilité de crise par fenêtre, prédite par un modèle qui n'a JAMAIS vu le patient (leave-one-recording-out)."""
    out = np.zeros(len(ds))
    for rec in ds["rec_id"].unique():
        te = (ds["rec_id"] == rec).to_numpy()
        if model == "Random Forest":
            m = make_pipeline(StandardScaler(), RandomForestClassifier(n_estimators=120, min_samples_leaf=2, class_weight="balanced", n_jobs=-1, random_state=seed))
        else:
            m = get_models(True, seed)[model]
        m.fit(ds.loc[~te, feats], ds.loc[~te, "Class"])
        out[te] = m.predict_proba(ds.loc[te, feats])[:, 1]
    return out
