"""Modèles, split et évaluation (Parties VIII-IX).

Source unique des modèles : `build_model` / `get_models`. L'application, les scripts d'évaluation
(`python -m src.evaluate`) et les documents utilisent exactement les mêmes réglages.
"""
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (balanced_accuracy_score, confusion_matrix, f1_score, precision_recall_curve,
                             precision_score, recall_score)
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

MODEL_NAMES = ["Decision Tree", "Random Forest", "KNN", "SVM"]


class UnderSampled(BaseEstimator, ClassifierMixin):
    """Sous-échantillonne la classe 0 à l'entraînement : on garde toutes les fenêtres de crise (classes != 0)
    et `ratio` fois plus de fenêtres hors crise, tirées au hasard. Le KNN n'a pas de `class_weight` :
    c'est la façon de lui faire voir autant de crises que de non-crises."""

    def __init__(self, estimator=None, ratio=5, seed=0):
        self.estimator = estimator
        self.ratio = ratio
        self.seed = seed

    def fit(self, X, y):
        X, y = np.asarray(X), np.asarray(y)
        rng = np.random.default_rng(self.seed)
        pos, neg = np.where(y != 0)[0], np.where(y == 0)[0]
        keep = rng.choice(neg, min(len(neg), self.ratio * len(pos)), replace=False)
        idx = np.concatenate([pos, keep])
        self.estimator_ = clone(self.estimator).fit(X[idx], y[idx])
        self.classes_ = self.estimator_.classes_
        return self

    def predict_proba(self, X):
        return self.estimator_.predict_proba(np.asarray(X))

    def predict(self, X):
        return self.estimator_.predict(np.asarray(X))


class Augmented(BaseEstimator, ClassifierMixin):
    """Data augmentation à l'entraînement : sur-échantillonne les fenêtres de crise (classes != 0) en créant
    `factor` fois plus d'exemples synthétiques = fenêtre de crise réelle + bruit gaussien (jitter), de
    écart-type `noise` x l'écart-type de chaque feature sur les crises. Appliqué APRÈS la standardisation,
    uniquement sur le jeu d'entraînement : le test (et le patient laissé de côté) reste intact."""

    def __init__(self, estimator=None, factor=2, noise=0.1, seed=0):
        self.estimator = estimator
        self.factor = factor
        self.noise = noise
        self.seed = seed

    def fit(self, X, y):
        X, y = np.asarray(X, dtype=float), np.asarray(y)
        rng = np.random.default_rng(self.seed)
        pos = np.where(y != 0)[0]
        if len(pos) and self.factor > 0:
            src = rng.choice(pos, int(round(self.factor * len(pos))), replace=True)
            sd = X[pos].std(axis=0)
            Xa = X[src] + rng.normal(0.0, 1.0, (len(src), X.shape[1])) * sd * self.noise
            X, y = np.vstack([X, Xa]), np.concatenate([y, y[src]])
        self.estimator_ = clone(self.estimator).fit(X, y)
        self.classes_ = self.estimator_.classes_
        return self

    def predict_proba(self, X):
        return self.estimator_.predict_proba(np.asarray(X))

    def predict(self, X):
        return self.estimator_.predict(np.asarray(X))


def build_model(name, balanced=True, seed=42, rf_min_leaf=2, knn_k=31, knn_undersample=5, augment=False,
                aug_factor=2, aug_noise=0.1):
    """Construit UN modèle (Pipeline StandardScaler + classifieur).

    balanced        pondération des classes (DT, RF, SVM) ; sous-échantillonnage pour le KNN
    rf_min_leaf     taille minimale des feuilles du Random Forest (2 par défaut, 5 = feuilles plus larges)
    knn_k           nombre de voisins du KNN (31 : plus stable que 7, voir l'évaluation)
    knn_undersample nb de non-crises gardées par crise à l'entraînement du KNN (None pour désactiver)
    augment         data augmentation des crises (jitter gaussien), `aug_factor` x `aug_noise` : voir `Augmented`
    """
    cw = "balanced" if balanced else None
    if name == "Decision Tree":
        clf = DecisionTreeClassifier(max_depth=8, min_samples_leaf=5, class_weight=cw, random_state=seed)
    elif name == "Random Forest":
        clf = RandomForestClassifier(n_estimators=200, min_samples_leaf=rf_min_leaf, class_weight=cw, n_jobs=-1, random_state=seed)
    elif name == "KNN":
        clf = KNeighborsClassifier(n_neighbors=knn_k, weights="distance")
        if balanced and knn_undersample:
            clf = UnderSampled(clf, ratio=knn_undersample, seed=seed)
    elif name == "SVM":
        clf = SVC(kernel="rbf", C=1.0, class_weight=cw, probability=True, random_state=seed)
    else:
        raise ValueError(f"modèle inconnu : {name}")
    if augment:
        clf = Augmented(clf, factor=aug_factor, noise=aug_noise, seed=seed)
    return make_pipeline(StandardScaler(), clf)


def get_models(balanced=True, seed=42, **kw):
    return {n: build_model(n, balanced, seed, **kw) for n in MODEL_NAMES}


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
    return {"Balanced Accuracy": balanced_accuracy_score(y, p), "Precision": precision_score(y, p, **kw),
            "Recall": recall_score(y, p, **kw), "F1-score": f1_score(y, p, **kw)}


THR_MIN, THR_MAX = 0.1, 0.9      # garde-fou : un arbre seul donne des probabilités ~0/1, le seuil optimal y est instable


def learn_threshold(y, s):
    """Seuil de probabilité qui maximise le F1 sur (y, s), borné à [0,1 ; 0,9]. À APPRENDRE sur des données qui ne sont pas le test."""
    pr, rc, th = precision_recall_curve(y, s)
    return float(np.clip(th[np.argmax((2 * pr * rc / (pr + rc + 1e-12))[:-1])], THR_MIN, THR_MAX))


def _inner_split(y, groups, seed):
    """Sous-ensemble de validation (20 %) du train, stratifié ; par enregistrement si `groups` est fourni."""
    idx = np.arange(len(y))
    if groups is None:
        return train_test_split(idx, test_size=0.2, stratify=y, random_state=seed)
    for s in range(seed, seed + 50):
        a, b = next(GroupShuffleSplit(1, test_size=0.2, random_state=s).split(idx, groups=groups))
        if y[b].sum() > 20 and y[a].sum() > 20:
            break
    return a, b


def evaluate_all(models, Xtr, ytr, Xte, yte, learn_thr=False, groups=None, seed=42):
    """Entraîne et évalue chaque modèle. Avec `learn_thr`, le seuil de décision est appris sur 20 % du train (jamais sur le test)
    puis appliqué au test ; `res.attrs["seuils"]` contient le seuil de chaque modèle."""
    rows, cms, preds, fitted, thrs, split_info = [], {}, {}, {}, {}, None
    Xtr, ytr = np.asarray(Xtr), np.asarray(ytr)
    for name, m in models.items():
        thr = 0.5
        if learn_thr:
            iin, iva = _inner_split(ytr, None if groups is None else np.asarray(groups), seed)
            split_info = dict(n_fit=len(iin), pos_fit=int((ytr[iin] != 0).sum()), n_val=len(iva), pos_val=int((ytr[iva] != 0).sum()))
            m.fit(Xtr[iin], ytr[iin])
            thr = learn_threshold(ytr[iva], m.predict_proba(Xtr[iva])[:, 1])
        else:
            m.fit(Xtr, ytr)
        p = (m.predict_proba(Xte)[:, 1] > thr).astype(int)
        rows.append({"Modèle": name, **metrics(yte, p)})
        cms[name] = confusion_matrix(yte, p, labels=[0, 1])
        preds[name], fitted[name], thrs[name] = p, m, thr
    res = pd.DataFrame(rows).set_index("Modèle")
    res.attrs["seuils"] = thrs
    res.attrs["validation"] = split_info      # None sans seuil appris
    return res, cms, preds, fitted


def loo_proba(ds, feats, model="Random Forest", seed=42, n_jobs=4, **kw):
    """Probabilité de crise par fenêtre, prédite par un modèle qui n'a JAMAIS vu le patient (leave-one-recording-out).

    Les réglages (`kw`) sont ceux de `build_model`. Les 22 entraînements sont indépendants : ils sont
    parallélisés, sauf pour le Random Forest qui l'est déjà en interne.
    """
    X = ds[list(feats)].to_numpy()
    y = ds["Class"].to_numpy()
    groups = ds["rec_id"].to_numpy()
    recs = pd.unique(groups)

    def one(rec):
        te = groups == rec
        m = build_model(model, seed=seed, **kw).fit(X[~te], y[~te])
        return rec, m.predict_proba(X[te])[:, 1]

    res = [one(r) for r in recs] if model == "Random Forest" else Parallel(n_jobs=n_jobs)(delayed(one)(r) for r in recs)
    out = np.zeros(len(ds))
    for rec, p in res:
        out[groups == rec] = p
    return out
