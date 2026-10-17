"""EEG · Absence - détection de crises d'absence, application interactive patient par patient.

Lancer :  streamlit run app.py
Interface : ui/theme.py (palette, CSS, gabarit des graphiques) et ui/components.py (composants).
L'ancienne version organisée par questions du sujet est dans app_questions.py.
Les calculs scientifiques sont dans src/ : ce fichier ne fait qu'orchestrer et afficher.
"""
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from PIL import Image
from plotly.subplots import make_subplots
from scipy.signal import spectrogram, welch

os.environ.setdefault("LOKY_MAX_CPU_COUNT", "4")
ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))
from src.data_loader import FS, list_recordings, load_annotations, load_recording
from src.dataset import feature_cols, load_or_build_dataset
from src.detection import match_intervals, predictions_to_intervals
from src.features import BANDS, spectrum
from src.models import evaluate_all, get_models, loo_proba, metrics, split_data
from src.preprocessing import bandpass, robust_scale, select_channels
from src.selection import rank_features, select_uncorrelated
from ui import components as ui
from ui import theme as T

st.set_page_config(page_title="EEG · Absence", page_icon=Image.open(ROOT / "ui" / "assets" / "favicon.png"),
                   layout="wide", initial_sidebar_state="auto")
T.apply_theme()
R = T.ROLE
DASHES = ["solid", "dash", "dot", "dashdot"]      # les séries ne se distinguent pas par la seule couleur


# ============================================================ données (mises en cache)
@st.cache_data
def get_annots():
    return load_annotations()


@st.cache_data(show_spinner="Construction du dataset (fenêtrage + features)... ~1 min la première fois")
def get_dataset(win_s, overlap, pre_s):
    return load_or_build_dataset(win_s, overlap, pre_s)      # cache disque géré dans src/dataset.py


@st.cache_resource(show_spinner="Prétraitement du signal...")
def get_prep(rec, lo=0.5, hi=40.0):
    """(brut, filtré, normalisé, noms de canaux, fs) - tous les canaux EEG du patient."""
    x, ch, fs = load_recording(rec)
    raw, names = select_channels(x, ch)
    filt = bandpass(raw, fs, lo, hi)
    return raw.astype(np.float32), filt.astype(np.float32), robust_scale(filt).astype(np.float32), names, fs


@st.cache_data(show_spinner="Classement des features (4 critères)...")
def get_ranking(win_s, overlap, pre_s):
    ds = get_dataset(win_s, overlap, pre_s)
    return rank_features(ds, [c for c in feature_cols(ds) if c != "Class3"])


@st.cache_data(show_spinner="Prédiction leave-one-patient-out (22 entraînements)... ~1-2 min")
def get_loo(win_s, overlap, pre_s, feats, model, rf_leaf=2, augment=False):
    return loo_proba(get_dataset(win_s, overlap, pre_s), list(feats), model, rf_min_leaf=rf_leaf, augment=augment)


@st.cache_data
def get_rec_table():
    """Durée et nombre de canaux EEG de chaque enregistrement."""
    rows = {}
    for r in list_recordings():
        x_, ch_, _ = load_recording(r)
        rows[r] = {"duree_s": round(len(x_) / FS), "n_ch": len(select_channels(x_, ch_)[1])}
    return pd.DataFrame(rows).T


annots = get_annots()
RECS = list_recordings()
n_cr = annots.groupby("rec_id").size().to_dict()
rec_table = get_rec_table()

# ============================================================ navigation et état
STEPS = [
    dict(label="Vue d'ensemble", title="Vue d'ensemble", help="Le projet, les données et l'état de l'analyse",
         lead="Détection automatique des crises d'absence à partir de signaux EEG, du signal brut aux intervalles de crise. "
              "Choisissez un enregistrement dans le panneau de gauche, puis suivez les étapes dans l'ordre."),
    dict(label="Prétraitement", title="Prétraitement", help="Filtrage et normalisation du signal",
         lead="Observez l'effet de chaque étape de nettoyage du signal avant la segmentation. Ces réglages servent à explorer : "
              "le jeu de données du modèle utilise un filtre 0,5-40 Hz et la normalisation robuste."),
    dict(label="Fenêtre par fenêtre", title="Fenêtre par fenêtre", help="Parcourir l'enregistrement fenêtre par fenêtre",
         lead="Parcourez l'enregistrement fenêtre par fenêtre : signal, probabilité de crise estimée par un modèle qui n'a jamais vu "
              "cet enregistrement, verdict et caractéristiques de la fenêtre."),
    dict(label="Meilleures features", title="Meilleures features", help="Classement et sélection des caractéristiques",
         lead="Les caractéristiques extraites de chaque fenêtre sont classées selon quatre critères, puis les plus discriminantes "
              "et non redondantes sont retenues."),
    dict(label="Classification", title="Classification", help="Comparaison des quatre modèles",
         lead="Entraînement et comparaison de l'arbre de décision, de la forêt aléatoire, du KNN et du SVM sur des fenêtres "
              "étiquetées crise / hors crise, puis résultats patient par patient."),
    dict(label="Détection temporelle", title="Détection temporelle", help="Intervalles de crise dans le temps",
         lead="Reconstruction des intervalles de crise dans le temps à partir des probabilités du modèle, comparés aux "
              "annotations de référence."),
    dict(label="Prévention", title="Prévention (préparation)", help="Étude préparatoire, sans modèle de prévention",
         lead="Étude préparatoire : les caractéristiques changent-elles avant le début d'une crise ? Aucune prédiction de crise "
              "n'est proposée à ce stade."),
]
LABELS = [s["label"] for s in STEPS]


def go_to(i):
    st.session_state["step"] = i


if "step" not in st.session_state:
    try:
        st.session_state["step"] = min(max(int(st.query_params.get("step", 0)), 0), len(STEPS) - 1)
    except (TypeError, ValueError):
        st.session_state["step"] = 0
step = st.session_state["step"]
st.session_state.setdefault("visited", set()).add(step)

# Réglages des étapes : valeurs par défaut identiques à l'ancienne interface, conservées quand on change d'étape
DEFAULTS = {
    "pre_lo": 0.5, "pre_hi": 40.0, "pre_scale": True,
    "win_K": 12, "win_thr": 0.5, "win_model": "Random Forest",
    "feat_K": 15, "feat_corr": 0.9,
    "cls_fs": "Meilleures features", "cls_K": 15, "cls_split": "Par patient (réaliste)", "cls_bal": True, "cls_aug": False, "cls_thr": False,
    "m_pp": "Random Forest", "th_pp": 0.5,
    "det_K": 15, "det_model": "Random Forest", "det_th": 0.5, "det_smooth": 3, "det_mind": 2.0, "det_iou": 0.3,
    "prev_range": (-60, 20),
}
for _k, _v in DEFAULTS.items():
    st.session_state.setdefault(_k, _v)
    st.session_state[_k] = st.session_state[_k]          # un widget absent de l'étape courante garde sa valeur
MODELS = ["Random Forest", "SVM", "KNN", "Decision Tree"]

# ============================================================ barre latérale : contexte d'analyse
label = {r: f"{r}  ·  {n_cr.get(r, 0)} crise(s)" for r in RECS}
if "rec_sel" not in st.session_state and st.query_params.get("rec") in label:
    st.session_state["rec_sel"] = label[st.query_params.get("rec")]

with st.sidebar:
    ui.brand()
    ui.sidebar_label("Enregistrement")
    REC = {v: k for k, v in label.items()}[st.selectbox("Patient / enregistrement", list(label.values()), key="rec_sel")]
    ui.sidebar_label("Segmentation")
    WIN = st.select_slider("Fenêtre (s)", [1.0, 2.0, 4.0], value=2.0)
    OVL = st.select_slider("Recouvrement (%)", [0, 50, 75], value=50) / 100
    PRE = st.slider("Zone pré-crise (s)", 5, 30, 10)
    ui.sidebar_label("Modèles")
    RF_LEAF = st.select_slider("Random Forest : taille min. des feuilles", [2, 5], value=2,
                               help="2 = réglage de départ ; 5 = feuilles plus larges (testé : écart non significatif sur 19 patients)")

ds = get_dataset(WIN, OVL, PRE)
F_ALL = [c for c in feature_cols(ds) if c != "Class3"]
STEP = int(WIN * (1 - OVL) * FS)
sub = annots[annots.rec_id == REC].reset_index(drop=True)
d = ds[ds.rec_id == REC].reset_index(drop=True)
dur_s = int(rec_table.loc[REC, "duree_s"])
n_ch = int(rec_table.loc[REC, "n_ch"])
dur_txt = f"{dur_s // 60} min {dur_s % 60:02d} s"

with st.sidebar:
    ui.facts([("Durée", dur_txt), ("Canaux EEG", n_ch), ("Crises annotées", len(sub)),
              ("Fenêtres", f"{len(d):,}"), ("Part en crise", f"{100 * d.Class.mean():.1f} %")])
    ui.sidebar_note(ui.DISCLAIMER)


# ============================================================ aides
def shade(fig, intervals, color=T.CORAL_LIGHT, **kw):
    """Bandes pleines sous les tracés (couleur et libellé de légende toujours associés)."""
    for a, b in intervals:
        fig.add_vrect(x0=a, x1=b, fillcolor=color, opacity=1, line_width=0, layer="below", **kw)


def best_feats(k, thr):
    return select_uncorrelated(get_ranking(WIN, OVL, PRE), ds, k, thr)


def loo(feats, model):
    """Probabilités leave-one-patient-out ; mémorise la configuration pour l'état de l'analyse."""
    st.session_state.setdefault("loo_cfgs", set()).add((model, len(feats)))
    return get_loo(WIN, OVL, PRE, tuple(feats), model, RF_LEAF, bool(st.session_state.get("cls_aug", False)))


def fig_layout(fig, height, **kw):
    """Hauteur + réglages communs : un titre et une légende ne se chevauchent jamais."""
    fig.update_layout(height=height, **kw)
    if fig.layout.title.text:
        fig.update_layout(title_font=dict(size=13, color=T.SLATE), title_y=0.97, title_yanchor="top")
        if fig.layout.showlegend is not False and "margin" not in kw:
            fig.update_layout(margin=dict(t=84))
    return fig


LEGEND_REF = ("Annotation de référence (crise)", T.CORAL_LIGHT, "fill")
LEGEND_PRED = ("Prédiction du modèle", T.PETROL, "outline")
LEGEND_PRE = (f"Zone pré-critique (≤ {PRE} s avant le début annoté)", T.AMBER_LIGHT, "fill")


# ============================================================ en-tête et navigation principale
S = STEPS[step]
ui.stepper(LABELS, step, st.session_state["visited"], go_to, [s["help"] for s in STEPS])
ui.page_header(step, len(STEPS) - 1, S["title"], S["lead"],
               [("Enregistrement", REC), ("Durée", dur_txt), ("Canaux EEG", n_ch), ("Crises annotées", len(sub)),
                ("Fenêtre", f"{WIN:g} s · {int(OVL * 100)} %")])


# ============================================================ Vue d'ensemble
def page_overview():
    tab = []
    for r in RECS:
        dr, a = ds[ds.rec_id == r], annots[annots.rec_id == r]
        tab.append({"Patient": r, "Durée (s)": int(rec_table.loc[r, "duree_s"]), "Canaux EEG": int(rec_table.loc[r, "n_ch"]),
                    "Crises": len(a), "Durée moy. crise (s)": round(a.duree_s.mean(), 1) if len(a) else None,
                    "Fenêtres": len(dr), "Fenêtres crise": int(dr.Class.sum()), "% crise": round(100 * dr.Class.mean(), 2)})
    tab = pd.DataFrame(tab)
    c = st.columns(4)
    c[0].metric("Patients", len(RECS))
    c[1].metric("Crises annotées", int(tab.Crises.sum()))
    c[2].metric("Fenêtres", f"{len(ds):,}")
    c[3].metric("% fenêtres en crise", f"{100 * ds.Class.mean():.1f} %")

    ui.section("Chaîne de traitement", "Signal brut → filtre 0,5-40 Hz → normalisation robuste par canal → fenêtres → features par canal "
                                      "(agrégées moyenne / max) → sélection → classification → reconstruction des crises → prévention.")
    ui.flow([("Prétraitement", "Filtre passe-bande, normalisation par canal"),
             ("Fenêtres", f"{WIN:g} s, recouvrement {int(OVL * 100)} %"),
             ("Features", f"{len(F_ALL)} caractéristiques par fenêtre"),
             ("Classification", "Arbre, forêt, KNN, SVM"),
             ("Détection", "Intervalles de crise dans le temps"),
             ("Prévention", "Étude préparatoire")])

    left, right = st.columns([3, 2], gap="large")
    with left:
        ui.section("État de l'analyse", "Ce qui a été calculé ou consulté dans cette session.")
        best, cfgs = st.session_state.get("best_feats"), st.session_state.get("loo_cfgs", set())
        ui.status_table([
            ("Données", True, f"{len(RECS)} enregistrements, {int(tab.Crises.sum())} crises annotées valides."),
            ("Prétraitement", True, "Filtre 0,5-40 Hz et normalisation robuste appliqués dans le jeu de données du modèle."),
            ("Fenêtres", True, f"{len(ds):,} fenêtres de {WIN:g} s, recouvrement {int(OVL * 100)} %."),
            ("Features", best is not None, f"{len(F_ALL)} calculées ; " + (f"{len(best)} retenues." if best is not None else "sélection à consulter à l'étape 3.")),
            ("Classification", "cls" in st.session_state, "Résultats disponibles pour les 4 modèles." if "cls" in st.session_state else "Entraînement non lancé dans cette session."),
            ("Détection", bool(cfgs), ("Prédictions calculées : " + ", ".join(f"{m} ({k} features)" for m, k in sorted(cfgs)) + ".") if cfgs else "Prédictions calculées à l'ouverture des étapes 2 ou 5."),
            ("Prévention", 6 in st.session_state["visited"], "Étude préparatoire consultée." if 6 in st.session_state["visited"] else "Étude préparatoire non consultée."),
        ])
    with right:
        ui.section("Accès rapide")
        for i in range(1, len(STEPS)):
            st.button(f"{i}  {STEPS[i]['title']}", key=f"quick_{i}", on_click=go_to, args=(i,), use_container_width=True, help=STEPS[i]["help"])

    res_f, det_f = ROOT / "results" / "summary_models.csv", ROOT / "results" / "summary_detection.csv"
    ui.section("Résultats principaux disponibles", "Évaluation de référence en validation « un patient laissé de côté » (`python -m src.evaluate`).")
    if res_f.exists() and det_f.exists():
        m_, d_ = pd.read_csv(res_f, index_col=0), pd.read_csv(det_f, index_col=0)
        out = pd.DataFrame({"PR-AUC": m_["PR-AUC"], **({"Balanced Accuracy": m_["Balanced Accuracy"]} if "Balanced Accuracy" in m_ else {}), "Recall": m_["Recall"], "Précision": m_["Précision"], "F1": m_["F1"],
                            "Crises retrouvées (strict)": d_["Crises retrouvées (strict)"], "Fausses alarmes / h": d_["Fausses alarmes / h"]}).round(3)
        st.dataframe(out, use_container_width=True)
        st.caption("Seuil de décision 0,5 sur la probabilité de chaque modèle. Appariement strict des intervalles (recouvrement ≥ 0,3). "
                   "Les meilleurs modèles ne sont pas statistiquement distinguables avec 19 patients.")
    else:
        ui.callout("neutral", "Aucun résultat de référence trouvé dans `results/`. Lancez `python -m src.evaluate` pour les produire.")

    ui.section("Enregistrements")
    st.dataframe(tab, use_container_width=True, hide_index=True, height=420)
    fig = px.bar(tab, x="Patient", y="Crises", color_discrete_sequence=[R["reference"]], title="Nombre de crises par patient")
    fig.update_traces(marker_line_width=0)
    st.plotly_chart(fig_layout(fig, 300), use_container_width=True)


# ============================================================ 1 · Prétraitement
def page_preprocessing():
    with ui.panel("Paramètres de prétraitement"):
        c = st.columns([1, 1, 1.4, 1])
        lo = c[0].number_input("Passe-haut (Hz)", 0.1, 5.0, step=0.1, key="pre_lo")
        hi = c[1].number_input("Passe-bas (Hz)", 20.0, 100.0, step=5.0, key="pre_hi")
        scale_on = c[2].toggle("Normalisation robuste (médiane/IQR)", key="pre_scale")
        raw, filt, scaled, names, fs = get_prep(REC, lo, hi)
        c[3].metric("Canaux EEG", len(names))
    ui.callout("neutral", "Le dataset du modèle est construit avec 0,5-40 Hz + normalisation. Ces réglages servent à explorer l'effet du prétraitement.")
    dur = len(raw) / fs
    ui.section("Canaux et plage de temps")
    cc = st.columns([2, 3, 2])
    chs = cc[0].multiselect("Canaux", names, default=[n for n in ["Fp1", "Cz", "O1"] if n in names] or names[:2])
    default_rng = (0.0, min(60.0, dur))
    if len(sub):
        default_rng = (max(0.0, float(sub.debut_s[0]) - 15), min(dur, float(sub.debut_s[0]) + 25))
    t_rng = cc[1].slider("Plage de temps (s)", 0.0, float(dur), default_rng, 1.0)
    i0, i1 = int(t_rng[0] * fs), int(t_rng[1] * fs)
    tt = np.arange(i0, i1) / fs
    ui.section("Résultats")
    tabs = st.tabs(["Signal : brut → filtré → normalisé", "Spectre avant / après", "Spectrogramme", "Qualité des canaux"])
    with tabs[0]:
        ui.legend([LEGEND_REF, ("Signal brut", R["raw"], "line"), ("Signal filtré", R["filtered"], "line"),
                   ("Signal normalisé" if scale_on else "Signal filtré (sans normalisation)", R["normalized"], "line")])
        for cn in chs:
            k = names.index(cn)
            fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.05,
                                subplot_titles=(f"{cn} · brut", "filtré", "normalisé" if scale_on else "filtré (sans normalisation)"))
            fig.add_trace(go.Scattergl(x=tt, y=raw[i0:i1, k], line=dict(color=R["raw"], width=1)), row=1, col=1)
            fig.add_trace(go.Scattergl(x=tt, y=filt[i0:i1, k], line=dict(color=R["filtered"], width=1)), row=2, col=1)
            fig.add_trace(go.Scattergl(x=tt, y=(scaled if scale_on else filt)[i0:i1, k], line=dict(color=R["normalized"], width=1)), row=3, col=1)
            for r_ in (1, 2, 3):
                shade(fig, zip(sub.debut_s, sub.fin_s), row=r_, col=1)
            fig.update_xaxes(range=[t_rng[0], t_rng[1]])
            fig.update_xaxes(title_text="Temps (s)", row=3, col=1)
            fig.update_annotations(x=0, xanchor="left", font=dict(size=11, color=T.SLATE))
            st.plotly_chart(fig_layout(fig, 500, showlegend=False, margin=dict(t=28)), use_container_width=True)
        st.caption("Le filtre retire la dérive lente et le bruit secteur ; la normalisation met tous les patients à la même échelle.")
    with tabs[1]:
        cn = st.selectbox("Canal pour le spectre", names, index=names.index(chs[0]) if chs else 0)
        k = names.index(cn)
        f1, p1 = welch(raw[:, k], fs, nperseg=4 * fs)
        f2, p2 = welch(filt[:, k], fs, nperseg=4 * fs)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=f1, y=p1, name="brut", line=dict(color=R["raw"])))
        fig.add_trace(go.Scatter(x=f2, y=p2, name="filtré", line=dict(color=R["filtered"])))
        for b, (a, z) in BANDS.items():
            fig.add_vrect(x0=a, x1=z, fillcolor=T.IVORY_DEEP, opacity=0.55, line_width=0, layer="below",
                          annotation_text=b, annotation_position="top left", annotation_font=dict(size=10, color=T.SLATE))
        fig.update_layout(xaxis_title="Fréquence (Hz)", yaxis_title="Densité spectrale", yaxis_type="log", xaxis_range=[0, 60])
        st.plotly_chart(fig_layout(fig, 400), use_container_width=True)
    with tabs[2]:
        cn = st.selectbox("Canal", names, index=names.index(chs[0]) if chs else 0, key="spec_ch")
        f, t, S_ = spectrogram(filt[:, names.index(cn)], fs, nperseg=2 * fs, noverlap=fs)
        m = f <= 30
        fig = go.Figure(go.Heatmap(x=t, y=f[m], z=10 * np.log10(S_[m] + 1e-12), colorscale=[[i / 5, c_] for i, c_ in enumerate(T.SEQ)],
                                   colorbar=dict(title="dB", outlinewidth=0)))
        for a, b in zip(sub.debut_s, sub.fin_s):
            fig.add_vrect(x0=a, x1=b, line_color=T.CORAL_LIGHT, line_width=2, fillcolor="rgba(0,0,0,0)")
        fig.update_layout(xaxis_title="Temps (s)", yaxis_title="Fréquence (Hz)", title="Les crises (cadres corail) apparaissent comme des bandes de forte énergie à ~3 Hz")
        st.plotly_chart(fig_layout(fig, 400), use_container_width=True)
    with tabs[3]:
        q = pd.DataFrame({"Canal": names, "Écart-type brut": raw.std(0), "Écart-type filtré": filt.std(0),
                          "Pic-à-pic brut": np.ptp(raw, axis=0), "Écart-type normalisé": scaled.std(0)}).round(2)
        st.dataframe(q, use_container_width=True, hide_index=True)
        fig = px.bar(q, x="Canal", y="Écart-type filtré", color_discrete_sequence=[R["filtered"]], title="Amplitude par canal (repérer les canaux bruités)")
        st.plotly_chart(fig_layout(fig, 300), use_container_width=True)


# ============================================================ 2 · Fenêtre par fenêtre
def seg_figure(wi, n):
    """Schéma des fenêtres voisines de la fenêtre courante (paramètres actifs, pas de données simulées)."""
    fig = go.Figure()
    idx = list(range(max(0, wi - 2), min(n, wi + 4)))
    for row, j in enumerate(idx):
        a, b = j * STEP / FS, j * STEP / FS + WIN
        cur = j == wi
        fig.add_shape(type="rect", x0=a, x1=b, y0=row - 0.34, y1=row + 0.34, line=dict(color=R["window"] if cur else T.PETROL_500, width=2 if cur else 1),
                      fillcolor=T.COPPER_LIGHT if cur else T.PETROL_200)
        fig.add_annotation(x=(a + b) / 2, y=row, text=f"fenêtre {j}" + (" (courante)" if cur else ""), showarrow=False, font=dict(size=10, color=T.INK))
    fig.update_yaxes(visible=False, range=[len(idx) - 0.4, -0.7])
    x0, x1 = idx[0] * STEP / FS, idx[-1] * STEP / FS + WIN
    fig.update_xaxes(title_text="Temps (s)", range=[x0 - 0.3, x1 + 0.3])
    fig.update_layout(margin=dict(l=10, r=10, t=8, b=40))
    return fig_layout(fig, 215)


def page_window():
    with ui.panel("Paramètres du modèle"):
        c = st.columns([1.5, 1.5, 2, 1])
        K = c[0].slider("Nb de meilleures features", 5, 30, key="win_K")
        THR = c[1].slider("Seuil de décision", 0.05, 0.95, step=0.05, key="win_thr")
        mname = c[2].selectbox("Modèle", MODELS, key="win_model")
    feats = tuple(best_feats(K, 0.9))
    proba = loo(feats, mname)
    dd = ds.assign(proba=proba)
    d = dd[dd.rec_id == REC].reset_index(drop=True)
    n = len(d)
    raw, filt, scaled, names, fs = get_prep(REC)
    key = f"wi_{REC}_{WIN}_{OVL}"
    st.session_state.setdefault(key, 0)

    def jump(v):
        st.session_state[key] = int(min(max(v, 0), n - 1))

    ui.section("Navigation dans l'enregistrement")
    nav = st.columns([1, 1, 1.2, 2.2, 2])
    nav[0].button("◀ Précédente", on_click=lambda: jump(st.session_state[key] - 1))
    nav[1].button("Suivante ▶", on_click=lambda: jump(st.session_state[key] + 1))
    nav[2].button("Prochaine alerte ⏭", help="Prochaine fenêtre dont la probabilité dépasse le seuil",
                  on_click=lambda: jump(next((i for i in range(st.session_state[key] + 1, n) if d.proba[i] > THR), st.session_state[key])))
    if len(sub):
        pick_lbl = [f"Crise {i + 1} · {sub.debut_s[i]:.0f}s" for i in range(len(sub))]
        pick = pick_lbl.index(nav[4].selectbox("Aller à une crise", pick_lbl, label_visibility="collapsed"))
        nav[3].button("Aller à la crise choisie", on_click=lambda: jump(int(sub.debut_s[pick] / (STEP / FS)) - 1))
    wi = st.slider("Fenêtre n°", 0, n - 1, key=key)
    w = d.iloc[wi]
    verdict = ("TP" if w.Class == 1 and w.proba > THR else "FN" if w.Class == 1 else "FP" if w.proba > THR else "TN")
    vmap = {"TP": ("ok", "Crise détectée", "vrai positif"), "TN": ("neutral", "Pas d'alerte", "vrai négatif : pas de crise"),
            "FP": ("warn", "Fausse alarme", "faux positif : alerte sans crise annotée"), "FN": ("critical", "Crise manquée", "faux négatif : crise annotée sans alerte")}[verdict]

    ui.section("Fenêtre courante")
    top = st.columns([1.3, 1, 1, 1, 1.7])
    top[0].metric("Temps", f"{w.t_start:.1f} - {w.t_end:.1f} s")
    top[1].metric("Part en crise", f"{100 * w.frac_crise:.0f} %")
    top[2].metric("Classe réelle", "CRISE" if w.Class == 1 else "hors crise")
    top[3].metric("Proba. modèle", f"{w.proba:.2f}")
    with top[4]:
        ui.verdict(*vmap)

    with st.expander("Comment le signal est découpé en fenêtres", expanded=True):
        st.caption(f"Fenêtre de **{WIN:g} s** ({int(WIN * fs)} échantillons à {fs} Hz), décalée de **{STEP / fs:g} s** (recouvrement {int(OVL * 100)} %) : "
                   f"{n} fenêtres pour cet enregistrement. Une fenêtre est « crise » si au moins 50 % de ses échantillons sont dans une crise annotée.")
        st.plotly_chart(seg_figure(wi, n), use_container_width=True)

    ui.section("Signal prétraité")
    s0 = int(w.t_start * fs)
    ctx = int(6 * fs)
    a, b = max(0, s0 - ctx), min(len(scaled), s0 + int(WIN * fs) + ctx)
    cur_ = st.session_state.get("w_ch")
    if st.session_state.get("w_ch_names") != tuple(names):      # autre jeu de canaux : on réaffiche tous les canaux
        cur_, st.session_state["w_ch_names"] = None, tuple(names)
    st.session_state["w_ch"] = [c_ for c_ in (cur_ if cur_ is not None else names) if c_ in names] or names
    show = st.multiselect("Canaux affichés", names, key="w_ch")
    ui.legend([LEGEND_REF, ("Fenêtre courante", T.COPPER_LIGHT, "fill"), ("Signal EEG normalisé", R["signal"], "line")])
    scl = np.percentile(np.abs(scaled[a:b]), 99) * 1.2 + 1e-9
    fig = go.Figure()
    tx = np.arange(a, b) / fs
    for j, cn in enumerate(show):
        k = names.index(cn)
        fig.add_trace(go.Scattergl(x=tx, y=scaled[a:b, k] / scl - j * 1.6, mode="lines", line=dict(width=1, color=R["signal"]), name=cn, showlegend=False))
    fig.update_yaxes(tickmode="array", tickvals=[-j * 1.6 for j in range(len(show))], ticktext=show, gridcolor="rgba(0,0,0,0)")
    shade(fig, [(x, y) for x, y in zip(sub.debut_s, sub.fin_s) if y >= a / fs and x <= b / fs])
    fig.add_vrect(x0=w.t_start, x1=w.t_end, fillcolor="rgba(176,122,58,.30)", line_width=2, line_color=R["window"], layer="below")   # translucide : l'annotation reste visible dessous
    fig.update_xaxes(range=[a / fs, b / fs])          # les bandes de crises lointaines ne doivent pas étirer l'axe
    fig.update_layout(xaxis_title="Temps (s)", margin=dict(t=10))
    st.plotly_chart(fig_layout(fig, max(380, 28 * len(show) + 120)), use_container_width=True)

    ui.section("Probabilité de crise sur tout l'enregistrement")
    ui.legend([("Probabilité estimée par le modèle", R["proba"], "line"), LEGEND_REF, ("Seuil de décision", T.SLATE_LIGHT, "line"), ("Fenêtre courante (trait noir)", T.INK, "line")])
    fig = go.Figure(go.Scatter(x=d.t_start, y=d.proba, line=dict(color=R["proba"], width=1.6), name="Proba. de crise"))
    shade(fig, zip(sub.debut_s, sub.fin_s))
    fig.add_hline(y=THR, line_dash="dash", line_color=T.SLATE_LIGHT)
    fig.add_vline(x=w.t_start, line_color=T.INK)
    fig.update_layout(xaxis_title="Temps (s)", yaxis_title="P(crise)", margin=dict(t=10, b=40))
    st.plotly_chart(fig_layout(fig, 230), use_container_width=True)

    ui.section("Caractéristiques de la fenêtre")
    cA, cB = st.columns(2, gap="large")
    g0, g1 = ds[ds.Class == 0][list(feats)], ds[ds.Class == 1][list(feats)]
    z = (w[list(feats)].astype(float) - g0.mean()) / (g0.std() + 1e-12)
    zf = pd.DataFrame({"Feature": z.index, "z-score vs hors crise": z.values}).iloc[::-1]
    fig = px.bar(zf, x="z-score vs hors crise", y="Feature", orientation="h", color=zf["z-score vs hors crise"] > 2,
                 color_discrete_map={True: R["reference"], False: T.PETROL_200},
                 title="Écart à la normale de chaque feature (corail : au-delà de 2 écarts-types)")
    fig.add_vline(x=2, line_dash="dot", line_color=T.SLATE_LIGHT, annotation_text="2", annotation_position="top")
    fig.update_layout(showlegend=False)
    cA.plotly_chart(fig_layout(fig, 420), use_container_width=True)
    tbl = pd.DataFrame({"Fenêtre": w[list(feats)].astype(float), "Moy. hors crise": g0.mean(), "Moy. crise": g1.mean()}).round(3)
    cA.dataframe(tbl, use_container_width=True)
    cn = cB.selectbox("Canal du spectre", names, index=names.index("C3") if "C3" in names else 0)
    seg = scaled[s0:s0 + int(WIN * fs), names.index(cn)]
    fr, am = spectrum(seg, fs)
    mk = fr <= 40
    fig = go.Figure(go.Scatter(x=fr[mk], y=am[mk], fill="tozeroy", fillcolor="rgba(176,122,58,.18)", line=dict(color=R["proba"], width=1.6)))
    for bn, (x0, x1) in BANDS.items():
        fig.add_vrect(x0=x0, x1=x1, fillcolor=T.IVORY_DEEP, opacity=0.5, line_width=0, layer="below",
                      annotation_text=bn, annotation_position="top left", annotation_font=dict(size=10, color=T.SLATE))
    fig.update_layout(xaxis_title="Fréquence (Hz)", yaxis_title="|FFT|", title=f"Spectre de la fenêtre · {cn}")
    cB.plotly_chart(fig_layout(fig, 300), use_container_width=True)
    rel = ["Rel_Delta_mean", "Rel_Theta_mean", "Rel_Alpha_mean", "Rel_Beta_mean", "Rel_Hz3_mean"]
    bp = pd.DataFrame({"Bande": [r.split("_")[1] for r in rel] * 3,
                       "Valeur": list(w[rel].astype(float)) + list(ds[ds.Class == 0][rel].mean()) + list(ds[ds.Class == 1][rel].mean()),
                       "Groupe": ["Cette fenêtre"] * 5 + ["Moy. hors crise"] * 5 + ["Moy. crise"] * 5})
    fig = px.bar(bp, x="Bande", y="Valeur", color="Groupe", barmode="group", title="Puissance relative par bande (moyenne sur les canaux)",
                 color_discrete_map={"Cette fenêtre": R["window"], "Moy. hors crise": R["normal"], "Moy. crise": R["reference"]})
    cB.plotly_chart(fig_layout(fig, 300), use_container_width=True)


# ============================================================ 3 · Meilleures features
def page_features():
    rk = get_ranking(WIN, OVL, PRE)
    with ui.panel("Paramètres de sélection"):
        c = st.columns(2)
        K = c[0].slider("Nombre de features retenues", 3, 40, key="feat_K")
        thr = c[1].slider("Seuil de corrélation maximal entre features retenues", 0.5, 1.0, step=0.05, key="feat_corr")
    sel = best_feats(K, thr)
    st.session_state["best_feats"] = sel
    ui.section(f"{len(sel)} features retenues", "Bien classées et non redondantes (aucune paire retenue ne dépasse le seuil de corrélation).")
    ui.chips(sel)
    ui.section("Classement des features")
    show = rk.copy()
    show.insert(0, "Retenue", [f in sel for f in show.index])
    t1, t2 = st.tabs(["Tableau complet", "Scores comparés (20 premières)"])
    with t1:
        st.dataframe(show.round(4), use_container_width=True, height=420,
                     column_config={"Score": st.column_config.ProgressColumn("Score", min_value=0, max_value=1, format="%.2f")})
    with t2:
        top20 = show.head(20).iloc[::-1]
        fig = go.Figure()
        for flag, name, color in ((True, "Retenue", R["prediction"]), (False, "Non retenue (redondante ou moins bien classée)", T.PETROL_200)):
            part = top20[top20["Retenue"] == flag]
            fig.add_trace(go.Bar(x=part["Score"], y=part.index, orientation="h", name=name, marker_color=color))
        fig.update_yaxes(categoryorder="array", categoryarray=list(top20.index))      # même ordre que le tableau (meilleur en haut)
        fig.update_layout(xaxis_title="Score (1 = meilleur)", barmode="overlay")
        st.plotly_chart(fig_layout(fig, 520), use_container_width=True)
    st.caption("Score = rang moyen de 4 critères (information mutuelle, importance Random Forest, |d| de Cohen, F-ANOVA). "
               "Les features très corrélées entre elles (ex. Std/Variance/RMS/Energy) sont dédoublonnées.")
    ui.section("Comportement des features retenues")
    tabs = st.tabs(["Évolution dans le temps (patient)", "Distribution par classe", "Corrélations"])
    with tabs[0]:
        pick = st.multiselect("Features", sel, default=sel[:4])
        if pick:
            z = (d[pick] - ds[ds.Class == 0][pick].mean()) / (ds[ds.Class == 0][pick].std() + 1e-12)
            ui.legend([LEGEND_REF])
            fig = go.Figure()
            colors = T.SERIES
            for i, f in enumerate(pick):
                fig.add_trace(go.Scatter(x=d.t_start, y=z[f].clip(-5, 15), name=f, mode="lines", line=dict(width=1.5, color=colors[i % len(colors)], dash=DASHES[i % 4])))
            shade(fig, zip(sub.debut_s, sub.fin_s))
            fig.update_layout(xaxis_title="Temps (s)", yaxis_title="z-score (vs fenêtres hors crise)",
                              title=f"{REC} : les features montent pendant les crises")
            st.plotly_chart(fig_layout(fig, 420), use_container_width=True)
    with tabs[1]:
        f = st.selectbox("Feature", sel)
        dd = ds.assign(Classe=ds.Class.map({0: "Hors crise", 1: "Crise"}))
        lo, hi = dd[f].quantile([0.005, 0.995])
        fig = px.violin(dd[(dd[f] >= lo) & (dd[f] <= hi)], x="Classe", y=f, color="Classe", box=True, points=False,
                        color_discrete_map={"Hors crise": R["normal"], "Crise": R["reference"]})
        st.plotly_chart(fig_layout(fig, 430), use_container_width=True)
    with tabs[2]:
        fig = px.imshow(ds[sel].corr(), color_continuous_scale=T.DIV, zmin=-1, zmax=1, text_auto=".2f")
        st.plotly_chart(fig_layout(fig, 600), use_container_width=True)


# ============================================================ 4 · Classification
def page_classification():
    with ui.panel("Paramètres d'entraînement"):
        c = st.columns(4)
        fs_mode = c[0].radio("Features utilisées", ["Meilleures features", "Toutes les features"], key="cls_fs")
        K = c[1].slider("Nb de meilleures", 5, 30, key="cls_K")
        split = {"Par patient (réaliste)": "patient", "Stratifié 80/20 (sujet)": "stratifie"}[
            c[2].radio("Séparation 80/20", ["Par patient (réaliste)", "Stratifié 80/20 (sujet)"], key="cls_split")]
        bal = c[3].checkbox("Pondérer les classes", key="cls_bal")
        aug = c[3].checkbox("Data augmentation des crises", key="cls_aug",
                            help="Ajoute 2x plus de fenêtres de crise synthétiques (réelles + bruit gaussien) à l'entraînement uniquement.")
        learn_thr = c[3].checkbox("Seuil appris sur le train", key="cls_thr",
                                  help="Au lieu de 0,5, le seuil de décision de chaque modèle maximise le F1 sur 20 % du train (jamais sur le test).")
        feats = best_feats(K, 0.9) if fs_mode == "Meilleures features" else F_ALL
        st.caption(f"{len(feats)} features utilisées. Le split par patient évite la fuite entre fenêtres chevauchantes ; le stratifié est celui du sujet (optimiste). "
                   "Avec « Pondérer les classes », le KNN (k = 31) voit 5 non-crises pour 1 crise à l'entraînement, car il n'a pas de pondération.")
        run = st.button("Entraîner DT / RF / KNN / SVM", type="primary")
    if run:
        tr, te = split_data(ds, feats, mode=split)
        with st.spinner("Entraînement..."):
            res, cms, _, _ = evaluate_all(get_models(bal, rf_min_leaf=RF_LEAF, augment=aug), tr[feats], tr.Class, te[feats], te.Class,
                                          learn_thr=learn_thr, groups=tr.rec_id if split == "patient" else None)
        st.session_state["cls"] = (res, cms, len(tr), len(te), int(te.Class.sum()), res.attrs["seuils"], int(tr.Class.sum()), res.attrs["validation"])
    ui.section("Résultats sur l'ensemble de test")
    if "cls" in st.session_state:
        res, cms, ntr, nte, npos, thrs, npos_tr, val = st.session_state["cls"]
        m_ = st.columns(2)
        m_[0].metric("Train", f"{ntr:,} fenêtres".replace(",", " "), f"dont {npos_tr:,} en crise ({100 * npos_tr / ntr:.1f} %)".replace(",", " "), delta_color="off")
        m_[1].metric("Test", f"{nte:,} fenêtres".replace(",", " "), f"dont {npos:,} en crise ({100 * npos / nte:.1f} %)".replace(",", " "), delta_color="off")
        if val:
            st.caption(f"Seuil appris : le train est lui-même coupé en {val['n_fit']:,} fenêtres pour entraîner ({val['pos_fit']:,} en crise) et {val['n_val']:,} "
                       f"pour choisir le seuil ({val['pos_val']:,} en crise) ; le test n'est pas utilisé pour ce choix. ".replace(",", " ")
                       + "Seuils : " + ", ".join(f"{k} {v:.2f}" for k, v in thrs.items()))
        else:
            st.caption("Seuil de décision : 0,5 pour tous les modèles.")
        st.dataframe(res.style.format("{:.3f}").highlight_max(color=T.SAGE_LIGHT), use_container_width=True)
        cols = st.columns(4)
        for i, (nm, cm) in enumerate(cms.items()):
            fig = px.imshow(cm, text_auto=True, x=["Hors crise", "Crise"], y=["Hors crise", "Crise"], color_continuous_scale=T.SEQ[::-1], title=nm)
            fig.update_layout(xaxis_title="Prédit", yaxis_title="Réel", coloraxis_showscale=False, margin=dict(l=70, t=40))
            cols[i].plotly_chart(fig_layout(fig, 300), use_container_width=True)
        st.caption("Matrices de confusion : lignes = classe réelle, colonnes = classe prédite. Les cases hors diagonale sont les erreurs.")
    else:
        ui.callout("neutral", "Aucun entraînement lancé dans cette session. Réglez les paramètres ci-dessus puis lancez l'entraînement.")
    ui.callout("info", "**Métrique à privilégier : Recall** (une crise manquée est le pire cas), contrôlé par le **F1-score**. "
                       "On affiche la **Balanced Accuracy** (moyenne du Recall des deux classes) plutôt que l'Accuracy, trompeuse ici : "
                       "~96 % des fenêtres sont sans crise, donc « jamais de crise » ferait 96 % d'Accuracy mais seulement 50 % de Balanced Accuracy.",
               title="Comment lire les métriques")

    ref = {k: ROOT / "results" / f for k, f in [("sa", "summary_models.csv"), ("sb", "summary_models_augment.csv"), ("st", "stratified_80_20.csv")]}
    if all(f.exists() for f in ref.values()):
        ui.section("Protocole du sujet contre protocole réaliste", "Résultats de référence : F1 et Balanced Accuracy des 4 modèles, sans puis avec les deux améliorations.")
        a, b = (pd.read_csv(ref[k], index_col=0) for k in ("sa", "sb"))
        t = pd.read_csv(ref["st"])
        t = t.set_index(["Augmentation", "Modèle", "Seuil"])
        rows = {}
        for m in MODELS[:4]:
            rows[m] = {"F1 stratifié : de base": t.loc[("non", m, "seuil 0,5"), "F1 moy"], "F1 stratifié : augm. + seuil appris": t.loc[("oui", m, "seuil appris"), "F1 moy"],
                       "F1 par patient : de base": a.loc[m, "F1"], "F1 par patient : augm. + seuil appris": b.loc[m, "F1 (seuil appris)"],
                       "Bal. Acc. stratifié : de base": t.loc[("non", m, "seuil 0,5"), "Balanced Accuracy moy"],
                       "Bal. Acc. stratifié : augm. + seuil appris": t.loc[("oui", m, "seuil appris"), "Balanced Accuracy moy"],
                       "Bal. Acc. par patient : de base": a.loc[m, "Balanced Accuracy"], "Bal. Acc. par patient : augm. + seuil appris": b.loc[m, "Balanced Accuracy (seuil appris)"]}
        st.dataframe(pd.DataFrame(rows).T.style.format("{:.3f}"), use_container_width=True)
        st.caption("De base = seuil 0,5 sans data augmentation. Le stratifié 80/20 (5 découpages, moyenne) est celui du sujet : il est plus optimiste, car des fenêtres voisines quasi identiques "
                   "se retrouvent dans le train et le test. Le « par patient » (un enregistrement laissé de côté) est l'estimation réaliste. Avec augmentation et seuil appris, les quatre modèles se rapprochent.")

    ui.section("Résultats patient par patient", "Modèle jamais entraîné sur le patient évalué.")
    with ui.panel("Paramètres"):
        c = st.columns(2)
        m2 = c[0].selectbox("Modèle", MODELS, key="m_pp")
        th = c[1].slider("Seuil", 0.05, 0.95, step=0.05, key="th_pp")
        run_pp = st.button("Calculer par patient")
    if run_pp:
        st.session_state["pp"] = loo(tuple(feats), m2)
    if "pp" in st.session_state and len(st.session_state["pp"]) == len(ds):
        pr = st.session_state["pp"]
        rows = []
        for r in RECS:
            m = (ds.rec_id == r).to_numpy()
            y, p = ds.Class[m], (pr[m] > th).astype(int)
            mt = metrics(y, p)
            rows.append({"Patient": r, "Crises (fenêtres)": int(y.sum()), "TP": int(((y == 1) & (p == 1)).sum()), "FN": int(((y == 1) & (p == 0)).sum()),
                         "FP": int(((y == 0) & (p == 1)).sum()), "Recall": mt["Recall"], "Precision": mt["Precision"], "F1": mt["F1-score"]})
        st.dataframe(pd.DataFrame(rows).set_index("Patient").style.format({"Recall": "{:.2f}", "Precision": "{:.2f}", "F1": "{:.2f}"}),
                     use_container_width=True, height=500)
        st.caption("TP : crises détectées · FN : crises manquées · FP : fausses alarmes (fenêtres). Une fausse alarme peut être une crise non annotée dans l'Excel.")


# ============================================================ 5 · Détection temporelle
def page_detection():
    with ui.panel("Paramètres de détection"):
        c = st.columns([1, 1.5, 1, 1, 1, 1.25])
        K = c[0].slider("Nb features", 5, 30, key="det_K")
        mname = c[1].selectbox("Modèle", MODELS, key="det_model")
        th = c[2].slider("Seuil", 0.05, 0.95, step=0.05, key="det_th")
        smooth = c[3].slider("Lissage (fenêtres)", 1, 9, step=2, key="det_smooth")
        mind = c[4].slider("Durée min. (s)", 1.0, 6.0, step=0.5, key="det_mind")
        iou_min = c[5].slider("Recouvrement min. (IoU)", 0.1, 0.7, step=0.05, key="det_iou",
                              help="Un intervalle prédit compte pour une crise réelle si leur intersection / union dépasse cette valeur")
    feats = tuple(best_feats(K, 0.9))
    proba = loo(feats, mname)
    d = ds.assign(proba=proba)[ds.rec_id == REC].reset_index(drop=True)
    iv = predictions_to_intervals(d.t_start, d.t_end, (d.proba > th).astype(int), smooth=smooth, min_dur=mind)
    true_iv = sub[["debut_s", "fin_s", "duree_s"]]
    raw, filt, scaled, names, fs = get_prep(REC)

    ui.section("Bilan de la détection", "Appariement strict : chaque intervalle prédit ne compte que pour une seule crise réelle.")
    res = match_intervals(true_iv, iv, iou_min)
    m = st.columns(4)
    m[0].metric("Crises retrouvées", f"{res['tp']} / {len(true_iv)}")
    m[1].metric("Crises manquées", res["fn"])
    m[2].metric("Fausses alarmes", res["fp"])
    m[3].metric("Intervalles prédits", len(iv))

    ui.section("Signal, annotations de référence et prédictions")
    cn = st.selectbox("Canal affiché", names, index=names.index("C3") if "C3" in names else 0)
    sig = filt[:, names.index(cn)]
    ui.legend([("Annotation de référence (experts)", T.CORAL, "fill"), ("Prédiction du modèle (algorithme)", T.PETROL, "fill"), LEGEND_PRE,
               ("Probabilité de crise", R["proba"], "line"), ("Seuil", T.SLATE_LIGHT, "line")])
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, row_heights=[0.52, 0.18, 0.30], vertical_spacing=0.035)
    fig.add_trace(go.Scattergl(x=np.arange(0, len(sig), 4) / fs, y=sig[::4], line=dict(color=R["signal"], width=1), name=cn), row=1, col=1)
    pre_zones = [(max(0.0, a - PRE), a) for a in true_iv.debut_s]
    for r_ in (1, 2):
        shade(fig, pre_zones, T.AMBER_LIGHT, row=r_, col=1)
    shade(fig, zip(true_iv.debut_s, true_iv.fin_s), T.CORAL_LIGHT, row=1, col=1)
    for a, b in zip(true_iv.debut_s, true_iv.fin_s):          # couloir 1 : annotations de référence
        fig.add_shape(type="rect", x0=a, x1=b, y0=0.55, y1=1.0, fillcolor=T.CORAL, line_width=0, row=2, col=1)
    for a, b in zip(iv.debut_s, iv.fin_s):                    # couloir 2 : prédictions du modèle
        fig.add_shape(type="rect", x0=a, x1=b, y0=0.0, y1=0.45, fillcolor=T.PETROL, line_width=0, row=2, col=1)
        fig.add_shape(type="rect", x0=a, x1=b, y0=0.0, y1=1.0, fillcolor="rgba(20,80,92,.10)", line_width=0, row=3, col=1)
    fig.add_trace(go.Scatter(x=d.t_start, y=d.proba, line=dict(color=R["proba"], width=1.5), name="P(crise)"), row=3, col=1)
    fig.add_hline(y=th, line_dash="dash", line_color=T.SLATE_LIGHT, row=3, col=1)
    fig.update_yaxes(tickvals=[0.775, 0.225], ticktext=["Référence", "Modèle"], range=[-0.05, 1.05], showgrid=False, row=2, col=1)
    fig.update_yaxes(title_text="P(crise)", range=[0, 1.02], row=3, col=1)
    fig.update_yaxes(title_text="µV", row=1, col=1)
    fig.update_xaxes(title_text="Temps (s)", row=3, col=1)
    fig.update_layout(showlegend=False, margin=dict(t=10))
    st.plotly_chart(fig_layout(fig, 560), use_container_width=True)
    ui.callout("warn", "Les intervalles bleu pétrole sont des **prédictions algorithmiques**, pas un diagnostic. Les intervalles corail sont les annotations "
                       "de référence de l'Excel. Une « fausse alarme » peut être une vraie crise non annotée. La phase post-critique n'est pas annotée dans les données.",
               title="À lire avant d'interpréter")

    ui.section("Précision des bornes", "Erreur (prédit − réel, en secondes) pour les crises retrouvées.")
    if len(res["pairs"]):
        pr = res["pairs"].assign(**{"Crise n°": res["pairs"].vrai + 1}).rename(columns={"iou": "IoU", "err_debut": "Erreur début (s)", "err_fin": "Erreur fin (s)"})
        st.dataframe(pr[["Crise n°", "IoU", "Erreur début (s)", "Erreur fin (s)"]].round(2), hide_index=True, use_container_width=True)
    else:
        st.caption("Aucune crise retrouvée avec ces paramètres.")
    ui.section("Intervalles")
    cA, cB = st.columns(2, gap="large")
    cA.write("**Annotations réelles**")
    cA.dataframe(true_iv.round(1).reset_index(drop=True), use_container_width=True)
    cB.write("**Crises détectées**")
    cB.dataframe(iv.round(1), use_container_width=True)


# ============================================================ 6 · Prévention (préparation)
def page_prevention():
    ui.callout("warn", "Étude de faisabilité, pas un système de prévention. On teste si un modèle sait reconnaître les fenêtres des 30 s qui précèdent une crise, "
                       "par rapport aux fenêtres calmes, sur un enregistrement jamais vu. Résultat : **un signal faible existe, mais il est trop faible pour déclencher une alerte fiable**.",
               title="Étude de faisabilité")
    pf, lf = ROOT / "results" / "prevention_study.csv", ROOT / "results" / "prevention_lead.csv"
    if pf.exists() and lf.exists():
        ui.section("Peut-on reconnaître une pré-crise ?", "Pré-crise (30 s avant le début) contre fenêtres calmes (≥ 60 s avant et ≥ 15 s après une crise), un enregistrement laissé de côté.")
        pr_, ld_ = pd.read_csv(pf, index_col=0), pd.read_csv(lf, index_col=0)
        st.dataframe(pr_.style.format({"AUC": "{:.3f}", "PR-AUC": "{:.3f}", "PR-AUC aléatoire": "{:.3f}", "Balanced Accuracy": "{:.3f}", "AUC médiane / patient": "{:.3f}"}),
                     use_container_width=True)
        fig = px.bar(ld_.reset_index(), x="Délai avant la crise (s)", y="AUC", color_discrete_sequence=[T.PETROL_500], title="AUC selon le délai avant la crise")
        fig.add_hline(y=0.5, line_dash="dash", line_color=T.SLATE, annotation_text="hasard (0,5)")
        fig.update_yaxes(range=[0.4, 1])
        c_ = st.columns([3, 2])
        c_[0].plotly_chart(fig_layout(fig, 320), use_container_width=True)
        with c_[1]:
            ui.callout("info", "AUC 0,5 = hasard, 1 = parfait. Ici ~0,64 : mieux que le hasard dans 17 patients sur 19, mais très loin d'un détecteur utilisable "
                               "(Balanced Accuracy ~0,55). Le signal ne grandit pas à l'approche de la crise, ce qui suggère un état général plutôt qu'un vrai précurseur.",
                       title="Lecture")
        st.caption("Produit par `python -m tools.prevention_study`. Pistes pour aller plus loin : plus de patients et de crises, suivi de l'évolution des features sur plusieurs fenêtres "
                   "consécutives, modèles adaptés à chaque patient, et fréquence cardiaque (ECG) ou autres capteurs.")
    ui.section("Répartition des fenêtres par phase")
    c3 = ds.Class3.value_counts().rename({0: "Inter-critique", 1: "Pré-crise", 2: "Crise"})
    st.write(pd.DataFrame({"Fenêtres": c3}).T)
    ui.section("Évolution des features autour du début de crise")
    sel = st.session_state.get("best_feats") or best_feats(15, 0.9)
    with ui.panel("Paramètres"):
        c = st.columns([3, 2])
        pick = c[0].multiselect("Features à suivre", sel, default=sel[:4])
        before, after = c[1].slider("Fenêtre d'analyse autour du début de crise (s)", -90, 30, key="prev_range")
    rows = []
    for _, cz in annots.iterrows():
        dr = ds[ds.rec_id == cz.rec_id]
        rel = dr.t_start - cz.debut_s
        m = (rel >= before) & (rel <= after)
        if m.sum() == 0:
            continue
        base = dr[(rel >= before) & (rel < before + 20)][pick]
        z = (dr[m][pick] - base.mean()) / (base.std() + 1e-9)
        z["t_rel"] = rel[m].round().astype(int)
        rows.append(z)
    if rows and pick:
        al = pd.concat(rows).groupby("t_rel")[pick].mean().clip(upper=15)
        ui.legend([("Zone pré-critique", T.AMBER_LIGHT, "fill"), ("Début de crise (t = 0)", T.CORAL, "line")])
        fig = go.Figure()
        colors = T.SERIES
        for i, f in enumerate(pick):
            fig.add_trace(go.Scatter(x=al.index, y=al[f], name=f, mode="lines", line=dict(width=1.7, color=colors[i % len(colors)], dash=DASHES[i % 4])))
        fig.add_vrect(x0=-PRE, x1=0, fillcolor=T.AMBER_LIGHT, opacity=1, line_width=0, layer="below")
        fig.add_vline(x=0, line_color=T.CORAL, line_width=2)
        fig.update_layout(xaxis_title="Temps relatif au début de crise (s)", yaxis_title="z-score moyen (sur toutes les crises)",
                          title=f"Moyenne sur {len(rows)} crises, alignées sur leur début")
        st.plotly_chart(fig_layout(fig, 440), use_container_width=True)
        st.caption("Si une feature commence à monter avant t = 0, elle porte un signal précurseur exploitable pour la prévention.")


# ============================================================ routage
PAGES = [page_overview, page_preprocessing, page_window, page_features, page_classification, page_detection, page_prevention]
PAGES[step]()
ui.step_nav(step, LABELS, go_to)
ui.disclaimer()
