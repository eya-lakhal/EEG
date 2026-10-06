"""Détection de crises d'absence EEG - application interactive, patient par patient.

Lancer :  streamlit run app.py
(L'ancienne version organisée par questions du sujet est dans app_questions.py.)
"""
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots
from scipy.signal import spectrogram, welch

os.environ.setdefault("LOKY_MAX_CPU_COUNT", "4")
sys.path.insert(0, str(Path(__file__).parent))
from src.data_loader import CACHE, FS, list_recordings, load_annotations, load_recording
from src.dataset import build_dataset, feature_cols
from src.detection import compare_intervals, predictions_to_intervals
from src.features import BANDS, spectrum
from src.models import evaluate_all, get_models, loo_proba, metrics, split_data
from src.preprocessing import bandpass, robust_scale, select_channels
from src.selection import rank_features, select_uncorrelated

st.set_page_config(page_title="EEG - crises d'absence", layout="wide")
C0, C1, CP, CG = "#4C78A8", "#E45756", "#F2A93B", "#2CA02C"


# ============================================================ données (mises en cache)
@st.cache_data
def get_annots():
    return load_annotations()


@st.cache_data(show_spinner="Construction du dataset (fenêtrage + features)... ~1 min la première fois")
def get_dataset(win_s, overlap, pre_s):
    f = CACHE / f"dataset_all19_w{win_s}_o{int(overlap * 100)}_p{int(pre_s)}.csv"
    if f.exists():
        return pd.read_csv(f)
    df = build_dataset(win_s, overlap, pre_s)
    df.to_csv(f, index=False)
    return df


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
def get_loo(win_s, overlap, pre_s, feats, model):
    return loo_proba(get_dataset(win_s, overlap, pre_s), list(feats), model)


annots = get_annots()
RECS = list_recordings()
n_cr = annots.groupby("rec_id").size().to_dict()

# ============================================================ barre latérale
st.sidebar.title("EEG · crises d'absence")
PAGES = ["Vue d'ensemble", "1 · Prétraitement", "2 · Fenêtre par fenêtre", "3 · Meilleures features",
         "4 · Classification", "5 · Détection temporelle", "6 · Prévention (préparation)"]
page = st.sidebar.radio("Étape", PAGES)
st.sidebar.markdown("---")
label = {r: f"{r}  ·  {n_cr.get(r, 0)} crise(s)" for r in RECS}
REC = {v: k for k, v in label.items()}[st.sidebar.selectbox("Patient / enregistrement", list(label.values()))]
st.sidebar.markdown("**Segmentation**")
WIN = st.sidebar.select_slider("Fenêtre (s)", [1.0, 2.0, 4.0], value=2.0)
OVL = st.sidebar.select_slider("Recouvrement (%)", [0, 50, 75], value=50) / 100
PRE = st.sidebar.slider("Zone pré-crise (s)", 5, 30, 10)
ds = get_dataset(WIN, OVL, PRE)
F_ALL = [c for c in feature_cols(ds) if c != "Class3"]
STEP = int(WIN * (1 - OVL) * FS)
sub = annots[annots.rec_id == REC].reset_index(drop=True)
d = ds[ds.rec_id == REC].reset_index(drop=True)


def shade(fig, intervals, color=C1, op=0.35, **kw):
    for a, b in intervals:
        fig.add_vrect(x0=a, x1=b, fillcolor=color, opacity=op, line_width=0, **kw)


def best_feats(k, thr):
    return select_uncorrelated(get_ranking(WIN, OVL, PRE), ds, k, thr)


# ============================================================ Vue d'ensemble
if page == PAGES[0]:
    st.title("Détection automatique des crises d'absence - EEG")
    tab = []
    for r in RECS:
        x_, ch_, _ = load_recording(r)
        _, nm = select_channels(x_, ch_)
        dr, a = ds[ds.rec_id == r], annots[annots.rec_id == r]
        tab.append({"Patient": r, "Durée (s)": round(len(x_) / FS), "Canaux EEG": len(nm), "Crises": len(a),
                    "Durée moy. crise (s)": round(a.duree_s.mean(), 1) if len(a) else None,
                    "Fenêtres": len(dr), "Fenêtres crise": int(dr.Class.sum()),
                    "% crise": round(100 * dr.Class.mean(), 2)})
    tab = pd.DataFrame(tab)
    c = st.columns(4)
    c[0].metric("Patients", len(RECS))
    c[1].metric("Crises annotées", int(tab.Crises.sum()))
    c[2].metric("Fenêtres", f"{len(ds):,}")
    c[3].metric("% fenêtres en crise", f"{100 * ds.Class.mean():.1f} %")
    st.dataframe(tab, use_container_width=True, hide_index=True, height=420)
    st.plotly_chart(px.bar(tab, x="Patient", y="Crises", color_discrete_sequence=[C1], title="Nombre de crises par patient"),
                    use_container_width=True)
    st.markdown("""
**Chaîne de traitement** : signal brut (tous les canaux EEG du patient) → filtre 0,5-40 Hz → normalisation robuste par canal →
fenêtres → features par canal (agrégées moyenne/max sur les canaux) → sélection des meilleures features → classification →
reconstruction des crises dans le temps → *(prochaine étape)* prévention.
Utilisez la barre latérale pour choisir le patient, puis suivez les étapes dans l'ordre.""")

# ============================================================ 1 Prétraitement
elif page == PAGES[1]:
    st.header(f"Prétraitement · {REC}")
    c = st.columns(4)
    lo = c[0].number_input("Passe-haut (Hz)", 0.1, 5.0, 0.5, 0.1)
    hi = c[1].number_input("Passe-bas (Hz)", 20.0, 100.0, 40.0, 5.0)
    scale_on = c[2].toggle("Normalisation robuste (médiane/IQR)", True)
    raw, filt, scaled, names, fs = get_prep(REC, lo, hi)
    c[3].metric("Canaux EEG", len(names))
    st.caption("Le dataset du modèle est construit avec 0,5-40 Hz + normalisation. Ces réglages servent à explorer l'effet du prétraitement.")
    dur = len(raw) / fs
    cc = st.columns([2, 3, 2])
    chs = cc[0].multiselect("Canaux", names, default=[n for n in ["Fp1", "Cz", "O1"] if n in names] or names[:2])
    default_rng = (0.0, min(60.0, dur))
    if len(sub):
        default_rng = (max(0.0, float(sub.debut_s[0]) - 15), min(dur, float(sub.debut_s[0]) + 25))
    t_rng = cc[1].slider("Plage de temps (s)", 0.0, float(dur), default_rng, 1.0)
    i0, i1 = int(t_rng[0] * fs), int(t_rng[1] * fs)
    tt = np.arange(i0, i1) / fs
    tabs = st.tabs(["Signal : brut → filtré → normalisé", "Spectre avant / après", "Spectrogramme", "Qualité des canaux"])
    with tabs[0]:
        for cn in chs:
            k = names.index(cn)
            fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.03,
                                subplot_titles=(f"{cn} - brut", "filtré", "normalisé" if scale_on else "filtré (sans normalisation)"))
            fig.add_trace(go.Scattergl(x=tt, y=raw[i0:i1, k], line=dict(color="#888", width=1)), row=1, col=1)
            fig.add_trace(go.Scattergl(x=tt, y=filt[i0:i1, k], line=dict(color=C0, width=1)), row=2, col=1)
            fig.add_trace(go.Scattergl(x=tt, y=(scaled if scale_on else filt)[i0:i1, k], line=dict(color=CG, width=1)), row=3, col=1)
            for r_ in (1, 2, 3):
                shade(fig, zip(sub.debut_s, sub.fin_s), row=r_, col=1)
            fig.update_xaxes(title_text="Temps (s)", row=3, col=1)
            fig.update_layout(height=520, showlegend=False, margin=dict(t=40))
            st.plotly_chart(fig, use_container_width=True)
        st.caption("Zones rouges = crises annotées. Le filtre retire la dérive lente et le bruit secteur ; la normalisation met tous les patients à la même échelle.")
    with tabs[1]:
        cn = st.selectbox("Canal pour le spectre", names, index=names.index(chs[0]) if chs else 0)
        k = names.index(cn)
        f1, p1 = welch(raw[:, k], fs, nperseg=4 * fs)
        f2, p2 = welch(filt[:, k], fs, nperseg=4 * fs)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=f1, y=p1, name="brut", line=dict(color="#888")))
        fig.add_trace(go.Scatter(x=f2, y=p2, name="filtré", line=dict(color=C0)))
        for b, (a, z) in BANDS.items():
            fig.add_vrect(x0=a, x1=z, opacity=0.08, line_width=0, annotation_text=b, annotation_position="top left")
        fig.update_layout(xaxis_title="Fréquence (Hz)", yaxis_title="Densité spectrale", yaxis_type="log", xaxis_range=[0, 60], height=420)
        st.plotly_chart(fig, use_container_width=True)
    with tabs[2]:
        cn = st.selectbox("Canal", names, index=names.index(chs[0]) if chs else 0, key="spec_ch")
        f, t, S = spectrogram(filt[:, names.index(cn)], fs, nperseg=2 * fs, noverlap=fs)
        m = f <= 30
        fig = go.Figure(go.Heatmap(x=t, y=f[m], z=10 * np.log10(S[m] + 1e-12), colorscale="Viridis", colorbar=dict(title="dB")))
        shade(fig, zip(sub.debut_s, sub.fin_s), color="white", op=0.25)
        fig.update_layout(xaxis_title="Temps (s)", yaxis_title="Fréquence (Hz)", height=420, title="Les crises apparaissent comme des bandes de forte énergie à ~3 Hz")
        st.plotly_chart(fig, use_container_width=True)
    with tabs[3]:
        q = pd.DataFrame({"Canal": names, "Écart-type brut": raw.std(0), "Écart-type filtré": filt.std(0),
                          "Pic-à-pic brut": np.ptp(raw, axis=0), "Écart-type normalisé": scaled.std(0)}).round(2)
        st.dataframe(q, use_container_width=True, hide_index=True)
        st.plotly_chart(px.bar(q, x="Canal", y="Écart-type filtré", color_discrete_sequence=[C0], title="Amplitude par canal (repérer les canaux bruités)"),
                        use_container_width=True)

# ============================================================ 2 Fenêtre par fenêtre
elif page == PAGES[2]:
    st.header(f"Fenêtre par fenêtre · {REC}")
    c = st.columns([1, 1, 2, 2])
    K = c[0].slider("Nb de meilleures features", 5, 30, 12)
    THR = c[1].slider("Seuil de décision", 0.05, 0.95, 0.5, 0.05)
    mname = c[2].selectbox("Modèle", ["Random Forest", "SVM", "KNN", "Decision Tree"])
    feats = tuple(best_feats(K, 0.9))
    proba = get_loo(WIN, OVL, PRE, feats, mname)
    dd = ds.assign(proba=proba)
    d = dd[dd.rec_id == REC].reset_index(drop=True)
    n = len(d)
    raw, filt, scaled, names, fs = get_prep(REC)
    key = f"wi_{REC}_{WIN}_{OVL}"
    st.session_state.setdefault(key, 0)

    def go_to(v):
        st.session_state[key] = int(min(max(v, 0), n - 1))

    nav = st.columns([1, 1, 1, 3, 2])
    nav[0].button("◀ Précédente", on_click=lambda: go_to(st.session_state[key] - 1))
    nav[1].button("Suivante ▶", on_click=lambda: go_to(st.session_state[key] + 1))
    nav[2].button("Prochaine alerte ⏭", help="Prochaine fenêtre dont la probabilité dépasse le seuil",
                  on_click=lambda: go_to(next((i for i in range(st.session_state[key] + 1, n) if d.proba[i] > THR), st.session_state[key])))
    if len(sub):
        pick_lbl = [f"Crise {i + 1} · {sub.debut_s[i]:.0f}s" for i in range(len(sub))]
        pick = pick_lbl.index(nav[4].selectbox("Aller à une crise", pick_lbl, label_visibility="collapsed"))
        nav[3].button("Aller à la crise choisie", on_click=lambda: go_to(int(sub.debut_s[pick] / (STEP / FS)) - 1))
    wi = st.slider("Fenêtre n°", 0, n - 1, key=key)
    w = d.iloc[wi]
    verdict = ("TP" if w.Class == 1 and w.proba > THR else "FN" if w.Class == 1 else "FP" if w.proba > THR else "TN")
    msg = {"TP": ("success", "Crise détectée (vrai positif)"), "TN": ("info", "Pas de crise, pas d'alerte (vrai négatif)"),
           "FP": ("warning", "Fausse alarme (faux positif)"), "FN": ("error", "Crise manquée (faux négatif)")}[verdict]
    top = st.columns(5)
    top[0].metric("Temps", f"{w.t_start:.1f} - {w.t_end:.1f} s")
    top[1].metric("Part en crise", f"{100 * w.frac_crise:.0f} %")
    top[2].metric("Classe réelle", "CRISE" if w.Class == 1 else "hors crise")
    top[3].metric("Proba. modèle", f"{w.proba:.2f}")
    getattr(top[4], msg[0])(msg[1])

    # --- signal : tous les canaux, fenêtre surlignée avec contexte
    s0 = int(w.t_start * fs)
    ctx = int(6 * fs)
    a, b = max(0, s0 - ctx), min(len(scaled), s0 + int(WIN * fs) + ctx)
    show = st.multiselect("Canaux affichés", names, default=names, key="w_ch")
    scl = np.percentile(np.abs(scaled[a:b]), 99) * 1.2 + 1e-9
    fig = go.Figure()
    tx = np.arange(a, b) / fs
    for j, cn in enumerate(show):
        k = names.index(cn)
        fig.add_trace(go.Scattergl(x=tx, y=scaled[a:b, k] / scl - j * 1.6, mode="lines", line=dict(width=1, color="#333"), name=cn, showlegend=False))
    fig.update_yaxes(tickmode="array", tickvals=[-j * 1.6 for j in range(len(show))], ticktext=show)
    shade(fig, zip(sub.debut_s, sub.fin_s), C1, 0.25)
    fig.add_vrect(x0=w.t_start, x1=w.t_end, fillcolor=CP, opacity=0.25, line_width=2, line_color=CP)
    fig.update_layout(height=max(380, 28 * len(show) + 120), xaxis_title="Temps (s)", margin=dict(t=30),
                      title="Signal prétraité (jaune = fenêtre courante, rouge = crise annotée)")
    st.plotly_chart(fig, use_container_width=True)

    # --- timeline proba
    fig = go.Figure(go.Scatter(x=d.t_start, y=d.proba, line=dict(color=CP), name="Proba. de crise"))
    shade(fig, zip(sub.debut_s, sub.fin_s), C1, 0.35)
    fig.add_hline(y=THR, line_dash="dash", line_color="#888")
    fig.add_vline(x=w.t_start, line_color="black")
    fig.update_layout(height=220, xaxis_title="Temps (s)", yaxis_title="P(crise)", margin=dict(t=10, b=0))
    st.plotly_chart(fig, use_container_width=True)

    # --- features / spectre
    cA, cB = st.columns(2)
    g0, g1 = ds[ds.Class == 0][list(feats)], ds[ds.Class == 1][list(feats)]
    z = (w[list(feats)].astype(float) - g0.mean()) / (g0.std() + 1e-12)
    zf = pd.DataFrame({"Feature": z.index, "z-score vs hors crise": z.values}).iloc[::-1]
    fig = px.bar(zf, x="z-score vs hors crise", y="Feature", orientation="h", height=420,
                 color=zf["z-score vs hors crise"] > 2, color_discrete_map={True: C1, False: C0},
                 title="Meilleures features de cette fenêtre (écart à la normale)")
    fig.update_layout(showlegend=False)
    cA.plotly_chart(fig, use_container_width=True)
    tbl = pd.DataFrame({"Fenêtre": w[list(feats)].astype(float), "Moy. hors crise": g0.mean(), "Moy. crise": g1.mean()}).round(3)
    cA.dataframe(tbl, use_container_width=True)
    cn = cB.selectbox("Canal du spectre", names, index=names.index("C3") if "C3" in names else 0)
    seg = scaled[s0:s0 + int(WIN * fs), names.index(cn)]
    fr, am = spectrum(seg, fs)
    mk = fr <= 40
    fig = go.Figure(go.Scatter(x=fr[mk], y=am[mk], fill="tozeroy", line=dict(color=CP)))
    for bn, (x0, x1) in BANDS.items():
        fig.add_vrect(x0=x0, x1=x1, opacity=0.07, line_width=0, annotation_text=bn, annotation_position="top left")
    fig.update_layout(height=300, xaxis_title="Fréquence (Hz)", yaxis_title="|FFT|", title=f"Spectre de la fenêtre - {cn}")
    cB.plotly_chart(fig, use_container_width=True)
    rel = ["Rel_Delta_mean", "Rel_Theta_mean", "Rel_Alpha_mean", "Rel_Beta_mean", "Rel_Hz3_mean"]
    bp = pd.DataFrame({"Bande": [r.split("_")[1] for r in rel] * 3,
                       "Valeur": list(w[rel].astype(float)) + list(ds[ds.Class == 0][rel].mean()) + list(ds[ds.Class == 1][rel].mean()),
                       "Groupe": ["Cette fenêtre"] * 5 + ["Moy. hors crise"] * 5 + ["Moy. crise"] * 5})
    cB.plotly_chart(px.bar(bp, x="Bande", y="Valeur", color="Groupe", barmode="group", height=300,
                           title="Puissance relative par bande (moyenne sur les canaux)"), use_container_width=True)

# ============================================================ 3 Meilleures features
elif page == PAGES[3]:
    st.header("Meilleures features")
    rk = get_ranking(WIN, OVL, PRE)
    c = st.columns(2)
    K = c[0].slider("Nombre de features retenues", 3, 40, 15)
    thr = c[1].slider("Seuil de corrélation maximal entre features retenues", 0.5, 1.0, 0.9, 0.05)
    sel = best_feats(K, thr)
    st.session_state["best_feats"] = sel
    st.success(f"**{len(sel)} features retenues** (bien classées, non redondantes) : " + ", ".join(sel))
    show = rk.copy()
    show.insert(0, "Retenue", [f in sel for f in show.index])
    st.dataframe(show.round(4), use_container_width=True, height=420,
                 column_config={"Score": st.column_config.ProgressColumn("Score", min_value=0, max_value=1, format="%.2f")})
    st.caption("Score = rang moyen de 4 critères (information mutuelle, importance Random Forest, |d| de Cohen, F-ANOVA). "
               "Les features très corrélées entre elles (ex. Std/Variance/RMS/Energy) sont dédoublonnées.")
    tabs = st.tabs(["Évolution dans le temps (patient)", "Distribution par classe", "Corrélations"])
    with tabs[0]:
        pick = st.multiselect("Features", sel, default=sel[:4])
        if pick:
            z = (d[pick] - ds[ds.Class == 0][pick].mean()) / (ds[ds.Class == 0][pick].std() + 1e-12)
            fig = go.Figure()
            for f in pick:
                fig.add_trace(go.Scatter(x=d.t_start, y=z[f].clip(-5, 15), name=f, mode="lines", line=dict(width=1.3)))
            shade(fig, zip(sub.debut_s, sub.fin_s))
            fig.update_layout(height=420, xaxis_title="Temps (s)", yaxis_title="z-score (vs fenêtres hors crise)",
                              title=f"{REC} - les features montent pendant les crises (zones rouges)")
            st.plotly_chart(fig, use_container_width=True)
    with tabs[1]:
        f = st.selectbox("Feature", sel)
        dd = ds.assign(Classe=ds.Class.map({0: "Hors crise", 1: "Crise"}))
        lo, hi = dd[f].quantile([0.005, 0.995])
        st.plotly_chart(px.violin(dd[(dd[f] >= lo) & (dd[f] <= hi)], x="Classe", y=f, color="Classe", box=True, points=False,
                                  color_discrete_map={"Hors crise": C0, "Crise": C1}), use_container_width=True)
    with tabs[2]:
        st.plotly_chart(px.imshow(ds[sel].corr(), color_continuous_scale="RdBu_r", zmin=-1, zmax=1, text_auto=".2f", height=600),
                        use_container_width=True)

# ============================================================ 4 Classification
elif page == PAGES[4]:
    st.header("Classification crise / non crise")
    c = st.columns(4)
    fs_mode = c[0].radio("Features utilisées", ["Meilleures features", "Toutes les features"])
    K = c[1].slider("Nb de meilleures", 5, 30, 15)
    split = {"Par patient (réaliste)": "patient", "Stratifié 80/20 (sujet)": "stratifie"}[c[2].radio("Séparation 80/20", ["Par patient (réaliste)", "Stratifié 80/20 (sujet)"])]
    bal = c[3].checkbox("Pondérer les classes", True)
    feats = best_feats(K, 0.9) if fs_mode == "Meilleures features" else F_ALL
    st.caption(f"{len(feats)} features utilisées. Le split par patient évite la fuite entre fenêtres chevauchantes ; le stratifié est celui du sujet (optimiste).")
    if st.button("Entraîner DT / RF / KNN / SVM", type="primary"):
        tr, te = split_data(ds, feats, mode=split)
        with st.spinner("Entraînement..."):
            res, cms, _, _ = evaluate_all(get_models(bal), tr[feats], tr.Class, te[feats], te.Class)
        st.session_state["cls"] = (res, cms, len(tr), len(te), int(te.Class.sum()))
    if "cls" in st.session_state:
        res, cms, ntr, nte, npos = st.session_state["cls"]
        st.write(f"Train : {ntr:,} fenêtres - Test : {nte:,} fenêtres ({npos} crises)")
        st.dataframe(res.style.format("{:.3f}").highlight_max(color="#cfe8cf"), use_container_width=True)
        cols = st.columns(4)
        for i, (nm, cm) in enumerate(cms.items()):
            fig = px.imshow(cm, text_auto=True, x=["Hors crise", "Crise"], y=["Hors crise", "Crise"], color_continuous_scale="Blues", title=nm)
            fig.update_layout(xaxis_title="Prédit", yaxis_title="Réel", height=320, coloraxis_showscale=False)
            cols[i].plotly_chart(fig, use_container_width=True)
    st.markdown("**Métrique à privilégier : Recall (une crise manquée est le pire cas) contrôlé par le F1-score.** "
                "L'Accuracy est trompeuse : ~96 % des fenêtres sont sans crise, donc « jamais de crise » ferait déjà 96 %.")
    st.subheader("Résultats patient par patient (modèle jamais entraîné sur le patient)")
    c = st.columns(2)
    m2 = c[0].selectbox("Modèle", ["Random Forest", "SVM", "KNN", "Decision Tree"], key="m_pp")
    th = c[1].slider("Seuil", 0.05, 0.95, 0.5, 0.05, key="th_pp")
    if st.button("Calculer par patient"):
        st.session_state["pp"] = get_loo(WIN, OVL, PRE, tuple(feats), m2)
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

# ============================================================ 5 Détection temporelle
elif page == PAGES[5]:
    st.header(f"Détection temporelle · {REC}")
    c = st.columns(5)
    K = c[0].slider("Nb features", 5, 30, 15)
    mname = c[1].selectbox("Modèle", ["Random Forest", "SVM", "KNN", "Decision Tree"])
    th = c[2].slider("Seuil", 0.05, 0.95, 0.5, 0.05)
    smooth = c[3].slider("Lissage (fenêtres)", 1, 9, 3, 2)
    mind = c[4].slider("Durée min. (s)", 1.0, 6.0, 2.0, 0.5)
    feats = tuple(best_feats(K, 0.9))
    proba = get_loo(WIN, OVL, PRE, feats, mname)
    d = ds.assign(proba=proba)[ds.rec_id == REC].reset_index(drop=True)
    iv = predictions_to_intervals(d.t_start, d.t_end, (d.proba > th).astype(int), smooth=smooth, min_dur=mind)
    true_iv = sub[["debut_s", "fin_s", "duree_s"]]
    raw, filt, scaled, names, fs = get_prep(REC)
    cn = st.selectbox("Canal affiché", names, index=names.index("C3") if "C3" in names else 0)
    sig = filt[:, names.index(cn)]
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.7, 0.3], vertical_spacing=0.04)
    fig.add_trace(go.Scattergl(x=np.arange(0, len(sig), 4) / fs, y=sig[::4], line=dict(color="#555", width=1), name=cn), row=1, col=1)
    shade(fig, zip(true_iv.debut_s, true_iv.fin_s), C1, 0.4, row=1, col=1)
    fig.add_trace(go.Scatter(x=d.t_start, y=d.proba, line=dict(color=CP), name="P(crise)"), row=2, col=1)
    shade(fig, zip(iv.debut_s, iv.fin_s), CG, 0.35, row=2, col=1)
    fig.add_hline(y=th, line_dash="dash", line_color="#888", row=2, col=1)
    fig.update_layout(height=520, title="Rouge = crises réelles · Vert = crises détectées", xaxis2_title="Temps (s)")
    st.plotly_chart(fig, use_container_width=True)
    hit, nt, fa = compare_intervals(true_iv, iv)
    m = st.columns(3)
    m[0].metric("Crises détectées", f"{hit} / {nt}")
    m[1].metric("Fausses alarmes", fa)
    m[2].metric("Intervalles prédits", len(iv))
    cA, cB = st.columns(2)
    cA.write("**Annotations réelles**")
    cA.dataframe(true_iv.round(1).reset_index(drop=True), use_container_width=True)
    cB.write("**Crises détectées**")
    cB.dataframe(iv.round(1), use_container_width=True)

# ============================================================ 6 Prévention
else:
    st.header("Prévention - préparation")
    st.info("Étape à venir : prédire la crise **avant** son début. Les bases sont déjà prêtes : `Class3` (inter-critique / pré-crise / crise) "
            "dans le dataset, et la zone pré-crise réglable dans la barre latérale. Ci-dessous : les features changent-elles AVANT le début de la crise ?")
    c3 = ds.Class3.value_counts().rename({0: "Inter-critique", 1: "Pré-crise", 2: "Crise"})
    st.write(pd.DataFrame({"Fenêtres": c3}).T)
    sel = st.session_state.get("best_feats") or best_feats(15, 0.9)
    pick = st.multiselect("Features à suivre", sel, default=sel[:4])
    before, after = st.slider("Fenêtre d'analyse autour du début de crise (s)", -90, 30, (-60, 20))
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
        fig = go.Figure()
        for f in pick:
            fig.add_trace(go.Scatter(x=al.index, y=al[f], name=f, mode="lines"))
        fig.add_vline(x=0, line_color=C1, annotation_text="début de crise")
        fig.add_vrect(x0=-PRE, x1=0, fillcolor=CP, opacity=0.15, line_width=0, annotation_text="pré-crise")
        fig.update_layout(height=450, xaxis_title="Temps relatif au début de crise (s)", yaxis_title="z-score moyen (sur toutes les crises)",
                          title=f"Moyenne sur {len(rows)} crises, alignées sur leur début")
        st.plotly_chart(fig, use_container_width=True)
        st.caption("Si une feature commence à monter avant t = 0, elle porte un signal précurseur exploitable pour la prévention.")
