"""Détection automatique des crises d'absence à partir de signaux EEG - application Streamlit.

Lancer :  streamlit run app.py
"""
import os
import sys
from pathlib import Path

os.environ.setdefault("LOKY_MAX_CPU_COUNT", "4")  # évite un avertissement joblib sous Windows

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots
from sklearn.feature_selection import mutual_info_classif
from sklearn.metrics import confusion_matrix

sys.path.insert(0, str(Path(__file__).parent))
from src.data_loader import CACHE, FS, list_recordings, load_annotations, load_header, load_recording
from src.dataset import META_COLS, build_dataset, feature_cols
from src.detection import compare_intervals, predictions_to_intervals
from src.features import BANDS, spectrum
from src.models import evaluate_all, get_models, metrics, split_data
from src.preprocessing import COMMON_EEG, bandpass, make_windows, robust_scale, sample_labels, select_channels

st.set_page_config(page_title="Détection de crises d'absence (EEG)", layout="wide")
C0, C1, CP = "#4C78A8", "#E45756", "#F2A93B"  # hors crise / crise / pré-crise


# ----------------------------------------------------------------------------- données
@st.cache_data(show_spinner="Lecture des annotations...")
def get_annots():
    a = load_annotations()
    return a, a.attrs.get("dropped", [])


@st.cache_data(show_spinner="Chargement de l'enregistrement...")
def get_rec(rec):
    x, ch, fs = load_recording(rec)
    return x, ch, fs


@st.cache_data(show_spinner="Construction du dataset (fenêtrage + features), ~30 s la première fois...")
def get_dataset(win_s, overlap, pre_s):
    f = CACHE / f"dataset_all19_w{win_s}_o{int(overlap * 100)}_p{int(pre_s)}.csv"
    if f.exists():
        return pd.read_csv(f)
    df = build_dataset(win_s, overlap, pre_s)
    df.to_csv(f, index=False)
    return df


annots, dropped = get_annots()
RECS = list_recordings()

st.sidebar.title("EEG - crises d'absence")
PAGES = ["0. Accueil & pipeline", "1. Exploration des données", "2. Annotations des crises",
         "3. Variable cible", "4. Segmentation & features", "5. Analyse exploratoire",
         "6. Modèles & évaluation", "7. Détection temporelle", "8. Erreurs & améliorations"]
page = st.sidebar.radio("Étape", PAGES)
st.sidebar.markdown("---")
st.sidebar.subheader("Paramètres de segmentation")
WIN = st.sidebar.select_slider("Taille de fenêtre (s)", [1.0, 2.0, 4.0], value=2.0)
OVL = st.sidebar.select_slider("Recouvrement (%)", [0, 50, 75], value=50) / 100
PRE = st.sidebar.slider("Durée pré-crise (s)", 5, 30, 10)


def feats_of(df):
    return [c for c in feature_cols(df) if c != "Class3"]


# ----------------------------------------------------------------------------- 0
if page == PAGES[0]:
    st.title("Détection automatique des crises d'absence à partir de signaux EEG")
    st.markdown(f"""
**Objectif** : chaîne Data Mining complète *signal EEG brut → fenêtres → features → classification → intervalles de crise*.

| | |
|---|---|
| Enregistrements | **{len(RECS)}** fichiers `.mat` (timetable MATLAB, {FS} Hz) |
| Crises annotées | **{len(annots)}** crises valides dans **{annots.rec_id.nunique()}** enregistrements |
| Canaux utilisés | **tous les canaux EEG** de chaque patient : 19 (système 10-20) pour 21 fichiers, 8 pour 200625A_F. ECG/EMG/SLI exclus (non EEG) |

**Choix de conception (justifiés)**
1. *Annotations* : `Annotations MAJ.xlsx` donne des temps en **fraction de jour** → temps relatif = (DebCE − DebutTrace) × 86 400 s.
2. *Prétraitement* : passe-bande 0,5-40 Hz (Butterworth, phase nulle) puis normalisation robuste (médiane/IQR) **par enregistrement et par canal** pour absorber les écarts d'amplitude entre patients.
3. *Fenêtres* : 2 s, recouvrement 50 %, classe 1 si ≥ 50 % d'échantillons en crise.
4. *Features* : 14 temporelles + puissances de bandes (Delta/Theta/Alpha/Beta **+ bande 3 Hz**, signature des pointes-ondes d'absence), puissances relatives, entropie spectrale, fréquence dominante, SEF95. Calculées sur **chaque canal** puis agrégées (moyenne et max sur tous les canaux) → indépendant du nombre de canaux disponibles.
5. *Évaluation* : split stratifié 80/20 (demandé) **et** split par enregistrement (sans fuite).
6. *Déséquilibre* : `class_weight="balanced"`, métrique principale **Recall / F1**, pas l'Accuracy (on affiche la **Balanced Accuracy**).
7. *Pré-crise* : la classe `Class3` (0 inter-critique, 1 pré-crise, 2 crise) est disponible pour une classification à trois états.
""")
    st.graphviz_chart("""digraph { rankdir=LR; node [shape=box, style=rounded];
    "Signal EEG" -> "Exploration" -> "Prétraitement" -> "Segmentation" -> "Features" -> "Dataset" -> "ML (DT/RF/KNN/SVM)" -> "Évaluation" -> "Détection temporelle" -> "Début/Fin des crises" }""")
    if dropped:
        st.warning(f"Crise(s) écartée(s) car incohérente(s) (fin erronée dans l'Excel) : {dropped}")
    no_ann = [r for r in RECS if r not in set(annots.rec_id)]
    st.info(f"Enregistrements sans crise annotée (utilisés comme exemples négatifs) : {no_ann}")

# ----------------------------------------------------------------------------- 1
elif page == PAGES[1]:
    st.header("Partie I - Compréhension et exploration des données")
    rec = st.selectbox("Enregistrement", RECS, index=0)
    x, ch, fs = get_rec(rec)
    st.subheader("Question 1 - Informations du fichier")
    miss = int(np.isnan(x).sum())
    c = st.columns(5)
    c[0].metric("Lignes (échantillons)", f"{x.shape[0]:,}")
    c[1].metric("Variables (canaux)", x.shape[1])
    c[2].metric("Fréquence d'échantillonnage", f"{fs} Hz")
    c[3].metric("Durée totale", f"{x.shape[0] / fs:.0f} s ({x.shape[0] / fs / 60:.1f} min)")
    c[4].metric("Valeurs manquantes", miss)
    df_head = pd.DataFrame(x[:10], columns=ch)
    df_head.insert(0, "Time (s)", np.arange(10) / fs)
    st.dataframe(df_head, use_container_width=True)
    st.caption("Le fichier brut est une timetable (1 ligne/seconde contenant 256 échantillons par canal) ; elle est aplatie en "
               "signal continu (1 ligne = 1 échantillon).")
    hdr = load_header(rec)
    with st.expander("Annotations textuelles du fichier *_h.mat (marqueurs de l'enregistreur)"):
        st.dataframe(hdr, use_container_width=True, height=250)

    st.subheader("Question 2 - Signal EEG en fonction du temps")
    chs = st.multiselect("Canaux", ch, default=[c for c in ["Fp1", "Cz", "O1", "C3"] if c in ch][:3] or ch[:2])
    filt = st.checkbox("Signal filtré 0,5-40 Hz", value=True)
    tmax = x.shape[0] / fs
    t_rng = st.slider("Plage de temps (s)", 0.0, float(tmax), (0.0, float(tmax)), step=1.0)
    i0, i1 = int(t_rng[0] * fs), int(t_rng[1] * fs)
    step = max(1, (i1 - i0) // 40000)
    fig = make_subplots(rows=len(chs), cols=1, shared_xaxes=True, vertical_spacing=0.02)
    sub = annots[annots.rec_id == rec]
    for k, cn in enumerate(chs, 1):
        s = x[:, ch.index(cn)].astype(float)
        if filt:
            s = bandpass(s[:, None], fs)[:, 0]
        fig.add_trace(go.Scattergl(x=np.arange(i0, i1, step) / fs, y=s[i0:i1:step], mode="lines", name=cn,
                                   line=dict(width=1)), row=k, col=1)
        fig.update_yaxes(title_text=f"{cn} (µV)", row=k, col=1)
        for _, r in sub.iterrows():
            fig.add_vrect(x0=r.debut_s, x1=r.fin_s, fillcolor=C1, opacity=0.35, line_width=0, row=k, col=1)
    fig.update_xaxes(title_text="Temps (s)", row=len(chs), col=1)
    fig.update_layout(height=220 * len(chs) + 80, title=f"EEG {rec} - zones rouges = crises annotées", showlegend=False)
    st.plotly_chart(fig, use_container_width=True)
    st.markdown("**Observation** : pendant les crises d'absence, l'amplitude augmente nettement et le signal devient très "
                "rythmique : décharges de **pointes-ondes généralisées à ~3 Hz**, visibles simultanément sur tous les canaux, "
                "alors que le fond est désorganisé hors crise. Les artefacts (mouvements, clignements) produisent aussi de "
                "grandes amplitudes mais sont non rythmiques et plus localisés - d'où le besoin de features spectrales.")

# ----------------------------------------------------------------------------- 2
elif page == PAGES[2]:
    st.header("Partie II - Exploitation des annotations")
    st.markdown("Temps Excel = fraction de jour. Début relatif = (DebCE − DebutTrace) × 86 400 s. Doublons "
                "(230406B-A, 230515B-G) : dernière ligne conservée ; crise invalide écartée.")
    st.subheader("Question 3 - Début, fin et nombre de crises")
    rec = st.selectbox("Enregistrement", ["(tous)"] + sorted(annots.rec_id.unique()))
    t = annots if rec == "(tous)" else annots[annots.rec_id == rec]
    show = t.rename(columns={"crise": "Crise", "debut_s": "Début (s)", "fin_s": "Fin (s)", "duree_s": "Durée (s)"})
    st.dataframe(show.round(2), use_container_width=True)
    st.metric("Nombre total de crises", len(t))
    st.subheader("Question 4 - Statistiques")
    d = t.duree_s
    inter = []
    for _, g in t.groupby("rec_id"):
        g = g.sort_values("debut_s")
        inter += list(g.debut_s.values[1:] - g.fin_s.values[:-1])
    c = st.columns(4)
    c[0].metric("Durée moyenne", f"{d.mean():.2f} s")
    c[1].metric("Durée minimale", f"{d.min():.2f} s")
    c[2].metric("Durée maximale", f"{d.max():.2f} s")
    c[3].metric("Intervalle moyen entre crises", f"{np.mean(inter):.1f} s" if inter else "n/a")
    fig = px.histogram(t, x="duree_s", nbins=25, labels={"duree_s": "Durée (s)"}, title="Distribution des durées de crise",
                       color_discrete_sequence=[C1])
    st.plotly_chart(fig, use_container_width=True)
    cv = d.std() / d.mean() if len(d) > 1 else 0
    st.markdown(f"**Les crises ont-elles une durée similaire ?** Non : coefficient de variation = **{cv:.0%}** (écart-type "
                f"{d.std():.1f} s pour une moyenne de {d.mean():.1f} s). Les durées vont de {d.min():.0f} s à {d.max():.0f} s : "
                "des crises très brèves (2-3 s) coexistent avec des crises longues. Conséquence : une fenêtre de 2 s est adaptée "
                "aux crises courtes et les crises < 2 s seraient mal capturées.")

# ----------------------------------------------------------------------------- 3
elif page == PAGES[3]:
    st.header("Partie III - Construction de la variable cible")
    rec = st.selectbox("Enregistrement", sorted(annots.rec_id.unique()), index=0)
    x, ch, fs = get_rec(rec)
    sub = annots[annots.rec_id == rec]
    y = sample_labels(len(x), fs, list(zip(sub.debut_s, sub.fin_s)))
    st.subheader("Question 5 - Colonne Class (0 = hors crise, 1 = crise)")
    cn = st.selectbox("Canal affiché", ch, index=ch.index("Cz") if "Cz" in ch else 0)
    s = pd.DataFrame({"Time": np.arange(len(x)) / fs, cn: x[:, ch.index(cn)], "Class": y})
    st.dataframe(s.iloc[int(sub.debut_s.iloc[0] * fs) - 5: int(sub.debut_s.iloc[0] * fs) + 5].reset_index(drop=True))
    st.subheader("Question 6 - Signal avec crises distinguées")
    step = 4
    sig = bandpass(x[:, [ch.index(cn)]].astype(float), fs)[:, 0]
    tt = np.arange(len(x)) / fs
    ys = np.where(y == 1, np.nan, sig)
    yc = np.where(y == 0, np.nan, sig)
    fig = go.Figure()
    fig.add_trace(go.Scattergl(x=tt[::step], y=ys[::step], mode="lines", name="Hors crise", line=dict(color=C0, width=1)))
    fig.add_trace(go.Scattergl(x=tt[::step], y=yc[::step], mode="lines", name="Pendant crise", line=dict(color=C1, width=1.5)))
    fig.update_layout(xaxis_title="Temps (s)", yaxis_title="Amplitude (µV)", title=f"{rec} - canal {cn}", height=400)
    st.plotly_chart(fig, use_container_width=True)
    st.subheader("Déséquilibre des classes")
    ds = get_dataset(WIN, OVL, PRE)
    n_s = pd.Series(y).value_counts(normalize=True).mul(100)
    cw = ds.Class.value_counts(normalize=True).mul(100)
    c = st.columns(2)
    c[0].write("**Échantillons (cet enregistrement)**")
    c[0].dataframe(pd.DataFrame({"%": n_s.round(2)}).rename(index={0: "0 - hors crise", 1: "1 - crise"}))
    c[1].write("**Fenêtres (tous enregistrements)**")
    c[1].dataframe(pd.DataFrame({"%": cw.round(2), "Nombre": ds.Class.value_counts()}).rename(index={0: "0 - hors crise", 1: "1 - crise"}))
    st.markdown(f"**Le déséquilibre est-il important ?** Oui : seulement **{cw.get(1, 0):.1f} %** des fenêtres sont des crises "
                f"(ratio ≈ 1:{cw.get(0, 0) / max(cw.get(1, 1e-9), 1e-9):.0f}). Il faudra donc pondérer les classes et utiliser Recall/F1.")

# ----------------------------------------------------------------------------- 4
elif page == PAGES[4]:
    st.header("Parties IV, V, VI - Segmentation, features, analyse fréquentielle")
    ds = get_dataset(WIN, OVL, PRE)
    F = feats_of(ds)
    st.subheader("Question 7 - Fenêtrage")
    st.markdown(f"Fenêtre = **{WIN} s = {int(WIN * FS)} échantillons**, recouvrement **{int(OVL * 100)} %** "
                f"(pas = {WIN * (1 - OVL):.2f} s). Classe 1 si ≥ 50 % des échantillons sont en crise. 1 ligne = 1 fenêtre.")
    st.subheader("Question 8 - Vérification du dataset")
    c = st.columns(3)
    c[0].metric("Fenêtres totales", f"{len(ds):,}")
    c[1].metric("Classe 0", f"{(ds.Class == 0).sum():,}")
    c[2].metric("Classe 1", f"{(ds.Class == 1).sum():,}")
    st.caption(f"{len(F)} features par fenêtre. Colonnes spécifiées par le sujet : Mean, Std, Variance, Min, Max, RMS, Energy, + Amplitude ; "
               "suffixe _mean/_max = agrégation sur les canaux.")
    base = ["rec_id", "t_start"] + [f"{k}_mean" for k in ["Mean", "Std", "Variance", "Min", "Max", "Amplitude", "RMS", "Energy"]] + ["Class"]
    st.dataframe(ds[base].head(20), use_container_width=True)
    st.download_button("Télécharger le dataset (CSV)", ds.to_csv(index=False).encode(), "dataset_eeg.csv")
    st.subheader("Question 9 - Analyse fréquentielle (FFT) et puissance par bande")
    rec = st.selectbox("Enregistrement", sorted(annots.rec_id.unique()))
    x, ch, fs = get_rec(rec)
    xe, enames = select_channels(x, ch)
    xs = robust_scale(bandpass(xe, fs))
    sub = annots[annots.rec_id == rec]
    cn = st.selectbox("Canal", enames, index=enames.index("C3") if "C3" in enames else 0)
    t_c = float(sub.debut_s.iloc[0]) + 1
    t_n = max(0.0, float(sub.debut_s.iloc[0]) - 60)
    fig = make_subplots(rows=2, cols=2, subplot_titles=("Temps - hors crise", "Spectre - hors crise", "Temps - crise", "Spectre - crise"))
    for r_, t0, col in [(1, t_n, C0), (2, t_c, C1)]:
        seg = xs[int(t0 * fs): int((t0 + WIN) * fs), enames.index(cn)]
        fr, am = spectrum(seg, fs)
        fig.add_trace(go.Scatter(x=np.arange(len(seg)) / fs, y=seg, line=dict(color=col), showlegend=False), row=r_, col=1)
        m = fr <= 40
        fig.add_trace(go.Scatter(x=fr[m], y=am[m], line=dict(color=col), fill="tozeroy", showlegend=False), row=r_, col=2)
        fig.update_xaxes(title_text="Temps (s)", row=r_, col=1)
        fig.update_xaxes(title_text="Fréquence (Hz)", row=r_, col=2)
        fig.update_yaxes(title_text="Amplitude (u.a.)", row=r_, col=1)
        fig.update_yaxes(title_text="|FFT|", row=r_, col=2)
    fig.update_layout(height=550)
    st.plotly_chart(fig, use_container_width=True)
    st.markdown("Bandes : " + ", ".join(f"**{k}** {a}-{b} Hz" for k, (a, b) in BANDS.items()) +
                ". Une bande **3 Hz (2,5-4,5 Hz)** a été ajoutée : les pointes-ondes d'absence y concentrent leur énergie.")
    pw = ds[[f"Power_{b}_mean" for b in BANDS]].copy()
    pw["Classe"] = ds.Class.map({0: "Hors crise", 1: "Crise"})
    st.dataframe(pw.groupby("Classe").mean().round(4), use_container_width=True)

# ----------------------------------------------------------------------------- 5
elif page == PAGES[5]:
    st.header("Partie VII - Analyse exploratoire")
    ds = get_dataset(WIN, OVL, PRE)
    F = feats_of(ds)
    st.subheader("Question 10 - Comparaison des classes")
    default = ["Energy_mean", "Power_Delta_mean", "Power_Theta_mean", "Std_mean", "Power_Hz3_mean", "Rel_Hz3_mean"]
    sel = st.multiselect("Features", F, default=[d for d in default if d in F])
    d2 = ds.assign(Classe=ds.Class.map({0: "Hors crise", 1: "Crise"}))
    st.dataframe(d2.groupby("Classe")[sel].mean().T.round(4), use_container_width=True)
    if sel:
        cols = st.columns(2)
        for i, f in enumerate(sel):
            lo, hi = d2[f].quantile([0.005, 0.995])
            fig = px.box(d2[(d2[f] >= lo) & (d2[f] <= hi)], x="Classe", y=f, color="Classe", points=False,
                         color_discrete_map={"Hors crise": C0, "Crise": C1}, title=f)
            fig.update_layout(showlegend=False, height=320)
            cols[i % 2].plotly_chart(fig, use_container_width=True)
    st.subheader("Question 11 - Corrélations")
    corr = ds[F].corr()
    fig = px.imshow(corr, color_continuous_scale="RdBu_r", zmin=-1, zmax=1, aspect="auto", height=750)
    st.plotly_chart(fig, use_container_width=True)
    pairs = corr.where(np.triu(np.ones(corr.shape), 1).astype(bool)).stack()
    strong = pairs[pairs.abs() > 0.95].sort_values(key=abs, ascending=False)
    st.write(f"**1. Features fortement corrélées (|r| > 0,95)** : {len(strong)} paires. Top 10 :")
    st.dataframe(strong.head(10).rename("r").round(3))
    st.markdown("**2. Redondance** : Std, Variance, RMS, Energy (et Amplitude/Min/Max) mesurent la même chose (la puissance du signal) ; "
                "les puissances relatives et absolues de bandes se recouvrent. On peut en supprimer sans perte (voir page *Erreurs & améliorations*).")
    st.write("**3. Features les plus discriminantes** (information mutuelle avec la classe, + taille d'effet de Cohen) :")
    sample = ds.sample(min(len(ds), 12000), random_state=0)
    mi = pd.Series(mutual_info_classif(sample[F], sample.Class, random_state=0), index=F)
    g1, g0 = ds[ds.Class == 1][F], ds[ds.Class == 0][F]
    cohen = ((g1.mean() - g0.mean()) / np.sqrt((g1.var() + g0.var()) / 2)).abs()
    top = pd.DataFrame({"Info mutuelle": mi, "|d| de Cohen": cohen}).sort_values("Info mutuelle", ascending=False).head(15)
    st.dataframe(top.round(3), use_container_width=True)

# ----------------------------------------------------------------------------- 6
elif page == PAGES[6]:
    st.header("Parties VIII & IX - Modèles et évaluation")
    ds = get_dataset(WIN, OVL, PRE)
    F = feats_of(ds)
    c = st.columns(3)
    mode = {"Stratifiée sur les fenêtres (sujet)": "stratifie", "Par enregistrement (sans fuite)": "patient"}[
        c[0].radio("Type de séparation 80/20", ["Stratifiée sur les fenêtres (sujet)", "Par enregistrement (sans fuite)"])]
    bal = c[1].checkbox("Pondération des classes (class_weight=balanced)", value=True)
    tgt = {"Binaire (crise / non crise)": "Class", "3 états (inter / pré-crise / crise)": "Class3"}[
        c[2].radio("Cible", ["Binaire (crise / non crise)", "3 états (inter / pré-crise / crise)"])]
    if mode == "patient":
        st.info("Un même patient/enregistrement n'est jamais à la fois en train et en test : c'est l'estimation **réaliste** "
                "de la généralisation à un nouveau patient. Le split stratifié est optimiste car les fenêtres à 50 % de "
                "recouvrement voisines (quasi identiques) se retrouvent des deux côtés.")
    if st.button("Entraîner les 4 modèles", type="primary"):
        tr, te = split_data(ds, F, target=tgt, mode=mode)
        with st.spinner("Entraînement (DT, RF, KNN, SVM)..."):
            models = get_models(bal)
            if tgt == "Class3":
                from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
                rows, cms = [], {}
                for n, m in models.items():
                    m.fit(tr[F], tr[tgt])
                    p = m.predict(te[F])
                    rows.append({"Modèle": n, **metrics(te[tgt], p, avg="macro")})
                    cms[n] = confusion_matrix(te[tgt], p, labels=[0, 1, 2])
                res, fitted = pd.DataFrame(rows).set_index("Modèle"), models
            else:
                res, cms, _, fitted = evaluate_all(models, tr[F], tr[tgt], te[F], te[tgt])
        st.session_state["res6"] = dict(res=res, cms=cms, n_tr=len(tr), n_te=len(te), tgt=tgt, mode=mode,
                                        pos_tr=int((tr.Class == 1).sum()), pos_te=int((te.Class == 1).sum()))
    r = st.session_state.get("res6")
    if r:
        st.write(f"Train : {r['n_tr']:,} fenêtres ({r['pos_tr']} crises) - Test : {r['n_te']:,} fenêtres ({r['pos_te']} crises)"
                 + ("  *(métriques macro-moyennées)*" if r["tgt"] == "Class3" else ""))
        st.dataframe(r["res"].style.format("{:.3f}").highlight_max(color="#cfe8cf"), use_container_width=True)
        labels = ["Hors crise", "Crise"] if r["tgt"] == "Class" else ["Inter", "Pré-crise", "Crise"]
        cols = st.columns(4)
        for i, (n, cm) in enumerate(r["cms"].items()):
            fig = px.imshow(cm, text_auto=True, x=labels, y=labels, color_continuous_scale="Blues", title=n)
            fig.update_layout(xaxis_title="Prédit", yaxis_title="Réel", height=330, coloraxis_showscale=False)
            cols[i].plotly_chart(fig, use_container_width=True)
    st.subheader("Question 13 - Quelle métrique privilégier ?")
    st.markdown("""
- **Accuracy trompeuse** (d'où la **Balanced Accuracy** = moyenne du Recall de chaque classe, 0,5 pour « jamais de crise ») : avec ~96 % de fenêtres sans crise, un modèle qui répond toujours « pas de crise » obtient ~96 % d'Accuracy
  avec **Recall = 0** : il ne détecte aucune crise.
- **Precision** = parmi les alarmes, part de vraies crises (coût des fausses alarmes).
- **Recall** = part des crises réellement détectées. **Un faux négatif (crise manquée) est le plus grave** en pratique clinique.
- **F1-score** = compromis Precision/Recall, robuste au déséquilibre.

**Choix : le Recall est prioritaire, contrôlé par le F1-score** (pour éviter un modèle qui crie « crise » en permanence).
La matrice de confusion montre le détail FP/FN.
""")

# ----------------------------------------------------------------------------- 7
elif page == PAGES[7]:
    st.header("Partie X - Détection temporelle des crises")
    ds = get_dataset(WIN, OVL, PRE)
    F = feats_of(ds)
    st.markdown("Pour que les fenêtres soient continues dans le temps, on teste sur **un enregistrement entier mis de côté** "
                "(*leave-one-recording-out*) : le modèle n'a jamais vu ce patient.")
    c = st.columns(4)
    rec = c[0].selectbox("Enregistrement de test", sorted(annots.rec_id.unique()), index=sorted(annots.rec_id.unique()).index("210914B_A"))
    mname = c[1].selectbox("Modèle", ["Random Forest", "SVM", "Decision Tree", "KNN"])
    smooth = c[2].slider("Lissage (fenêtres)", 1, 9, 3, step=2)
    min_dur = c[3].slider("Durée min. (s)", 1.0, 6.0, 2.0, step=0.5)
    key = ("det", rec, mname, WIN, OVL)
    if key not in st.session_state:
        with st.spinner("Entraînement sur les autres enregistrements..."):
            tr, te = ds[ds.rec_id != rec], ds[ds.rec_id == rec].reset_index(drop=True)
            m = get_models(True)[mname].fit(tr[F], tr.Class)
            te = te.assign(pred=m.predict(te[F]))
            te["proba"] = m.predict_proba(te[F])[:, 1]
            st.session_state[key] = te
    te = st.session_state[key]
    true_iv = annots[annots.rec_id == rec][["debut_s", "fin_s", "duree_s"]]
    pred_iv = predictions_to_intervals(te.t_start, te.t_end, te.pred, smooth=smooth, min_dur=min_dur)
    x, ch, fs = get_rec(rec)
    sig = bandpass(x[:, [ch.index("C3")]].astype(float), fs)[:, 0]
    tt = np.arange(len(sig)) / fs
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.7, 0.3], vertical_spacing=0.04)
    fig.add_trace(go.Scattergl(x=tt[::4], y=sig[::4], mode="lines", line=dict(color="#555", width=1), name="EEG C3"), row=1, col=1)
    for _, r in true_iv.iterrows():
        fig.add_vrect(x0=r.debut_s, x1=r.fin_s, fillcolor=C1, opacity=0.4, line_width=0, row=1, col=1)
    fig.add_trace(go.Scatter(x=te.t_start, y=te.proba, mode="lines", line=dict(color=CP), name="Proba. crise (modèle)"), row=2, col=1)
    for _, r in pred_iv.iterrows():
        fig.add_vrect(x0=r.debut_s, x1=r.fin_s, fillcolor="#2CA02C", opacity=0.35, line_width=0, row=2, col=1)
    fig.update_xaxes(title_text="Temps (s)", row=2, col=1)
    fig.update_yaxes(title_text="Amplitude (µV)", row=1, col=1)
    fig.update_yaxes(title_text="Proba", row=2, col=1)
    fig.update_layout(height=550, title="Rouge = crises réelles ; vert = crises prédites")
    st.plotly_chart(fig, use_container_width=True)
    st.subheader("Question 14 - Intervalles détectés vs annotations")
    hit, n_true, fa = compare_intervals(true_iv, pred_iv)
    c = st.columns(3)
    c[0].metric("Crises détectées", f"{hit} / {n_true}")
    c[1].metric("Fausses alarmes (intervalles)", fa)
    c[2].metric("Crises prédites", len(pred_iv))
    for i, r in pred_iv.reset_index(drop=True).iterrows():
        st.write(f"Crise détectée {i + 1} : {r.debut_s:.1f} s → {r.fin_s:.1f} s")
    c = st.columns(2)
    c[0].write("**Annotations réelles**")
    c[0].dataframe(true_iv.round(1).reset_index(drop=True))
    c[1].write("**Détections**")
    c[1].dataframe(pred_iv.round(1))

# ----------------------------------------------------------------------------- 8
else:
    st.header("Parties XI & XII - Analyse des erreurs et améliorations")
    ds = get_dataset(WIN, OVL, PRE)
    F = feats_of(ds)
    st.subheader("Analyse des erreurs (split par enregistrement)")
    mname = st.selectbox("Modèle", ["Random Forest", "SVM", "Decision Tree", "KNN"])
    if st.button("Analyser", type="primary"):
        tr, te = split_data(ds, F, mode="patient")
        m = get_models(True)[mname].fit(tr[F], tr.Class)
        te = te.assign(pred=m.predict(te[F]))
        st.session_state["err"] = (te, mname)
    if "err" in st.session_state:
        te, mn = st.session_state["err"]
        tp = ((te.Class == 1) & (te.pred == 1)).sum(); tn = ((te.Class == 0) & (te.pred == 0)).sum()
        fp = ((te.Class == 0) & (te.pred == 1)).sum(); fn = ((te.Class == 1) & (te.pred == 0)).sum()
        c = st.columns(4)
        c[0].metric("Vrais positifs", tp); c[1].metric("Vrais négatifs", tn)
        c[2].metric("Faux positifs", fp); c[3].metric("Faux négatifs", fn)
        te = te.assign(Type=np.select([(te.Class == 1) & (te.pred == 1), (te.Class == 0) & (te.pred == 1), (te.Class == 1) & (te.pred == 0)],
                                      ["TP", "FP", "FN"], "TN"))
        err = te[te.Type.isin(["FP", "FN"])]
        st.write("Erreurs par enregistrement :")
        st.dataframe(err.groupby(["rec_id", "Type"]).size().unstack(fill_value=0))
        st.write("Les **faux négatifs** tombent souvent sur les bords des crises (fenêtre ayant 50-60 % de crise) et sur les crises "
                 "courtes/peu amples ; les **faux positifs** sur des artefacts (mouvements, clignements, EMG) d'amplitude élevée.")
        feat = st.selectbox("Comparer une feature entre TP / FN / FP / TN", F, index=F.index("Power_Hz3_mean") if "Power_Hz3_mean" in F else 0)
        lo, hi = te[feat].quantile([0.005, 0.995])
        st.plotly_chart(px.box(te[(te[feat] >= lo) & (te[feat] <= hi)], x="Type", y=feat, color="Type", points=False), use_container_width=True)
        st.dataframe(err.sort_values(["rec_id", "t_start"])[["rec_id", "t_start", "t_end", "frac_crise", "Class", "pred", "Type"]].head(200), use_container_width=True)

    st.subheader("Question - Améliorations testées")
    st.markdown("Chaque amélioration est évaluée en **validation croisée par enregistrement** (5 plis, Random Forest, F1 et Recall "
                "sur la classe crise) pour que la comparaison soit honnête.")
    opts = st.multiselect("Améliorations à tester", [
        "Référence (toutes features, class_weight)", "Sans pondération des classes (pour mesurer l'effet)",
        "Sélection de features (suppression corrélations > 0,95)", "Top 15 features (importance RF)",
        "Seuil de décision optimisé (proba > 0,3)"], default=["Référence (toutes features, class_weight)", "Sélection de features (suppression corrélations > 0,95)", "Seuil de décision optimisé (proba > 0,3)"])
    if st.button("Lancer la comparaison"):
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.model_selection import GroupKFold
        g = ds.rec_id
        corr = ds[F].corr().abs()
        up = corr.where(np.triu(np.ones(corr.shape), 1).astype(bool))
        F_sel = [c for c in F if not (up[c] > 0.95).any()]
        rf0 = RandomForestClassifier(n_estimators=100, min_samples_leaf=2, class_weight="balanced", n_jobs=-1, random_state=0).fit(ds[F], ds.Class)
        F_top = list(pd.Series(rf0.feature_importances_, index=F).nlargest(15).index)
        res = []
        for o in opts:
            feats, cw, thr = F, "balanced", 0.5
            if o.startswith("Sans"):
                cw = None
            elif o.startswith("Sélection"):
                feats = F_sel
            elif o.startswith("Top"):
                feats = F_top
            elif o.startswith("Seuil"):
                thr = 0.3
            ps = np.zeros(len(ds))
            for tr_i, te_i in GroupKFold(5).split(ds, groups=g):
                m = RandomForestClassifier(n_estimators=100, min_samples_leaf=2, class_weight=cw, n_jobs=-1, random_state=0)
                m.fit(ds.iloc[tr_i][feats], ds.Class.iloc[tr_i])
                ps[te_i] = m.predict_proba(ds.iloc[te_i][feats])[:, 1]
            mt = metrics(ds.Class, (ps > thr).astype(int))
            res.append({"Variante": o, "Nb features": len(feats), **mt})
        st.dataframe(pd.DataFrame(res).set_index("Variante").style.format({k: "{:.3f}" for k in ["Balanced Accuracy", "Precision", "Recall", "F1-score"]}), use_container_width=True)
    st.markdown("""
**Autres pistes** (non implémentées ici) : changer taille de fenêtre/recouvrement (barre latérale, le dataset est reconstruit),
ajouter des features inter-canaux (cohérence, synchronie : les absences sont *généralisées*), contexte temporel
(features des fenêtres voisines), SMOTE, réseau de neurones 1D-CNN sur le signal brut.
""")
