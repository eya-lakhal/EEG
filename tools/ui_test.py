import warnings; warnings.filterwarnings("ignore")
from streamlit.testing.v1 import AppTest
fails = []
def check(cond, msg):
    print(("OK  " if cond else "ÉCHEC "), msg, flush=True)
    if not cond: fails.append(msg)
def clean(at, where):
    ex = [e.value[:150] for e in at.exception]; wa = [w.value[:150] for w in at.warning]; er = [e.value[:150] for e in at.error]
    check(not ex and not wa and not er, f"{where} : aucune exception / avertissement / erreur {ex or ''}{wa or ''}{er or ''}")
def press(at, label):
    b = [b for b in at.button if label in b.label]
    assert b, f"bouton introuvable: {label} (dispo: {[x.label for x in at.button][:12]})"
    b[0].click().run()

at = AppTest.from_file("app.py", default_timeout=3000).run()
clean(at, "démarrage"); check(at.session_state["step"] == 0, "démarre sur la Vue d'ensemble")

# --- stepper et accès rapide
press(at, "1 Prétraitement"); check(at.session_state["step"] == 1, "stepper : clic sur « 1 Prétraitement » -> étape 1"); clean(at, "étape 1")
at.number_input(key="pre_lo").set_value(1.0).run(); check(at.session_state["pre_lo"] == 1.0, "réglage passe-haut = 1.0")
press(at, "Vue d'ensemble"); check(at.session_state["step"] == 0, "retour à la vue d'ensemble")
press(at, "1  Prétraitement"); check(at.session_state["step"] == 1, "accès rapide -> étape 1")
check(at.number_input(key="pre_lo").value == 1.0, "persistance : passe-haut conservé après être passé par une autre étape")
at.toggle(key="pre_scale").set_value(False).run(); press(at, "Suivante : "); check(at.session_state["step"] == 2, "pied de page : « Suivante » -> étape 2"); clean(at, "étape 2 (fenêtres)")
press(at, "Précédente : "); check(at.session_state["step"] == 1, "pied de page : « Précédente » -> étape 1")
check(at.toggle(key="pre_scale").value is False, "persistance : normalisation désactivée conservée")

# --- features : persistance du curseur
press(at, "3 Meilleures features"); at.slider(key="feat_K").set_value(20).run(); clean(at, "étape 3")
press(at, "4 Classification"); press(at, "3 Meilleures features")
check(at.slider(key="feat_K").value == 20, "persistance : K = 20 conservé après un détour par la classification")
check(len(at.session_state["best_feats"]) == 20, "état partagé : 20 features retenues mémorisées pour la prévention")
press(at, "6 Prévention"); clean(at, "étape 6 (prévention)")

# --- changement de patient, y compris l'enregistrement à 8 canaux
press(at, "2 Fenêtre par fenêtre"); clean(at, "étape 2, patient 1")
sel = at.sidebar.selectbox[0]; opts = list(sel.options)
sel.select(opts[2]).run(); clean(at, "étape 2 après passage à l'enregistrement à 8 canaux (" + opts[2][:9] + ")")
check(len(at.multiselect(key="w_ch").value) == 8, "canaux affichés : 8 canaux pour 200625A_F (pas d'erreur de sélection périmée)")
sel = at.sidebar.selectbox[0]; sel.select(opts[8]).run(); clean(at, "étape 2 après retour à un patient à 19 canaux")
check(len(at.multiselect(key="w_ch").value) == 19, "canaux affichés : 19 canaux à nouveau")
for lab in ("Suivante", "Prochaine alerte"):
    press(at, lab); clean(at, "navigation « " + lab + " »")
check(len([m for m in at.metric]) == 4, "4 indicateurs de fenêtre affichés")

# --- détection + classification (calculs longs)
press(at, "5 Détection temporelle"); clean(at, "étape 5"); print("   indicateurs :", [(m.label, m.value) for m in at.metric])
at.slider(key="det_iou").set_value(0.5).run(); clean(at, "étape 5 avec IoU 0.5")
press(at, "4 Classification")
press(at, "Entraîner"); clean(at, "entraînement des 4 modèles"); check("cls" in at.session_state, "résultats d'entraînement mémorisés")
press(at, "Calculer par patient"); clean(at, "calcul patient par patient"); check("pp" in at.session_state, "résultats patient par patient mémorisés")
press(at, "Vue d'ensemble"); clean(at, "vue d'ensemble après calculs")
html = " ".join(str(m.value) for m in at.markdown)
check("Fait" in html and "Résultats disponibles pour les 4 modèles" in html and "À faire" not in html, "état de l'analyse : toutes les étapes consultées apparaissent comme faites")
print("\nÉCHECS :", fails if fails else "aucun")
