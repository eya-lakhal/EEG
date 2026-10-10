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

press(at, "2 Fenêtre par fenêtre"); clean(at, "étape 2")
sel = at.sidebar.selectbox[0]; opts = list(sel.options)
sel.select(opts[2]).run(); clean(at, "8 canaux"); check(len(at.multiselect(key="w_ch").value) == 8, "8 canaux pour 200625A_F")
sel = at.sidebar.selectbox[0]; sel.select(opts[8]).run(); clean(at, "retour 19 canaux"); check(len(at.multiselect(key="w_ch").value) == 19, "19 canaux à nouveau")
at.multiselect(key="w_ch").set_value(["Fp1", "Cz"]).run(); sel = at.sidebar.selectbox[0]; sel.select(opts[9]).run()
check(len(at.multiselect(key="w_ch").value) == 2, "sélection de 2 canaux conservée entre deux patients à 19 canaux (comme avant)")
print("ÉCHECS :", fails if fails else "aucun")
