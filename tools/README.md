# Outils de développement

Scripts utilisés pendant le projet. À lancer **depuis la racine du projet** (ex. `python tools/compare.py`).
Ils lisent/écrivent des fichiers intermédiaires dans `cache/` (non versionné, régénérable).

| Script | Rôle |
|---|---|
| `compare.py`, `metrics.py` | Comparaison des 4 modèles en leave-one-patient-out, métriques, intervalles de confiance (première version ; la référence actuelle est `python -m src.evaluate`) |
| `experiments.py`, `analyse2.py`, `tuning.py` | Expériences complémentaires : normalisation des features par enregistrement (abandonnée), correction des bornes (abandonnée), réglages du Random Forest, KNN avec sous-échantillonnage (adopté) |
| `inventory.py` | Inventaire des widgets de chaque page de l'interface (vérifier qu'aucune fonctionnalité n'est perdue lors d'une refonte) |
| `ui_test.py`, `ui_test2.py` | Tests de comportement de l'interface (navigation, conservation des réglages, changement de patient, entraînement) |
| `shot.py`, `shot_click.py`, `restart.sh` | Captures d'écran de l'interface avec le navigateur Edge (Playwright) ; `restart.sh` relance le serveur de développement |
| `make_pdf.py` | Génère `Guide_complet_interface_et_methodologie.pdf` |
| `make_doc.js` | Génère un document Word de méthodologie. **Périmé** : il contient les chiffres d'avant l'évaluation de référence ; à mettre à jour avant usage |

Dépendances : `pip install -r tools/requirements-dev.txt`.
