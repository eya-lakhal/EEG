# Détection des crises d'absence à partir de signaux EEG

Mini-projet Data Mining (3AINFO) : du signal EEG brut aux intervalles de crise (prétraitement, fenêtres, features,
sélection, classification DT / RF / KNN / SVM, data augmentation des crises, détection temporelle, étude de faisabilité de la prévention) avec une application Streamlit interactive.

## Démarrage rapide

    pip install -r requirements.txt
    python -m src.convert_to_csv      # une seule fois : .mat + Excel -> data_csv/ (3,3 Go, vérifié bit à bit)
    streamlit run app.py              # interface patient par patient (navigation par étapes)

Premier lancement : la construction du dataset prend environ 1 minute, la première prédiction « un patient laissé de côté »
1 à 2 minutes (ensuite tout est en cache). Un lien direct est possible : `http://localhost:8501/?step=5&rec=210914B_A`.

Évaluation de référence (environ 25 min, dont 18 pour le SVM) : `python -m src.evaluate` -> `results/`.
Même évaluation avec data augmentation des crises : `python -m src.evaluate --augment` -> fichiers `results/*_augment.*`.
Étude de la prévention (pré-crise contre calme) : `python -m tools.prevention_study` -> `results/prevention_*.csv`.

Métrique principale affichée : **Balanced Accuracy** (moyenne du Recall des deux classes, 0,5 pour « jamais de crise »), avec Recall, Précision, F1 et PR-AUC.

## Données

- `*_0000d.mat` (signal, 22 enregistrements) et `*_0000h.mat` (marqueurs de l'enregistreur) : données sources fournies pour le cours.
- `Annotations MAJ.xlsx` : début et fin des crises (temps en fraction de jour).
- `data_csv/` n'est **pas** dans le dépôt (fichiers de 380 Mo, GitHub refuse au-delà de 100 Mo) : il se régénère avec
  `python -m src.convert_to_csv`. Toute l'application lit ces CSV, jamais les `.mat` directement.

> Ce sont des enregistrements EEG d'enfants (identifiants anonymisés, âge et sexe dans l'Excel).

## Organisation

| Dossier / fichier | Rôle |
|---|---|
| `app.py` | Application Streamlit (7 pages, navigation par étapes) |
| `ui/` | Identité visuelle : `theme.py` (palette, CSS, graphiques), `components.py` (composants) ; thème de base dans `.streamlit/config.toml` |
| `src/` | Tous les calculs : `convert_to_csv`, `data_loader`, `preprocessing`, `features`, `dataset`, `selection`, `models` (fabrique unique `build_model`), `detection` (appariement strict), `evaluate` |
| `results/` | Résultats de référence (scores leave-one-patient-out, résumés, réglages) lus par l'application |
| `tools/` | Scripts de développement : expériences, tests de l'interface, captures d'écran, génération du PDF (voir `tools/README.md`) |
| `app_questions.py` | Ancienne interface organisée selon les questions du sujet |
| `Guide_complet_interface_et_methodologie.pdf` | Méthodologie complète, choix justifiés, interface, limites |
| `audit_210208B_G_marqueurs.png` | Signal de 210208B_G autour de 3 marqueurs « absence » absents de l'Excel (étiquettes probablement incomplètes) |
| `cache/`, `data_csv/` | Fichiers dérivés, non versionnés, régénérés automatiquement |

## Résultats principaux

Validation « un patient laissé de côté » (22 enregistrements, 82 crises, seuil 0,5) : le Random Forest a la meilleure PR-AUC (0,59)
mais n'est pas statistiquement distinguable du KNN corrigé ni du SVM avec 19 patients ; le Decision Tree est nettement plus faible.
Détection avec appariement strict : Random Forest 68 / 82 crises, 4 fausses alarmes par heure. Détails, limites et points d'attention :
voir le PDF (parties B, D, E).
