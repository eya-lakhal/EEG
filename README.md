# Détection des crises d'absence (EEG) - Mini-projet Data Mining

    pip install -r requirements.txt
    python -m src.convert_to_csv      # une seule fois : .mat + Excel -> data_csv/ (sans perte, vérifié)
    streamlit run app.py

- `data_csv/` : données converties (signal complet par enregistrement, marqueurs, annotations)
- `src/convert_to_csv.py` : conversion + vérification d'égalité exacte avec les .mat
- `src/data_loader.py` : lecture des CSV et des annotations
- `src/preprocessing.py` : filtre 0,5-40 Hz, normalisation robuste, étiquettes, fenêtrage
- `src/features.py` : features temporelles et fréquentielles
- `src/dataset.py`, `src/selection.py`, `src/models.py`, `src/detection.py` : dataset, sélection, DT/RF/KNN/SVM, intervalles
- `app.py` : application Streamlit patient par patient ; `app_questions.py` : version par questions du sujet
- `cache/` : caches dérivés (régénérés automatiquement)
