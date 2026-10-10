const fs = require('fs');
const { Document, Packer, Paragraph, TextRun, HeadingLevel, Table, TableRow, TableCell, WidthType,
  ShadingType, AlignmentType, LevelFormat, BorderStyle, PageBreak, Footer, PageNumber, TableOfContents } = require('docx');

const FONT = 'Calibri';
function runs(t, extra = {}) {
  return t.split(/(\*\*[^*]+\*\*|`[^`]+`)/).filter(Boolean).map(s => {
    if (s.startsWith('**')) return new TextRun({ text: s.slice(2, -2), bold: true, font: FONT, ...extra });
    if (s.startsWith('`')) return new TextRun({ text: s.slice(1, -1), font: 'Consolas', size: 19, ...extra });
    return new TextRun({ text: s, font: FONT, ...extra });
  });
}
const P = (t, o = {}) => new Paragraph({ children: runs(t), spacing: { after: 110, line: 276 }, ...o });
const H1 = t => new Paragraph({ heading: HeadingLevel.HEADING_1, children: [new TextRun({ text: t, font: FONT })] });
const H2 = t => new Paragraph({ heading: HeadingLevel.HEADING_2, children: [new TextRun({ text: t, font: FONT })] });
const B = t => new Paragraph({ numbering: { reference: 'bul', level: 0 }, children: runs(t), spacing: { after: 60, line: 270 } });
const N = t => new Paragraph({ numbering: { reference: 'num', level: 0 }, children: runs(t), spacing: { after: 60, line: 270 } });
const note = t => new Paragraph({ children: runs(t, { italics: true, color: '555555' }), spacing: { after: 120 },
  border: { left: { style: BorderStyle.SINGLE, size: 12, color: '4C78A8', space: 8 } }, indent: { left: 200 } });

const bd = { style: BorderStyle.SINGLE, size: 4, color: 'BBBBBB' };
const borders = { top: bd, bottom: bd, left: bd, right: bd };
function table(head, rows, widths, opts = {}) {
  const tot = widths.reduce((a, b) => a + b, 0);
  const cell = (t, w, h, i) => new TableCell({
    width: { size: w, type: WidthType.DXA }, borders,
    shading: h ? { type: ShadingType.CLEAR, fill: 'DCE6F1', color: 'auto' } : (opts.hl && opts.hl.includes(i) ? { type: ShadingType.CLEAR, fill: 'E8F3E8', color: 'auto' } : undefined),
    margins: { top: 60, bottom: 60, left: 100, right: 100 },
    children: [new Paragraph({ children: runs(String(t), { size: 19, bold: h }), spacing: { after: 0 } })],
  });
  return new Table({
    width: { size: tot, type: WidthType.DXA }, columnWidths: widths,
    rows: [new TableRow({ tableHeader: true, children: head.map((t, i) => cell(t, widths[i], true)) }),
      ...rows.map((r, ri) => new TableRow({ cantSplit: true, children: r.map((t, i) => cell(t, widths[i], false, ri)) }))],
  });
}
const gap = () => new Paragraph({ children: [], spacing: { after: 100 } });

const c = [];
c.push(new Paragraph({ alignment: AlignmentType.CENTER, spacing: { before: 600, after: 120 },
  children: [new TextRun({ text: 'Détection automatique des crises d\'absence à partir de signaux EEG', bold: true, size: 40, font: FONT, color: '1F3864' })] }));
c.push(new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 120 },
  children: [new TextRun({ text: 'Méthodologie détaillée et justification de chaque choix', size: 28, font: FONT, color: '4C78A8' })] }));
c.push(new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 360 },
  children: [new TextRun({ text: 'Mini-projet Data Mining - 3AINFO - Prétraitement, fenêtrage, features, classification, détection temporelle', size: 21, font: FONT, color: '666666' })] }));
c.push(new TableOfContents('Sommaire', { hyperlink: true, headingStyleRange: '1-2' }));
c.push(new Paragraph({ children: [new PageBreak()] }));

// 1
c.push(H1('1. Vue d\'ensemble du pipeline'));
c.push(P('L\'objectif est de passer d\'un signal EEG brut à des intervalles de crise (début, fin). Chaque étape a un fichier Python dédié, ce qui permet de la relire et de la tester isolément.'));
c.push(table(['Étape', 'Entrée', 'Sortie', 'Fichier'], [
  ['Conversion en CSV (une fois)', 'Fichiers .mat + Annotations MAJ.xlsx', 'data_csv/*.csv identiques bit à bit aux sources', 'src/convert_to_csv.py'],
  ['Lecture', 'CSV de data_csv/', 'Signal continu (échantillons × canaux) + liste des crises en secondes', 'src/data_loader.py'],
  ['Prétraitement', 'Signal brut', 'Signal filtré 0,5-40 Hz et normalisé', 'src/preprocessing.py'],
  ['Étiquetage + fenêtrage', 'Signal + crises', 'Fenêtres de 2 s (pas 1 s) avec une classe', 'src/preprocessing.py'],
  ['Features', 'Fenêtres', '56 features par fenêtre', 'src/features.py'],
  ['Dataset', 'Tous les patients', '33 206 lignes × 62 colonnes', 'src/dataset.py'],
  ['Sélection', 'Dataset', '15 meilleures features non redondantes', 'src/selection.py'],
  ['Classification', 'Features', 'Probabilité de crise par fenêtre', 'src/models.py'],
  ['Détection temporelle', 'Probabilités', 'Intervalles [début, fin] de crise', 'src/detection.py'],
], [1900, 2500, 3338, 1900]));
c.push(gap());
c.push(P('L\'application `app.py` (Streamlit) expose ces étapes patient par patient. L\'ancienne version organisée par questions du sujet est conservée dans `app_questions.py`.'));

// 2
c.push(H1('2. Données et annotations'));
c.push(H2('2.1 Format des signaux'));
c.push(B('**22 enregistrements** de 1 173 s à 3 600 s (environ 9,2 heures au total), échantillonnés à **256 Hz**.'));
c.push(B('Chaque fichier `*_0000d.mat` est une `timetable` MATLAB (format MCOS) que `scipy` ne sait pas lire. J\'utilise la bibliothèque `mat-io` une seule fois, pour la conversion en CSV (section 2.5). Chaque ligne de la table d\'origine représente 1 seconde et contient 256 échantillons par canal ; je les aplatis en un signal continu (1 ligne = 1 échantillon), comme le demande le sujet.'));
c.push(B('Les fichiers `*_0000h.mat` contiennent les marqueurs de l\'enregistreur (« crise », « ARTEFACT », « HPN », « SLI=… »). Ils ont servi à **valider** les annotations Excel (voir 2.3), pas à entraîner.'));
c.push(H2('2.2 Choix des canaux'));
c.push(P('Les 19 canaux EEG du système 10-20 sont : Fp1, Fp2, F7, F3, Fz, F4, F8, T3, C3, Cz, C4, T4, T5, P3, Pz, P4, T6, O1, O2. **21 fichiers sur 22** les possèdent tous ; le fichier 200625A_F n\'en a que 8 (Fp1, Fp2, C3, C4, O1, O2, T3, T4). Chaque patient utilise **tous les canaux EEG disponibles** dans son fichier.'));
c.push(P('Les canaux **ECG, EMG et SLI sont exclus** : ce ne sont pas des signaux cérébraux. L\'ECG est cardiaque, l\'EMG est musculaire (source d\'artefacts), et SLI est le signal de stimulation lumineuse intermittente. Les inclure risquerait d\'apprendre à reconnaître un artefact ou la stimulation plutôt que la crise.'));
c.push(H2('2.3 Annotations Excel'));
c.push(B('Les instants de `Annotations MAJ.xlsx` sont des **fractions de jour** (l\'heure de la journée) : une durée de 0,00008 correspond à 7 s. Le temps relatif à l\'enregistrement est donc **(DebCE − DebutTrace) × 86 400 s**.'));
c.push(B('**Validation** : pour 190304A_E, la formule donne une crise à 1 291 s, ce qui coïncide avec le marqueur « crise » à 21:32 du fichier `h`.'));
c.push(B('**82 crises valides** dans 19 enregistrements : durée moyenne 13,2 s, minimale 2 s, maximale 60 s. Les durées sont très hétérogènes (écart-type de 10 s).'));
c.push(B('Trois enregistrements n\'ont aucune crise annotée (200625A_F, 210204B_C, 210208B_G). Je les garde comme exemples négatifs.'));
c.push(H2('2.4 Anomalies détectées et corrigées'));
c.push(B('**Doublons** : 230406B-A et 230515B-G apparaissent deux fois avec de légères différences. Je garde la **dernière ligne** (le fichier s\'appelle « MAJ », mise à jour).'));
c.push(B('**Crise aberrante** : pour 210504B-C, FinCE1 donne une crise de 21 608 s (6 h) dans un enregistrement de 3 121 s. Une règle automatique écarte toute crise de durée ≤ 0, > 120 s ou de début négatif. Cette crise est donc ignorée et signalée dans l\'application.'));
c.push(note('Choix à confirmer avec l\'encadrant : j\'ai supposé que la ligne la plus récente est la bonne pour les doublons.'));

c.push(H2('2.5 Conversion de tous les fichiers en CSV, sans perte'));
c.push(P('Tout le pipeline travaille désormais sur des **fichiers CSV** et plus sur les `.mat` ni l\'Excel. La conversion est faite une fois par `python -m src.convert_to_csv` et produit le dossier `data_csv/` :'));
c.push(table(['Fichier', 'Contenu'], [
  ['<rec>.csv (22 fichiers)', 'Colonne `Time` (s) puis **tous les canaux** (19 EEG + ECG/EMG/SLI selon le fichier), avec les noms et l\'ordre d\'origine. Une ligne = un échantillon à 256 Hz.'],
  ['<rec>_events.csv (22 fichiers)', 'Marqueurs de l\'enregistreur : Onset_s, Duration_s, Annotations.'],
  ['Annotations_MAJ.csv', 'Copie de l\'Excel (seules les lignes entièrement vides sont omises ; le fichier Excel en contenait plus d\'un million).'],
  ['manifest.csv', 'Récapitulatif : nombre de lignes, canaux, taille, résultat de la vérification.'],
], [3000, 6638]));
c.push(gap());
c.push(B('**Aucune perte numérique** : les valeurs sont écrites en float64 (pleine précision) et chaque CSV est relu et comparé aux données du `.mat` d\'origine. Les 22 fichiers sont **identiques bit à bit** (égalité exacte des valeurs, des noms de colonnes et de l\'axe du temps). Les canaux ECG, EMG et SLI sont conservés dans les CSV même s\'ils ne sont pas utilisés par les modèles.'));
c.push(B('Taille : environ 3,2 Go au total (de 110 Mo à 380 Mo par enregistrement). La lecture des 22 CSV avec `pyarrow` prend environ 5 secondes.'));
c.push(B('Un cache `cache/<rec>_f64.npz` (float64) est dérivé du CSV pour accélérer les ouvertures ; il peut être supprimé et se régénère depuis le CSV.'));
c.push(note('Seule exception : dans les marqueurs textuels (fichiers events), les lettres accentuées (par exemple « Début ») sont déjà abîmées à la lecture du `.mat` par la bibliothèque (caractères de remplacement). Ces textes sont purement informatifs et ne sont pas utilisés par le pipeline.'));
c.push(note('Après conversion, le dataset a été reconstruit à partir des CSV. Les 56 features sont identiques à l\'ancienne version (calculée en float32) avec une corrélation supérieure à 0,999999998 ; tous les résultats de ce document ont été recalculés depuis les CSV.'));

// 3
c.push(H1('3. Prétraitement'));
c.push(H2('3.1 Filtre passe-bande 0,5-40 Hz'));
c.push(P('Filtre de **Butterworth d\'ordre 4**, appliqué en **aller-retour** (`sosfiltfilt`), sous forme de sections du second ordre (SOS).'));
c.push(table(['Choix', 'Justification'], [
  ['Passe-haut 0,5 Hz', 'Supprime la dérive lente (mouvements, sueur, polarisation des électrodes) qui domine l\'amplitude brute sans rapport avec la crise.'],
  ['Passe-bas 40 Hz', 'Élimine le bruit secteur à 50 Hz et l\'activité musculaire haute fréquence. Les pointes-ondes d\'absence (environ 3 Hz) et leurs harmoniques restent dans la bande.'],
  ['Phase nulle (aller-retour)', 'Un filtre causal décalerait le signal dans le temps. Comme on veut localiser le début et la fin des crises à la seconde près, aucun décalage n\'est acceptable.'],
  ['Forme SOS', 'Numériquement stable, contrairement à la forme polynomiale (b, a) pour un passe-bande.'],
], [2600, 7038]));
c.push(gap());
c.push(H2('3.2 Normalisation robuste par canal et par enregistrement'));
c.push(P('Pour chaque canal : **x normalisé = (x − médiane) / IQR**, avec IQR = écart interquartile (Q3 − Q1), calculés sur tout l\'enregistrement.'));
c.push(B('Les amplitudes varient fortement d\'un patient à l\'autre et d\'une session à l\'autre (impédances de 15 kΩ à 90 kΩ dans le fichier `h`, âges de 5 à 25 ans). Sans normalisation, un modèle apprendrait « le patient » plutôt que « la crise ».'));
c.push(B('La médiane et l\'IQR sont **robustes** : ni les artefacts ni les crises elles-mêmes (moins de 5 % du temps) ne les déplacent, contrairement à la moyenne et à l\'écart-type.'));
c.push(H2('3.3 Ce qui n\'est pas fait et pourquoi'));
c.push(B('Pas de suppression d\'artefacts (ICA, rejet de fenêtres) : cela complexifie le pipeline et les artefacts font partie du problème réel (voir les faux positifs).'));
c.push(B('Pas de re-référencement : les montages ne sont pas documentés dans les fichiers.'));
c.push(note('Limite : la normalisation utilise tout l\'enregistrement. En usage temps réel, il faudrait une ligne de base calculée sur une fenêtre glissante passée.'));
c.push(H2('3.4 Visualisation dans l\'application (page « 1 · Prétraitement »)'));
c.push(P('La page affiche, canal par canal, le signal **brut → filtré → normalisé** avec les crises en rouge, le spectre avant/après filtrage (Welch), un spectrogramme (les crises y apparaissent comme des bandes d\'énergie vers 3 Hz) et un tableau de qualité par canal pour repérer les canaux bruités. Les fréquences de coupure et la normalisation sont réglables pour observer leur effet.'));

// 4
c.push(H1('4. Segmentation en fenêtres et étiquetage'));
c.push(H2('4.1 Étiquette par échantillon'));
c.push(P('Un vecteur de 0/1 est créé pour chaque échantillon : 1 entre le début et la fin de chaque crise annotée, 0 ailleurs (Questions 5 et 6 du sujet).'));
c.push(H2('4.2 Paramètres de fenêtrage'));
c.push(table(['Paramètre', 'Valeur', 'Justification'], [
  ['Taille', '2 s = 512 échantillons', 'Une fenêtre contient environ 6 cycles d\'une pointe-onde à 3 Hz. La résolution fréquentielle est de 0,5 Hz (1/2 s), suffisante pour séparer Delta, Theta et la bande 3 Hz. 1 s donnerait 1 Hz de résolution (trop grossier) ; 4 s lisserait les crises de 2 à 5 s.'],
  ['Recouvrement', '50 % (pas de 1 s)', 'Donne une décision par seconde et évite qu\'une crise soit coupée en deux fenêtres inexploitables. C\'est aussi la valeur du sujet.'],
  ['Règle de classe', '≥ 50 % d\'échantillons en crise → classe 1', 'Règle du sujet. Une fenêtre à cheval sur le début de crise est « crise » si au moins une seconde sur deux est ictale.'],
], [1600, 2400, 5638]));
c.push(gap());
c.push(H2('4.3 Dataset obtenu'));
c.push(table(['Quantité', 'Valeur'], [
  ['Fenêtres totales', '33 206'],
  ['Classe 0 (hors crise)', '32 038 (96,5 %)'],
  ['Classe 1 (crise)', '1 168 (3,5 %)'],
  ['Pré-crise (10 s avant le début, `Class3` = 1)', '806 fenêtres'],
  ['Colonnes', '62 = 56 features + rec_id, t_start, t_end, frac_crise, Class, Class3'],
], [5000, 4638]));
c.push(gap());
c.push(P('Le **déséquilibre de 1 pour 27** est important. Il impose de pondérer les classes (`class_weight="balanced"`) et de juger les modèles sur le Recall, la précision et le F1-score plutôt que sur l\'Accuracy (un modèle qui répond toujours « pas de crise » obtient 96,5 %).'));
c.push(note('Conséquence du recouvrement : deux fenêtres voisines partagent la moitié de leurs échantillons. Un split aléatoire met donc des fenêtres quasi identiques en train et en test (fuite d\'information). C\'est pourquoi j\'évalue aussi en séparant par patient (section 8).'));

// 5
c.push(H1('5. Extraction des features'));
c.push(P('Pour chaque fenêtre et **chaque canal**, 28 caractéristiques sont calculées (`src/features.py`), puis agrégées sur les canaux par la **moyenne** et le **maximum** : 28 × 2 = **56 features** par fenêtre.'));
c.push(H2('5.1 Features temporelles (14)'));
c.push(table(['Feature', 'Définition', 'Intérêt pour les crises'], [
  ['Mean', 'Moyenne des échantillons', 'Proche de 0 après filtrage : peu informative (exigée par le sujet).'],
  ['Std / Variance', 'Écart-type et variance', 'L\'amplitude augmente nettement pendant une crise.'],
  ['Min / Max', 'Valeurs extrêmes', 'Capte les pointes de grande amplitude.'],
  ['Amplitude', 'Max − Min (pic-à-pic)', 'Mesure directe de l\'ampleur des décharges.'],
  ['RMS', 'Racine de la moyenne des carrés', 'Puissance moyenne du signal.'],
  ['Energy', 'Σ x² / fs', 'Énergie par seconde de signal.'],
  ['LineLength', 'Σ |x(t+1) − x(t)|', 'Longueur de la courbe : grande quand le signal est ample et pointu.'],
  ['HjorthMobility', '√(var(dx) / var(x))', 'Fréquence moyenne approchée.'],
  ['HjorthComplexity', 'Mobilité de la dérivée / mobilité', 'Écart à une sinusoïde pure (le signal pointe-onde est moins sinusoïdal).'],
  ['ZeroCross', 'Passages par zéro par seconde', 'Reflète la fréquence dominante : un rythme à 3 Hz donne environ 6 passages par seconde.'],
  ['Skew / Kurtosis', 'Asymétrie et aplatissement', 'Détectent les distributions « pointues » (pointes).'],
], [1900, 3200, 4538]));
c.push(gap());
c.push(H2('5.2 Features fréquentielles (14)'));
c.push(P('Le signal de la fenêtre est centré, multiplié par une **fenêtre de Hann** (réduit les fuites spectrales), puis transformé par FFT. On en déduit une densité spectrale de puissance.'));
c.push(table(['Feature', 'Définition', 'Intérêt'], [
  ['Power_Delta', 'Puissance 0,5-4 Hz', 'Bande du sujet.'],
  ['Power_Theta', 'Puissance 4-8 Hz', 'Bande du sujet.'],
  ['Power_Alpha', 'Puissance 8-13 Hz', 'Bande du sujet.'],
  ['Power_Beta', 'Puissance 13-30 Hz', 'Bande du sujet.'],
  ['Power_Hz3', 'Puissance 2,5-4,5 Hz', '**Bande ajoutée** : l\'énergie des pointes-ondes d\'absence se concentre autour de 3 Hz.'],
  ['Rel_* (5 features)', 'Puissance de bande / puissance totale 0,5-40 Hz', 'Indépendante de l\'amplitude : robuste entre patients.'],
  ['SpecEntropy', 'Entropie du spectre normalisée', 'Faible quand l\'énergie est concentrée sur une fréquence (rythme de crise), élevée pour un fond désorganisé.'],
  ['DomFreq', 'Fréquence du pic spectral', 'Environ 3 Hz pendant une crise.'],
  ['SEF95', 'Fréquence sous laquelle se trouvent 95 % de la puissance', 'Mesure l\'étalement spectral.'],
  ['Ratio_Theta_Alpha', 'Power_Theta / Power_Alpha', 'Rapport classique en EEG.'],
], [1900, 3200, 4538]));
c.push(gap());
c.push(H2('5.3 Pourquoi agréger par moyenne et maximum sur les canaux'));
c.push(B('Les absences sont des crises **généralisées** : tous les canaux changent ensemble. La **moyenne** capture ce comportement global.'));
c.push(B('Le **maximum** détecte un canal qui réagit fortement, utile quand la crise est plus marquée sur une région (et utile aussi pour repérer des artefacts locaux).'));
c.push(B('Les features restent identiques quel que soit le nombre de canaux (19 ou 8). Garder les 532 features (19 × 28) aurait aussi aggravé le surapprentissage avec seulement 1 168 fenêtres de crise.'));

// 6
c.push(H1('6. Sélection des meilleures features'));
c.push(P('Le fichier `src/selection.py` combine **quatre critères** puis écarte les redondances.'));
c.push(N('**Information mutuelle** avec la classe (capte aussi les relations non linéaires).'));
c.push(N('**Importance Random Forest** (contribution aux séparations des arbres).'));
c.push(N('**|d| de Cohen** : écart entre les moyennes des deux classes, en écarts-types (taille d\'effet).'));
c.push(N('**F-ANOVA** : rapport variance inter-classes / intra-classes.'));
c.push(P('Chaque critère classe les 56 features ; le **score** est le rang moyen. Pour éviter que les négatifs (32 038) écrasent les positifs (1 168), le calcul utilise toutes les fenêtres de crise et 12 000 fenêtres hors crise tirées au hasard.'));
c.push(P('Ensuite, une **sélection gloutonne** parcourt les features par score décroissant et écarte toute feature dont la corrélation absolue avec une feature déjà retenue dépasse **0,9**. On s\'arrête à K = 15. Cela évite de garder à la fois Std, Variance, RMS et Energy, qui mesurent la même chose.'));
c.push(H2('6.1 Les 15 features retenues'));
c.push(table(['Famille', 'Features retenues', 'Lecture'], [
  ['Bande 3 Hz', 'Rel_Hz3_mean, Rel_Hz3_max, Power_Hz3_mean', 'Confirme la signature pointe-onde à 3 Hz : la bande ajoutée est parmi les plus discriminantes.'],
  ['Amplitude', 'Std_mean, Variance_mean, Min_max, Max_max', 'Les crises sont plus amples.'],
  ['Forme / rythme', 'LineLength_mean, LineLength_max, ZeroCross_mean, ZeroCross_max, HjorthComplexity_max, DomFreq_mean', 'Le signal devient pointu et rythmique.'],
  ['Autres bandes', 'Rel_Beta_max, Power_Beta_mean', 'Contraste avec l\'activité de fond.'],
], [1700, 4200, 3738]));
c.push(gap());
c.push(note('Limite : le classement est calculé sur tous les patients avant l\'évaluation leave-one-patient-out. Cela introduit une légère fuite, identique pour les 4 modèles, donc elle ne fausse pas leur comparaison relative mais rend les scores un peu optimistes.'));
c.push(P('Dans l\'application (page « 3 · Meilleures features »), K et le seuil de corrélation sont réglables, et l\'on voit l\'évolution des features dans le temps chez le patient choisi, leur distribution par classe et leur matrice de corrélation.'));

// 7
c.push(H1('7. Les quatre modèles'));
c.push(P('Les quatre modèles sont des `Pipeline` scikit-learn : `StandardScaler` (appris uniquement sur l\'ensemble d\'entraînement) puis le classifieur. La mise à l\'échelle est indispensable pour le KNN et le SVM, inoffensive pour les arbres. Les trois modèles qui le permettent utilisent `class_weight="balanced"`.'));
c.push(H2('7.1 Decision Tree'));
c.push(B('**Principe** : une suite de règles « feature > seuil » apprises en découpant les données pour réduire l\'impureté.'));
c.push(B('**Réglages** : `max_depth=8`, `min_samples_leaf=5`, classes pondérées. La profondeur limitée évite d\'apprendre le bruit.'));
c.push(B('**Pourquoi** : référence simple et lisible (on peut lire les règles).'));
c.push(B('**Limites** : instable (un petit changement des données modifie l\'arbre), tendance au surapprentissage.'));
c.push(B('**Résultat** : Recall élevé (0,80) mais précision faible (0,33), donc beaucoup de fausses alarmes. C\'est le plus faible en PR-AUC (0,50).'));
c.push(H2('7.2 Random Forest'));
c.push(B('**Principe** : 200 arbres entraînés chacun sur un échantillon aléatoire des données et un sous-ensemble aléatoire de features, dont on moyenne les votes.'));
c.push(B('**Réglages** : `n_estimators=200`, `min_samples_leaf=2`, classes pondérées.'));
c.push(B('**Pourquoi** : la moyenne de nombreux arbres réduit fortement la variance du Decision Tree, gère bien les features corrélées et de natures différentes, et fournit une probabilité de crise exploitable pour la détection temporelle.'));
c.push(B('**Limites** : moins lisible qu\'un seul arbre ; échoue sur certains patients (F1 de 0,00 sur 230718A_A).'));
c.push(B('**Résultat** : meilleur équilibre précision/Recall, meilleure AUC (0,94) et meilleur PR-AUC (0,59).'));
c.push(H2('7.3 K-Nearest Neighbors (KNN)'));
c.push(B('**Principe** : une fenêtre reçoit la classe majoritaire de ses k voisins les plus proches dans l\'espace des features.'));
c.push(B('**Réglages** : `k=7`, vote pondéré par la distance. k impair limite les égalités ; la pondération donne plus de poids aux voisins très proches.'));
c.push(B('**Pourquoi** : méthode sans modèle explicite, sert de repère.'));
c.push(B('**Limites** : ne supporte pas la pondération de classes, donc penche vers la classe majoritaire (Recall de 0,50) ; sensible à l\'échelle des features et à leur nombre.'));
c.push(H2('7.4 Support Vector Machine (SVM)'));
c.push(B('**Principe** : cherche la frontière qui sépare au mieux les classes avec la plus grande marge ; le noyau gaussien (RBF) permet une frontière non linéaire.'));
c.push(B('**Réglages** : noyau RBF, `C=1`, `gamma="scale"`, classes pondérées. Pas d\'optimisation des hyperparamètres.'));
c.push(B('**Pourquoi** : bien adapté aux problèmes de dimension modérée avec frontière complexe.'));
c.push(B('**Limites** : le plus lent (environ 2 min pour 22 entraînements contre 1 min pour le RF) ; la probabilité (Platt) demande une validation croisée interne.'));
c.push(B('**Résultat** : le meilleur Recall (0,85) mais avec beaucoup de fausses alarmes (précision 0,42).'));

// 8
c.push(H1('8. Évaluation et résultats'));
c.push(H2('8.1 Protocoles utilisés'));
c.push(table(['Protocole', 'Description', 'Fiabilité'], [
  ['Split stratifié 80/20', 'Séparation aléatoire des fenêtres en conservant la proportion de crises (demandé par le sujet)', 'Optimiste : fenêtres voisines quasi identiques de part et d\'autre'],
  ['Split par patient 80/20', 'Aucun patient n\'est à la fois en train et en test', 'Réaliste, mais un seul tirage (5 patients de test)'],
  ['Leave-one-patient-out', '22 entraînements ; chaque patient est testé par un modèle qui ne l\'a jamais vu', '**La plus fiable**, utilisée pour comparer les modèles'],
], [2300, 4300, 3038]));
c.push(gap());
c.push(H2('8.2 Pourquoi privilégier le Recall et le F1-score'));
c.push(B('**Accuracy** : trompeuse (96,5 % pour « jamais de crise »).'));
c.push(B('**Recall** : part des crises détectées. Une crise manquée est le pire cas, donc c\'est la métrique prioritaire.'));
c.push(B('**Précision** : part des alertes qui sont de vraies crises ; elle mesure le coût des fausses alarmes.'));
c.push(B('**F1-score** : moyenne harmonique de précision et Recall ; empêche de « gagner » en prédisant crise partout.'));
c.push(B('**PR-AUC** (aire sous la courbe précision-rappel) : compare les modèles indépendamment du seuil, adaptée au déséquilibre.'));
c.push(H2('8.3 Résultats des splits uniques (56 features)'));
c.push(table(['Modèle', 'Split', 'Accuracy', 'Précision', 'Recall', 'F1'], [
  ['Decision Tree', 'Stratifié', '0,949', '0,397', '0,876', '0,546'],
  ['Random Forest', 'Stratifié', '0,984', '0,804', '0,718', '0,758'],
  ['KNN', 'Stratifié', '0,983', '0,817', '0,667', '0,734'],
  ['SVM', 'Stratifié', '0,965', '0,505', '0,872', '0,639'],
  ['Decision Tree', 'Par patient', '0,904', '0,230', '0,805', '0,357'],
  ['Random Forest', 'Par patient', '0,978', '0,788', '0,476', '0,594'],
  ['KNN', 'Par patient', '0,978', '0,810', '0,450', '0,579'],
  ['SVM', 'Par patient', '0,963', '0,467', '0,834', '0,599'],
], [1900, 1500, 1500, 1500, 1500, 1738]));
c.push(gap());
c.push(P('Le passage du split stratifié au split par patient fait chuter le F1 (par exemple 0,76 → 0,59 pour le Random Forest) : c\'est la preuve de la fuite d\'information du split aléatoire.'));
c.push(H2('8.4 Comparaison rigoureuse : leave-one-patient-out (15 features)'));
c.push(table(['Modèle', 'AUC', 'PR-AUC', 'Recall', 'Précision', 'F1', 'F1 moyen / patient'], [
  ['Random Forest', '0,942', '0,592', '0,575', '0,723', '0,641', '0,664'],
  ['SVM', '0,929', '0,553', '0,853', '0,423', '0,566', '0,646'],
  ['KNN', '0,875', '0,496', '0,499', '0,651', '0,565', '0,623'],
  ['Decision Tree', '0,847', '0,501', '0,796', '0,328', '0,464', '0,533'],
], [1900, 1000, 1100, 1100, 1200, 1100, 2238], { hl: [0] }));
c.push(gap());
c.push(P('**Intervalles de confiance à 95 %** (bootstrap sur les 19 patients ayant des crises, 300 tirages) :'));
c.push(table(['Modèle', 'PR-AUC', 'F1'], [
  ['Random Forest', '0,65 [0,49 - 0,80]', '0,66 [0,57 - 0,75]'],
  ['SVM', '0,59 [0,45 - 0,72]', '0,60 [0,49 - 0,71]'],
  ['KNN', '0,56 [0,44 - 0,69]', '0,59 [0,52 - 0,66]'],
  ['Decision Tree', '0,54 [0,38 - 0,71]', '0,51 [0,40 - 0,62]'],
], [3000, 3300, 3338]));
c.push(gap());
c.push(H2('8.5 Verdict'));
c.push(B('Le **Random Forest** est meilleur que le Decision Tree et le KNN de façon robuste : il gagne dans 100 % des tirages bootstrap en PR-AUC, et dans 97 % (contre Decision Tree) et 100 % (contre KNN) en F1.'));
c.push(B('Face au **SVM**, il gagne dans 89 % des tirages en PR-AUC et 80 % en F1, mais les intervalles se chevauchent : **avec 19 patients, la différence n\'est pas statistiquement prouvée**.'));
c.push(B('Selon le patient, le meilleur varie : RF gagnant pour 8 patients, SVM pour 7, Decision Tree pour 3, KNN pour 1.'));
c.push(B('Les modèles ne font pas le même compromis : le SVM détecte plus de crises mais crie plus souvent ; le RF est plus sélectif. **Recommandation : Random Forest comme modèle principal** ; le SVM si l\'on ne veut rater aucune crise.'));
c.push(note('Limites : hyperparamètres non optimisés (un réglage pourrait inverser le classement RF/SVM) ; seulement 19 patients avec crises ; la fuite légère de la sélection de features (section 6).'));

// 9
c.push(H1('9. Détection temporelle'));
c.push(P('Les prédictions par fenêtre sont transformées en intervalles de crise (`src/detection.py`). Les probabilités proviennent du leave-one-patient-out : le modèle n\'a jamais vu le patient testé.'));
c.push(H2('9.1 Algorithme'));
c.push(N('**Seuil** : une fenêtre est « crise » si sa probabilité dépasse le seuil (0,5 par défaut).'));
c.push(N('**Lissage** par filtre médian sur 3 fenêtres : supprime les détections isolées d\'une seule fenêtre (souvent des artefacts) et comble les trous d\'une fenêtre au milieu d\'une crise.'));
c.push(N('**Fusion** des détections consécutives : deux détections séparées de 2 s ou moins forment un seul intervalle (une crise peut présenter un creux bref).'));
c.push(N('**Durée minimale** de 2 s : correspond à la plus courte crise annotée.'));
c.push(N('Début de l\'intervalle = début de la première fenêtre ; fin = fin de la dernière fenêtre.'));
c.push(H2('9.2 Résultats sur les 22 patients (9,2 heures, 82 crises)'));
c.push(table(['Modèle / seuil', 'Crises détectées', 'Fausses alarmes', 'Fausses alarmes / heure', 'Erreur de début (médiane)'], [
  ['Random Forest, seuil 0,5', '74 / 82 (90 %)', '28', '3,0', '0 s'],
  ['Random Forest, seuil 0,3', '75 / 82 (91 %)', '50', '5,4', '0 s'],
  ['SVM, seuil par défaut', '82 / 82 (100 %)', '200', '21,7', '0 s'],
], [2600, 1900, 1500, 1800, 1838]));
c.push(gap());
c.push(B('Au niveau des **crises complètes**, la détection est bien meilleure qu\'au niveau des fenêtres : le lissage et la fusion corrigent beaucoup d\'erreurs isolées.'));
c.push(B('Le SVM ne rate aucune crise mais produit environ 7 fois plus de fausses alarmes que le Random Forest, ce qui illustre le compromis de la section 8.5.'));
c.push(B('Échecs principaux du Random Forest : 230718A_A (0 crise sur 7 détectée) et 220121B_D (2 sur 3). Je n\'ai pas vérifié la cause ; une hypothèse à tester dans la page « Fenêtre par fenêtre » est une morphologie de crise différente de celle des autres patients.'));
c.push(note('Limites de la mesure : une crise est comptée détectée si un intervalle prédit la chevauche (tolérance de 2 s). Un intervalle fusionnant plusieurs crises rapprochées compte pour toutes, ce qui est un peu optimiste. La résolution temporelle est limitée par le pas de 1 s.'));

// 10
c.push(H1('10. Analyse fenêtre par fenêtre (interface)'));
c.push(P('La page « 2 · Fenêtre par fenêtre » permet de parcourir un patient fenêtre par fenêtre (curseur, boutons Précédente/Suivante, « Prochaine alerte », « Aller à une crise »). Pour chaque fenêtre :'));
c.push(B('**Signal** : les canaux EEG prétraités, avec la fenêtre courante surlignée en jaune, 6 s de contexte de chaque côté, et les crises annotées en rouge.'));
c.push(B('**Probabilité du modèle** (leave-one-patient-out) et son évolution sur toute la durée avec le seuil et la position courante.'));
c.push(B('**Verdict** : vrai positif, vrai négatif, faux positif (fausse alarme) ou faux négatif (crise manquée), à partir de la classe réelle et de la probabilité.'));
c.push(B('**Meilleures features** de la fenêtre : écart à la normale en z-score, calculé par rapport à la moyenne et à l\'écart-type des fenêtres hors crise. Un tableau donne aussi la valeur de la fenêtre et les moyennes des deux classes.'));
c.push(B('**Spectre FFT** d\'un canal au choix et comparaison des puissances relatives par bande avec les moyennes hors crise et crise.'));
c.push(P('Cette page sert surtout à comprendre les erreurs : en sautant sur les faux négatifs et faux positifs, on voit si c\'est un artefact, une crise très courte ou un bord de crise.'));

// 11
c.push(H1('11. Limites et suite du projet'));
c.push(H2('11.1 Limites connues'));
c.push(B('Peu de patients (19 avec crises) : les comparaisons entre modèles restent peu tranchées.'));
c.push(B('Hyperparamètres non optimisés ; pas de validation croisée imbriquée.'));
c.push(B('Sélection des features faite sur tous les patients (légère fuite).'));
c.push(B('Normalisation sur l\'enregistrement entier, non applicable telle quelle en temps réel.'));
c.push(B('Pas de traitement spécifique des artefacts.'));
c.push(H2('11.2 Améliorations possibles'));
c.push(B('Régler les hyperparamètres (profondeur, nombre d\'arbres, C et gamma du SVM) en validation croisée par patient.'));
c.push(B('Ajouter des features inter-canaux (cohérence, synchronie) puisque les absences sont généralisées, et des features de contexte (fenêtres voisines).'));
c.push(B('Gérer le déséquilibre autrement (SMOTE, seuil optimisé) et tester un réseau convolutif 1D sur le signal brut.'));
c.push(H2('11.3 Prévention (étape suivante)'));
c.push(P('La colonne `Class3` (0 = inter-critique, 1 = pré-crise dans les 10 s avant le début, 2 = crise) et la page « 6 · Prévention (préparation) » préparent cette étape. Cette page aligne toutes les crises sur leur début et trace la moyenne des features avant le début : si une feature monte avant t = 0, elle porte un signal précurseur. **Je n\'ai pas encore analysé ces courbes ni entraîné de modèle de prévention** ; c\'est la prochaine étape.'));

// Annexe
c.push(H1('Annexe : fichiers du projet'));
c.push(table(['Fichier', 'Rôle'], [
  ['app.py', 'Application Streamlit patient par patient'],
  ['app_questions.py', 'Ancienne application organisée par questions du sujet'],
  ['src/convert_to_csv.py', 'Conversion sans perte des .mat et de l\'Excel en CSV, avec vérification exacte'],
  ['data_csv/', 'Données converties : 22 signaux, 22 fichiers de marqueurs, annotations, manifest'],
  ['src/data_loader.py', 'Lecture des CSV, correction des annotations'],
  ['src/preprocessing.py', 'Sélection des canaux, filtre, normalisation, étiquettes, fenêtrage'],
  ['src/features.py', 'Calcul des 28 features par canal'],
  ['src/dataset.py', 'Construction du dataset (1 ligne = 1 fenêtre)'],
  ['src/selection.py', 'Classement et sélection des features'],
  ['src/models.py', 'Les 4 modèles, splits, évaluation, leave-one-patient-out'],
  ['src/detection.py', 'Reconstruction des intervalles de crise'],
  ['cache/', 'Signaux et datasets mis en cache (régénérés automatiquement)'],
], [3000, 6638]));

const doc = new Document({
  creator: 'Projet Data Mining 3AINFO',
  title: 'Méthodologie détaillée - Détection de crises d\'absence EEG',
  styles: {
    default: { document: { run: { font: FONT, size: 21 } } },
    paragraphStyles: [
      { id: 'Heading1', name: 'Heading 1', basedOn: 'Normal', next: 'Normal', quickFormat: true, run: { size: 32, bold: true, font: FONT, color: '1F3864' }, paragraph: { spacing: { before: 320, after: 140 }, outlineLevel: 0 } },
      { id: 'Heading2', name: 'Heading 2', basedOn: 'Normal', next: 'Normal', quickFormat: true, run: { size: 25, bold: true, font: FONT, color: '2E5C8A' }, paragraph: { spacing: { before: 220, after: 100 }, outlineLevel: 1 } },
    ],
  },
  numbering: { config: [
    { reference: 'bul', levels: [{ level: 0, format: LevelFormat.BULLET, text: '•', alignment: AlignmentType.LEFT, style: { paragraph: { indent: { left: 540, hanging: 270 } } } }] },
    { reference: 'num', levels: [{ level: 0, format: LevelFormat.DECIMAL, text: '%1.', alignment: AlignmentType.LEFT, style: { paragraph: { indent: { left: 540, hanging: 300 } } } }] },
  ] },
  sections: [{
    properties: { page: { size: { width: 11906, height: 16838 }, margin: { top: 1134, bottom: 1134, left: 1134, right: 1134 } } },
    footers: { default: new Footer({ children: [new Paragraph({ alignment: AlignmentType.CENTER, children: [new TextRun({ children: ['Page ', PageNumber.CURRENT], size: 18, font: FONT, color: '888888' })] })] }) },
    children: c,
  }],
});
Packer.toBuffer(doc).then(b => {
  for (const name of ['Methodologie_detaillee_EEG.docx', 'Methodologie_detaillee_EEG_CSV.docx']) {
    try { fs.writeFileSync(name, b); console.log('ecrit: ' + name); return; } catch (e) { console.log('verrouille: ' + name); }
  }
});
