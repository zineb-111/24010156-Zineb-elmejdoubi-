# ZINEB EL MEJDOUBI
**Numéro d'étudiant** : 24010156
**Classe** : CAC2

Rapport d'analyse consolidé : Modèle de prédiction du diabète

1. Introduction

   Ce rapport détaille une analyse complète des facteurs influençant la prédiction du diabète, basée sur le jeu de données diabetes_prediction_dataset.csv. L'objectif principal était de construire et d'évaluer un modèle de classification pour prédire la présence de diabète chez les individus, en identifiant les prédicteurs clés et en évaluant l'efficacité du modèle.

2. Méthodologie

2.1. Nettoyage et Préparation des Données
   
Chargement des Données: Le jeu de données initial a été chargé depuis KaggleHub.
Gestion des Valeurs Manquantes: Aucune valeur manquante n'a été identifiée dans l'ensemble de données, assurant une intégrité des données complète.
Gestion des Doublons: Un total de 3854 lignes en double ont été identifiées et supprimées, passant le jeu de données de 100 000 à 96 146 entrées, améliorant ainsi la qualité des données pour l'analyse et la modélisation.
Encodage des Variables Catégorielles: Les variables catégorielles nominales gender et smoking_history ont été transformées en variables numériques via One-Hot Encoding pour les rendre compatibles avec les algorithmes d'apprentissage automatique.
Séparation des Caractéristiques et de la Cible: Les caractéristiques (X) ont été séparées de la variable cible (y), diabetes. X avait une forme de (96146, 13) et y (96146,).

2.2. Exploration des Données (EDA)
   
Déséquilibre de la Variable Cible: L'analyse a révélé un déséquilibre significatif dans la variable cible diabetes, avec environ 91,5% des individus étant non diabétiques et 8,5% étant diabétiques. Cette information est cruciale pour l'évaluation du modèle.
Analyse des Distributions: Des distributions ont été visualisées pour les caractéristiques clés. Il a été observé que les individus diabétiques ont tendance à être plus âgés et à avoir des valeurs plus élevées pour l'IMC (bmi), le niveau d'HbA1c (HbA1c_level) et le niveau de glucose sanguin (blood_glucose_level). Les niveaux d'HbA1c et de glucose sanguin sont apparus comme de très forts indicateurs.
Facteurs de Risque Catégoriels/Binaires: L'hypertension et les maladies cardiaques ont montré une association plus élevée avec la présence de diabète. L'historique de tabagisme (smoking_history) a également montré des variations, avec des fumeurs actuels ou anciens présentant potentiellement un risque plus élevé.
Analyse de Corrélation: Une carte thermique de corrélation a montré des corrélations positives notables entre la variable cible diabetes et les caractéristiques numériques comme HbA1c_level, blood_glucose_level, age et bmi.

2.3. Stratégie de Modélisation
Division des Données: Le jeu de données a été divisé en ensembles d'entraînement (80%) et de test (20%) en utilisant train_test_split avec random_state=42 pour assurer la reproductibilité. Les ensembles d'entraînement et de test pour les caractéristiques (X) et la cible (y) étaient respectivement de (76916, 13) et (19230, 13).
Algorithme: Un modèle RandomForestClassifier a été choisi et entraîné avec n_estimators=100 et random_state=42 sur les données d'entraînement.

3. Résultats & Discussion
   
3.1. Modèle de Classification (RandomForestClassifier)
Le modèle RandomForestClassifier a été évalué sur l'ensemble de test (X_test, y_test).

Précision Globale (Accuracy): Le modèle a atteint une précision globale de 96,80 % sur l'ensemble de test.
Rapport de Classification:
Classe 0 (Pas de Diabète):
Précision : 0,97 (97% des prédictions "Pas de Diabète" étaient correctes).
Rappel : 1,00 (Le modèle a correctement identifié 100% de tous les cas réels "Pas de Diabète").
F1-Score : 0,98 (F1-score élevé, indiquant d'excellentes performances pour la classe majoritaire).
Classe 1 (Diabète):
Précision : 0,94 (94% des prédictions "Diabète" étaient correctes).
Rappel : 0,69 (Le modèle a correctement identifié 69% de tous les cas réels "Diabète").
F1-Score : 0,79 (Un bon F1-score pour la classe minoritaire, mais le rappel plus faible suggère une marge d'amélioration).
Matrice de Confusion:
Vrais Négatifs (TN) : 17509 (Correctement prédit 'Pas de Diabète')
Faux Positifs (FP) : 67 (Incorrectement prédit 'Diabète' alors que c'était 'Pas de Diabète')
Faux Négatifs (FN) : 535 (Incorrectement prédit 'Pas de Diabète' alors que c'était 'Diabète')
Vrais Positifs (TP) : 1119 (Correctement prédit 'Diabète')

4. Conclusion
   
Résumé des Facteurs Influencant la Prédiction du Diabète
L'analyse et la modélisation ont convergé vers des facteurs clés qui influencent significativement la prédiction du diabète :

Niveaux d'HbA1c et de Glucose Sanguin: Ce sont les prédicteurs les plus forts, avec des valeurs significativement plus élevées chez les individus diabétiques.
Âge et IMC: Ces facteurs présentent une corrélation positive avec le diabète, les personnes plus âgées et celles ayant un IMC plus élevé étant plus sujettes.
Antécédents Médicaux: L'hypertension et les maladies cardiaques sont des facteurs de risque importants. L'historique de tabagisme a également un impact.
Limites du Modèle Développé
Déséquilibre de Classe: La principale limitation est le rappel modéré pour la classe minoritaire (Diabète, 0.69). Cela signifie que le modèle a manqué un nombre significatif de cas réels de diabète (535 Faux Négatifs), ce qui est critique dans un contexte médical où la détection précoce est essentielle.
Interprétabilité: Bien que le RandomForestClassifier soit performant, il peut être considéré comme une "boîte noire" par rapport à des modèles plus simples, ce qui rend l'interprétation directe des relations complexes plus difficile.
Propositions Concrètes de Pistes d'Amélioration Futures
Rééquilibrage des Classes: Pour améliorer la détection des cas de diabète, il est crucial d'implémenter des techniques de gestion du déséquilibre des classes, telles que le suréchantillonnage (ex. SMOTE) de la classe minoritaire ou le sous-échantillonnage de la classe majoritaire.
Optimisation Avancée du Modèle: Explorer des réglages d'hyperparamètres plus poussés pour le RandomForestClassifier ou tester d'autres algorithmes d'apprentissage automatique, tels que XGBoost, LightGBM ou les réseaux de neurones, qui sont souvent très efficaces pour les tâches de classification.
Ingénierie de Caractéristiques: La création de nouvelles caractéristiques basées sur les connaissances médicales ou les interactions entre les variables existantes pourrait apporter des informations supplémentaires et améliorer la performance du modèle.
Ajustement du Seuil de Décision: Dans le cas de la prédiction du diabète, où les faux négatifs sont plus coûteux que les faux positifs, l'ajustement du seuil de classification pour privilégier le rappel de la classe 'Diabète' pourrait être envisagé.
Ce rapport fournit une compréhension solide des facteurs de risque du diabète et de la performance d'un modèle de prédiction initial, tout en soulignant les domaines clés pour les améliorations futures afin de rendre le modèle plus robuste et cliniquement utile.

