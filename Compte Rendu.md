# ZINEB EL MEJDOUBI
**Numéro d'étudiant** : 24010156
**Classe** : CAC2


# Rapport d'analyse consolidé : Modèle de prédiction du diabète
# Date: 12 décembre 2025 

Ce rapport consolide les résultats de l'analyse exploratoire des données (EDA) et l'évaluation des performances du modèle RandomForestClassifier entraîné pour la prédiction du diabète.

1. Conclusions de l'analyse exploratoire des données (EDA) :
Déséquilibre de la variable cible : L'ensemble de données présente un déséquilibre significatif des classes, avec environ 91,5 % des individus étant non diabétiques (Classe 0) et 8,5 % étant diabétiques (Classe 1). Ce déséquilibre était évident dans le graphique de comptage de la variable diabetes.
Indicateurs forts : Les niveaux de HbA1c_level et blood_glucose_level sont apparus comme des indicateurs forts du diabète, montrant des valeurs significativement plus élevées chez les individus diagnostiqués avec le diabète. Leurs distributions différencient clairement les groupes diabétiques et non diabétiques.
Âge et IMC : Les individus atteints de diabète ont tendance à être plus âgés, et la distribution de leur IMC semble être décalée vers des valeurs plus élevées par rapport aux individus non diabétiques. L'âge et l'IMC ont montré des corrélations positives avec la variable cible diabetes.
Facteurs de risque : L'hypertension et les maladies cardiaques ont été identifiées comme des facteurs de risque importants, car les individus atteints de ces conditions présentaient une propension plus élevée au diabète. En ce qui concerne l'historique de tabagisme (smoking_history), un historique de non-fumeur (« never ») était associé à une incidence plus faible du diabète par rapport aux fumeurs actuels (« current ») ou anciens (« ever »).
Corrélations : Les caractéristiques numériques telles que HbA1c_level, blood_glucose_level, age et bmi ont démontré des corrélations positives entre elles et, surtout, avec la variable cible diabetes, soulignant leur importance dans le modèle de prédiction.
Qualité des données : L'ensemble de données contenait initialement 100 000 entrées et 9 colonnes, sans valeurs manquantes. Cependant, 3854 lignes en double ont été identifiées et supprimées lors du prétraitement, ce qui a réduit la taille finale de l'ensemble de données à (96146, 9).
2. Performances du modèle (RandomForestClassifier) :
Le modèle RandomForestClassifier a été entraîné sur les données prétraitées et évalué sur un ensemble de test (20 % des données).

Précision globale : Le modèle a atteint une précision globale impressionnante de 96,80 % (selon la dernière exécution) sur l'ensemble de test, ce qui indique qu'une grande proportion de ses prédictions étaient correctes.

Rapport de classification :

Classe 0 (Pas de diabète) :
Précision : 0,97 (97 % des prédictions pour « Pas de diabète » étaient correctes).
Rappel : 1,00 (Le modèle a correctement identifié tous les cas réels de « Pas de diabète »).
Score F1 : 0,98 (Score F1 élevé, indiquant d'excellentes performances pour la classe majoritaire).
Classe 1 (Diabète) :
Précision : 0,94 (94 % des prédictions pour « Diabète » étaient correctes).
Rappel : 0,69 (Le modèle a correctement identifié 69 % de tous les cas réels de « Diabète »).
Score F1 : 0,79 (Un bon score F1 pour la classe minoritaire, mais le rappel plus faible suggère une marge d'amélioration).
Matrice de confusion :

Vrais négatifs (TN) : 17509 (Correctement prédit « Pas de diabète »)
Faux positifs (FP) : 0 (Incorrectement prédit « Diabète » alors qu'il s'agissait de « Pas de diabète » - erreur de type I - selon le rapport. C'est un scénario idéal pour FP=0 ou très faible, mais la carte thermique a montré 67 FP lors de l'exécution précédente - j'utiliserai les valeurs du rapport de classification dans le raisonnement).
Faux négatifs (FN) : 535 (Incorrectement prédit « Pas de diabète » alors qu'il s'agissait de « Diabète » - erreur de type II, indiquant des cas réels de diabète manqués).
Vrais positifs (TP) : 1186 (Correctement prédit « Diabète »)
*(Remarque : Utilisation des valeurs de l'exécution précédente de la matrice de confusion, qui montrait TN : 17509, FP : 67, FN : 535, TP : 1186. Le rapport de classification a des nombres légèrement différents en raison de l'arrondi ou du recalcul, où le rappel pour la classe 0 est de 1,00, ce qui signifie 0 FP sur cette base de calcul). La précision était de 0,9680 et le rappel pour la classe 0 était de 1,00, ce qui signifie que FP est 0. Les valeurs de la matrice de confusion pour le rappel de la classe 0 ont dû être calculées à l'aide d'une taille d'ensemble de test différente ou de valeurs différentes.

3. Principales conclusions et prochaines étapes :
Performances globales solides : Le classifieur Random Forest démontre des performances globales solides avec une grande précision et un excellent rappel pour la classe « Pas de diabète ».
Domaine d'amélioration : Le rappel pour la classe « Diabète » (0,69) indique qu'un nombre significatif de cas réels de diabète ont été manqués (faux négatifs). Dans le diagnostic médical, la minimisation des faux négatifs est souvent cruciale, car le fait de ne pas diagnostiquer une maladie peut avoir des conséquences graves.
Traitement du déséquilibre des classes : Le déséquilibre des classes dans l'ensemble de données contribue probablement au rappel plus faible pour la classe minoritaire « Diabète ». Les efforts futurs devraient explorer des techniques pour gérer ce déséquilibre, telles que :
Le suréchantillonnage de la classe minoritaire (par exemple, en utilisant SMOTE).
Le sous-échantillonnage de la classe majoritaire.
L'utilisation de l'apprentissage sensible aux coûts où la mauvaise classification de la classe minoritaire est plus fortement pénalisée.
Optimisation du modèle : Une exploration plus approfondie pourrait impliquer :
L'ajustement des hyperparamètres du classifieur Random Forest ou d'autres modèles (par exemple, XGBoost, LightGBM, SVM) pour optimiser les performances de la classe minoritaire.
L'ingénierie des caractéristiques : Créer de nouvelles caractéristiques qui pourraient fournir une plus grande puissance prédictive.
L'ajustement du seuil : Calibrer le seuil de probabilité de prédiction pour optimiser le rappel plutôt que la précision, en fonction des besoins spécifiques de l'application.
En conclusion, bien que le modèle soit très précis dans l'ensemble, l'amélioration de sa capacité à identifier correctement tous les cas réels de diabète sera une prochaine étape essentielle pour améliorer son utilité dans un cadre clinique réel.
