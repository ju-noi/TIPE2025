# TIPE2025
**Sujet :** Réseaux de neurones artificiels et optimisation de l’apprentissage

**Problématique :** *Comment fonctionne l'algorithme de descente de gradient et en quoi permet-il l'apprentissage supervisé ?*

---

## 📌 Présentation du Projet

Ce dépôt GitHub regroupe les codes, simulations et applications développés dans le cadre de mon TIPE. L'objectif principal est d'implémenter **de zéro (from scratch)** l'algorithme de la **descente de gradient** en Python. Aucune bibliothèque de machine learning pré-intégrées (type Scikit-Learn) n'a été utilisée.

L'apprentissage supervisé y est formalisé comme un problème d'optimisation : il s'agit d'ajuster itérativement les paramètres internes d'un modèle (poids et biais) afin de minimiser une fonction de coût (telle que l'erreur quadratique moyenne) qui quantifie l'écart entre les valeurs prédites et les observations de référence de l'ensemble d'entraînement. Pour résoudre ce problème, la descente de gradient exploite le calcul différentiel en évaluant le gradient de la fonction de coût — c'est-à-dire le vecteur des dérivées partielles par rapport à chaque paramètre. En actualisant les poids dans la direction opposée à ce gradient, pondérée par un taux d'apprentissage, l'algorithme conduit le modèle à converger vers un état optimal. Cette démarche est mise en œuvre numériquement pour la régression linéaire simple et multiple, d'abord sur des données synthétiques contrôlées, puis validée sur des cas d'application réels (prédiction salariale).

---

## 📂 Structure du Dépôt

Le projet s'organise autour de plusieurs scripts Python autonomes :

* **`regressionLineaire.py`** : 
  * Implémentation de la **régression linéaire simple** ($y = ax + b$).
  * Simulation d'un nuage de points d'apprentissage.
  * Mise en œuvre de l'algorithme de descente de gradient pour l'optimisation des poids.
  * Étude comparative des taux d'apprentissage (taux fixe vs taux itératif/décroissant) et calcul du coefficient de Pearson.
  * **Application concrète :** Prévision du salaire en fonction de l'expérience professionnelle à partir du fichier `Salary_dataset.csv`.

* **`regressionMultiple.py`** :
  * Implémentation de la **régression linéaire multiple** en dimension 3 ($z = ax + by + c$).
  * Visualisation 3D du plan de régression.
  * Normalisation des données pour stabiliser l'apprentissage lorsque les variables ont des échelles différentes.
 

* **`test_regressionMultiple.py`** :
   * Application concrète : Analyse multi-variable sur le jeu de données médicales `diabetes.csv` (relation entre l'âge, l'IMC/BMI et le taux de glucose) --> non aboutis

---
