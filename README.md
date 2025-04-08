# Projet de Prédiction Immobilière

## Membres du Groupe
- **Philibert Benjamin**
- **Quillaume Quentin**
- **Antier Quillaume**
- **Lechat Noé**

---

## Informations sur le Dataset
- **Taille du dataset** : `(21613, 21)`
- **Cible** : Le prix (`price`)

---

## Hypothèse
Nous supposons que **la surface habitable** (`sqft_living`) et **la note** (`grade`) sont les facteurs les plus déterminants dans la prédiction du prix d'une maison. Cette hypothèse repose sur les points suivants :

1. **Grande variation des prix** : Les prix varient de **75k$** à **7.7M$**, ce qui suggère une forte influence de caractéristiques spécifiques.
2. **Surface habitable** : Elle est une mesure directe de l'espace utilisable, ce qui est un facteur clé pour les acheteurs.
3. **Note (grade)** : Elle reflète la qualité globale de la construction et des matériaux utilisés.

---

## Objectif
L'objectif de ce projet est de construire un modèle de machine learning capable de prédire le prix d'une maison en fonction des caractéristiques disponibles dans le dataset.

---

## Méthodologie
1. **Exploration des données** :
   - Analyse des distributions des variables.
   - Identification des relations entre les caractéristiques et la cible (`price`).

2. **Préparation des données** :
   - Nettoyage des données (suppression des valeurs aberrantes, encodage des variables catégorielles, etc.).
   - Division des données en ensembles d'entraînement et de test.

3. **Modélisation** :
   - Utilisation de différents modèles de régression, notamment :
     - Régression linéaire.
     - Arbre de décision.
     - Modèles optimisés avec recherche d'hyperparamètres.

4. **Évaluation** :
   - Utilisation de métriques telles que :
     - **RMSE** (Root Mean Squared Error).
     - **MAE** (Mean Absolute Error).
     - **R²** (Coefficient de détermination).

---

## Résultats attendus
- Construire un modèle performant avec une erreur minimale sur l'ensemble de test.
- Comparer les performances des différents modèles pour sélectionner le plus adapté.

---

## Conclusion
Ce projet vise à fournir une solution robuste pour prédire le prix des maisons en utilisant des techniques de machine learning. Les résultats obtenus permettront de mieux comprendre les facteurs influençant les prix immobiliers et d'améliorer les prédictions pour des applications réelles.

### Exemple de commande CURL

```bash
curl -X 'POST' \
  'http://127.0.0.1:8000/predict' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
  "id": 123456789,
  "date": "2025-04-08",
  "price": 450000,
  "bedrooms": 3,
  "bathrooms": 2.5,
  "sqft_living": 2000,
  "sqft_lot": 5000,
  "floors": 2,
  "waterfront": 0,
  "view": 1,
  "condition": 3,
  "grade": 7,
  "sqft_above": 1800,
  "sqft_basement": 200,
  "yr_built": 1995,
  "yr_renovated": 2010,
  "zipcode": 98178,
  "lat": 47.5112,
  "long": -122.257,
  "sqft_living15": 1500,
  "sqft_lot15": 4000
}'
```

Cette commande envoie une requête POST à une API de prédiction avec des caractéristiques d'une maison pour obtenir une estimation du prix.