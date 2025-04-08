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

## Prérequis
Avant de lancer le projet, assurez-vous d'avoir les éléments suivants installés sur votre machine :
- **Python 3.10 ou supérieur**
- **pip** (gestionnaire de paquets Python)
- **Virtualenv** (optionnel mais recommandé)

### Installation des dépendances
1. Clonez le dépôt du projet :
   ```bash
   git clone https://github.com/benphihe/immobilier.git
   cd immobilier
   ```

2. Créez un environnement virtuel (optionnel mais recommandé) :
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   ```

3. Installez les dépendances nécessaires :
   ```bash
   pip install -r requirements.txt
   ```

---

## Comment lancer le projet
1. Assurez-vous que le fichier `kc_house_data.csv` est présent dans le dossier `data/` :
   ```
   immobilier/
   ├── app.py
   ├── main.py
   ├── data/
   │   └── kc_house_data.csv
   ├── src/
   │   ├── processor_imo.py
   │   ├── regressionTree.py
   │   └── ...
   ├── tests/
   └── README.md
   ```

2. Lancez le serveur FastAPI :
   ```bash
   uvicorn app:app --reload
   ```

3. Accédez à la documentation interactive de l'API :
   - Ouvrez votre navigateur et allez sur [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs).

---

## Méthodologie
1. **Exploration des données** :
   - Analyse des distributions des variables.
   - Identification des relations entre les caractéristiques et la cible (`price`).

2. **Préparation des données** :
   - Nettoyage des données (suppression des valeurs aberrantes uniquement si le dataset contient plus de 100 lignes).
   - Ajout de nouvelles colonnes dérivées :
     - `price_per_sqft_living` : Prix par surface habitable.
     - `AsBeenRenovated` : Indique si la maison a été rénovée.
   - Suppression de colonnes inutiles comme `date`.
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

## Fonctionnalités de l'API
### Endpoint `/predict`
Cet endpoint permet de prédire le prix d'une maison en fonction des caractéristiques fournies.

#### Exemple de données d'entrée :
```json
{
  "id": 987654321,
  "date": "2025-04-08",
  "price": 750000,
  "bedrooms": 4,
  "bathrooms": 3.0,
  "sqft_living": 3000,
  "sqft_lot": 6000,
  "floors": 2,
  "waterfront": 1,
  "view": 2,
  "condition": 4,
  "grade": 8,
  "sqft_above": 2500,
  "sqft_basement": 500,
  "yr_built": 2010,
  "yr_renovated": 2020,
  "zipcode": 98052,
  "lat": 47.6396,
  "long": -122.128,
  "sqft_living15": 2800,
  "sqft_lot15": 5000
}
```

#### Exemple de réponse :
```json
{
  "predicted_price": 800000
}
```

---

## Résultats attendus
- Construire un modèle performant avec une erreur minimale sur l'ensemble de test.
- Comparer les performances des différents modèles pour sélectionner le plus adapté.

---

## Conclusion
Ce projet vise à fournir une solution robuste pour prédire le prix des maisons en utilisant des techniques de machine learning. Les résultats obtenus permettront de mieux comprendre les facteurs influençant les prix immobiliers et d'améliorer les prédictions pour des applications réelles.

---

## Exemple de commande CURL

```bash
curl -X 'POST' \
'http://127.0.0.1:8000/predict' \
-H 'accept: application/json' \
-H 'Content-Type: application/json' \
-d '{
"id": 987654321,
"date": "2025-04-08",
"price": 750000,
"bedrooms": 4,
"bathrooms": 3.0,
"sqft_living": 3000,
"sqft_lot": 6000,
"floors": 2,
"waterfront": 1,
"view": 2,
"condition": 4,
"grade": 8,
"sqft_above": 2500,
"sqft_basement": 500,
"yr_built": 2010,
"yr_renovated": 2020,
"zipcode": 98052,
"lat": 47.6396,
"long": -122.128,
"sqft_living15": 2800,
"sqft_lot15": 5000
}'
```

Cette commande envoie une requête POST à l'API pour prédire le prix d'une maison en fonction des caractéristiques fournies.