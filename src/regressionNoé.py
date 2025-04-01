from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error
import pandas as pd
import matplotlib.pyplot as plt
from .model import load_and_prepare_data

def main():
    # Étape 1 : Chargement et préparation des données
    df = load_and_prepare_data()
    X = df.drop('price', axis=1)
    y = df['price']

    # Étape 2 : Division des données en ensembles d'entraînement et de test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Étape 3 : Entraînement du modèle
    reg = LinearRegression()
    reg.fit(X_train, y_train)

    # Étape 4 : Prédictions
    y_pred = reg.predict(X_test)

    # Étape 5 : Calcul des métriques
    mse = mean_squared_error(y_test, y_pred)
    rmse = mse ** 0.5
    mae = mean_absolute_error(y_test, y_pred)
    print("RMSE:", rmse, "MAE:", mae)

    # Étape 6 : Visualisation des prédictions
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, y_pred, alpha=0.5, label="Prédictions")
    plt.xlabel('True Values')
    plt.ylabel('Predictions')

    # Ajuster les limites des axes
    plt.xlim(0, 2.00e6)
    plt.ylim(0, 2.00e6)

    # Ajouter une ligne de référence y = x
    plt.plot([0, 2.00e6], [0, 2.00e6], color='red', linestyle='--', label="Ligne parfaite")
    plt.legend()
    plt.show()

# Point d'entrée du script
if __name__ == "__main__":
    main()