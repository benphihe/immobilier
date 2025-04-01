import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, median_absolute_error, r2_score, explained_variance_score
from sklearn.tree import DecisionTreeRegressor
import numpy as np
from .model import load_and_prepare_data

def main():
    # Étape 1 : Chargement et préparation des données
    df = load_and_prepare_data()
    X = df.drop("price", axis=1)
    y = df["price"]

    # Étape 2 : Division des données en ensembles d'entraînement et de test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Étape 3 : Entraînement initial du modèle
    tree = DecisionTreeRegressor(random_state=42)
    tree.fit(X_train, y_train)

    # Étape 4 : Visualisation des importances des caractéristiques
    importances = pd.Series(tree.feature_importances_, index=X_train.columns)
    importances_sorted = importances.sort_values()
    plt.barh(importances_sorted.index, importances_sorted)
    plt.title("Feature Importances")
    plt.xlabel("Importance")
    plt.ylabel("Features")
    plt.savefig('feature_importances.png')
    plt.show()

    # Étape 5 : Évaluation initiale du modèle
    y_train_pred = tree.predict(X_train)
    y_test_pred = tree.predict(X_test)

    train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
    test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
    train_mae = mean_absolute_error(y_train, y_train_pred)
    test_mae = mean_absolute_error(y_test, y_test_pred)
    train_medae = median_absolute_error(y_train, y_train_pred)
    test_medae = median_absolute_error(y_test, y_test_pred)
    train_r2 = r2_score(y_train, y_train_pred)
    test_r2 = r2_score(y_test, y_test_pred)
    train_evs = explained_variance_score(y_train, y_train_pred)
    test_evs = explained_variance_score(y_test, y_test_pred)

    results = pd.DataFrame({
        'Métrique': ['RMSE', 'MAE', 'MedAE', 'R2', 'EVS'],
        'Train': [train_rmse, train_mae, train_medae, train_r2, train_evs],
        'Test': [test_rmse, test_mae, test_medae, test_r2, test_evs]
    })
    print(results)

    # Étape 6 : Recherche des hyperparamètres optimaux
    param_grid = {'max_depth': np.arange(1, 21)}
    grid_tree = GridSearchCV(tree, param_grid, cv=5, scoring='neg_mean_squared_error')
    grid_tree.fit(X_train, y_train)

    print("Profondeur optimale:", grid_tree.best_params_['max_depth'])

    # Visualisation de la courbe de validation
    plt.figure(figsize=(10, 6))
    plt.plot(param_grid['max_depth'], np.sqrt(-grid_tree.cv_results_['mean_test_score']))
    plt.xlabel('Max Depth')
    plt.ylabel('RMSE')
    plt.title('Validation Curve')

    min_rmse = np.min(np.sqrt(-grid_tree.cv_results_['mean_test_score']))
    min_rmse_depth = grid_tree.best_params_['max_depth']

    plt.axhline(y=min_rmse, color='r', linestyle='--')
    plt.axvline(x=min_rmse_depth, color='r', linestyle='--')
    plt.savefig('validation_curve.png')
    plt.show()

    # Étape 7 : Entraînement du modèle optimal
    tree_optimal = DecisionTreeRegressor(max_depth=grid_tree.best_params_['max_depth'], random_state=42)
    tree_optimal.fit(X_train, y_train)

    # Évaluation du modèle optimal
    y_train_pred = tree_optimal.predict(X_train)
    y_test_pred = tree_optimal.predict(X_test)

    train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
    test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
    train_mae = mean_absolute_error(y_train, y_train_pred)
    test_mae = mean_absolute_error(y_test, y_test_pred)
    train_medae = median_absolute_error(y_train, y_train_pred)
    test_medae = median_absolute_error(y_test, y_test_pred)
    train_r2 = r2_score(y_train, y_train_pred)
    test_r2 = r2_score(y_test, y_test_pred)
    train_evs = explained_variance_score(y_train, y_train_pred)
    test_evs = explained_variance_score(y_test, y_test_pred)

    results = pd.DataFrame({
        'Métrique': ['RMSE', 'MAE', 'MedAE', 'R2', 'EVS'],
        'Train': [train_rmse, train_mae, train_medae, train_r2, train_evs],
        'Test': [test_rmse, test_mae, test_medae, test_r2, test_evs]
    })
    print(results)

    # Visualisation des prédictions
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, y_test_pred)
    plt.xlabel('True Values')
    plt.ylabel('Predictions')
    plt.xlim(0, 2.00e6)
    plt.ylim(0, 2.00e6)
    plt.plot([0, 2.00e6], [0, 2.00e6], color='red', linestyle='--')
    plt.savefig('predictions_scatter.png')
    plt.show()

# Point d'entrée du script
if __name__ == "__main__":
    main()