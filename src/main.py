from data_manager import DataManager
from linear_regression_model import LinearRegressionModel

def main():
    # Initialisation du gestionnaire de données
    data_manager = DataManager()
    
    # Chargement et préparation des données
    print("Chargement des données...")
    data_manager.load_data()
    print("Informations sur le dataset:")
    print(data_manager.get_data_info())
    
    print("\nPréparation des données...")
    data_manager.prepare_data()
    
    # Initialisation et entraînement du modèle
    print("\nEntraînement du modèle...")
    model = LinearRegressionModel()
    model.set_feature_names(data_manager.get_feature_names())
    model.train(data_manager.X_train, data_manager.y_train)
    
    # Évaluation du modèle
    print("\nÉvaluation du modèle:")
    metrics = model.evaluate(data_manager.X_test, data_manager.y_test)
    for metric_name, value in metrics.items():
        print(f"{metric_name.upper()}: {value:.4f}")
    
    # Affichage des coefficients
    print("\nCoefficients du modèle:")
    coefficients = model.get_coefficients()
    for feature, coef in coefficients.items():
        print(f"{feature}: {coef:.4f}")
    
    print(f"\nIntercept: {model.get_intercept():.4f}")

if __name__ == "__main__":
    main() 