import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

class DataManager:
    def __init__(self, data_path='data/kc_house_data.csv'):
        self.data_path = data_path
        self.data = None
        self.X = None
        self.y = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.scaler = StandardScaler()
        
    def load_data(self):
        """Charge les données depuis le fichier CSV"""
        self.data = pd.read_csv(self.data_path)
        return self
        
    def prepare_data(self, target_column='price', test_size=0.2):
        """Prépare les données pour l'entraînement"""
        # Séparation des features et de la cible
        self.X = self.data.drop([target_column, 'id', 'date'], axis=1)
        self.y = self.data[target_column]
        
        # Division en ensembles d'entraînement et de test
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=test_size, random_state=42
        )
        
        # Standardisation des features
        self.X_train = self.scaler.fit_transform(self.X_train)
        self.X_test = self.scaler.transform(self.X_test)
        
        return self
    
    def get_feature_names(self):
        """Retourne les noms des features"""
        return self.X.columns.tolist()
    
    def get_data_info(self):
        """Retourne les informations sur le dataset"""
        return {
            'shape': self.data.shape,
            'columns': self.data.columns.tolist(),
            'missing_values': self.data.isnull().sum().to_dict(),
            'data_types': self.data.dtypes.to_dict()
        } 