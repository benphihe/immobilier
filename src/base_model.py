from abc import ABC, abstractmethod
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

class BaseModel(ABC):
    def __init__(self):
        self.model = None
        self.feature_names = None
        
    @abstractmethod
    def train(self, X_train, y_train):
        """Entraîne le modèle"""
        pass
    
    @abstractmethod
    def predict(self, X):
        """Fait des prédictions"""
        pass
    
    def evaluate(self, X_test, y_test):
        """Évalue le modèle avec différentes métriques"""
        y_pred = self.predict(X_test)
        
        metrics = {
            'mse': mean_squared_error(y_test, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
            'mae': mean_absolute_error(y_test, y_pred),
            'r2': r2_score(y_test, y_pred)
        }
        
        return metrics
    
    def set_feature_names(self, feature_names):
        """Définit les noms des features"""
        self.feature_names = feature_names
    
    def get_feature_importance(self):
        """Retourne l'importance des features si disponible"""
        if hasattr(self.model, 'feature_importances_'):
            return dict(zip(self.feature_names, self.model.feature_importances_))
        return None 