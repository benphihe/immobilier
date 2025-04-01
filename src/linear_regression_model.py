from sklearn.linear_model import LinearRegression
from .base_model import BaseModel

class LinearRegressionModel(BaseModel):
    def __init__(self):
        super().__init__()
        self.model = LinearRegression()
        
    def train(self, X_train, y_train):
        """Entraîne le modèle de régression linéaire"""
        self.model.fit(X_train, y_train)
        return self
    
    def predict(self, X):
        """Fait des prédictions avec le modèle"""
        return self.model.predict(X)
    
    def get_coefficients(self):
        """Retourne les coefficients du modèle"""
        if self.feature_names is None:
            return None
        return dict(zip(self.feature_names, self.model.coef_))
    
    def get_intercept(self):
        """Retourne l'intercept du modèle"""
        return self.model.intercept_ 