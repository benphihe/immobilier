import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error, mean_absolute_error
from .processor_imo import HouseDataAnalyzer

analyzer = HouseDataAnalyzer('../data/kc_house_data.csv')
analyzer.load_and_prepare_data()

class RegressionLinear:
    def __init__(self):
        self.df = analyzer.df

        self.x = self.df.drop("price", axis=1)
        self.y = self.df["price"]

    def split_data(self, test_size, random_state):
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.x, self.y, test_size=test_size, random_state=random_state
        )

    def train_model(self):
        self.reg = LinearRegression()
        self.reg.fit(self.X_train, self.y_train)

    def predict(self):
        self.y_pred = self.reg.predict(self.X_test)

    def evaluate_model(self):
        self.mse = mean_squared_error(self.y_test, self.y_pred)
        self.rmse = self.mse ** 0.5
        self.mae = mean_absolute_error(self.y_test, self.y_pred)

        print("RMSE : ", self.rmse, "MAE : ", self.mae)

    def linear_regression(self):
        self.train_model()
        self.predict()
        self.evaluate_model()

    def plot(self):
        plt.figure(figsize=(10, 6))
        plt.scatter(self.y_test, self.y_pred, alpha=0.5, label="Prédictions")
        plt.xlabel('True Values')
        plt.ylabel('Predictions')

        plt.xlim(0, 2.00e6)
        plt.ylim(0, 2.00e6)

        plt.plot([0, 2.00e6], [0, 2.00e6], color='red', linestyle='--', label="Ligne parfaite")
        plt.legend()
        plt.title("True vs Predicted Values")
        plt.show()

    def run(self):
        print("Training linear regression model...")
        self.linear_regression()

        self.plot()
