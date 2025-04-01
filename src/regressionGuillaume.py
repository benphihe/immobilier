import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error, mean_absolute_error
from .model import load_and_prepare_data

class regressionGuillaume:
    def __init__(self):
        self.df = load_and_prepare_data()
        self.x = self.df.drop("price", axis=1)
        self.y = self.df["price"]
    
    def column_transform(self):
        for col in self.x.select_dtypes(include=['object']).columns:
            if col == 'date':  
                self.x[col] = pd.to_datetime(self.x[col]).astype(int) / 10**9 
            else:
                le = LabelEncoder()
                self.x[col] = le.fit_transform(self.x[col])

    def split_data(self):
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.x, self.y, test_size=0.2, random_state=42)

    def linear_regression(self):
        reg = LinearRegression()
        reg.fit(self.X_train, self.y_train)

        self.y_pred = reg.predict(self.X_test)

        self.mse = mean_squared_error(self.y_test, self.y_pred)
        self.rmse = self.mse ** 0.5
        self.mae = mean_absolute_error(self.y_test, self.y_pred)
        print("RMSE : ", self.rmse, "MAE : ", self.mae)

    def plot(self):
        plt.figure(figsize=(10, 6))
        plt.scatter(self.y_test, self.y_pred, alpha=0.5, label="Prédictions")
        plt.xlabel('True Values')
        plt.ylabel('Predictions')

        plt.xlim(0, 2.00e6)
        plt.ylim(0, 2.00e6)

        plt.plot([0, 2.00e6], [0, 2.00e6], color='red', linestyle='--', label="Ligne parfaite")

        plt.show()

    def run(self):
        self.column_transform()
        self.split_data()
        self.linear_regression()
        self.plot()

