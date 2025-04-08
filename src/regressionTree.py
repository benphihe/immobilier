import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.tree import DecisionTreeRegressor
import numpy as np
from .processor_imo import HouseDataAnalyzer

analyzer = HouseDataAnalyzer('../data/kc_house_data.csv')
analyzer.load_and_prepare_data()

class RegressionTree:
    def __init__(self, show_plots=True):
        self.df = analyzer.df
        self.show_plots = show_plots
        self.x = self.df.drop("price", axis=1)
        self.y = self.df["price"]

    def split_data(self, test_size, random_state):
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.x, self.y, test_size=test_size, random_state=random_state
        )  

    def train_model(self, max_depth=None):
        self.tree = DecisionTreeRegressor(max_depth=max_depth, random_state=42)
        self.tree.fit(self.X_train, self.y_train)

    def predict(self):
        self.y_pred = self.tree.predict(self.X_test)

    def evaluate_model(self, model=None):
        if model is None:
            model = self.tree

        y_train_pred = model.predict(self.X_train)
        y_test_pred = model.predict(self.X_test)

        metrics = {
            'Train RMSE': np.sqrt(mean_squared_error(self.y_train, y_train_pred)),
            'Test RMSE': np.sqrt(mean_squared_error(self.y_test, y_test_pred)),
            'Train MAE': mean_absolute_error(self.y_train, y_train_pred),
            'Test MAE': mean_absolute_error(self.y_test, y_test_pred),
            'Train R2': r2_score(self.y_train, y_train_pred),
            'Test R2': r2_score(self.y_test, y_test_pred),
        }

        results = pd.DataFrame(metrics, index=['Value']).T
        print(results)

    def hyperparameter_tuning(self):
        param_grid = {'max_depth': np.arange(1, 21)}
        grid_tree = GridSearchCV(self.tree, param_grid, cv=5, scoring='neg_mean_squared_error')
        grid_tree.fit(self.X_train, self.y_train)

        self.best_depth = grid_tree.best_params_['max_depth']
        print("Profondeur optimale:", self.best_depth)

        plt.figure(figsize=(10, 6))
        plt.plot(param_grid['max_depth'], np.sqrt(-grid_tree.cv_results_['mean_test_score']))
        plt.xlabel('Max Depth')
        plt.ylabel('RMSE')
        plt.title('Validation Curve')

        min_rmse = np.min(np.sqrt(-grid_tree.cv_results_['mean_test_score']))
        plt.axhline(y=min_rmse, color='r', linestyle='--')
        plt.axvline(x=self.best_depth, color='r', linestyle='--')
        if self.show_plots:
            plt.show()

    def plot_predictions(self, model=None):
        if model is None:
            model = self.tree

        y_test_pred = model.predict(self.X_test)

        plt.figure(figsize=(10, 6))
        plt.scatter(self.y_test, y_test_pred)
        plt.xlabel('True Values')
        plt.ylabel('Predictions')
        plt.xlim(0, 2.00e6)
        plt.ylim(0, 2.00e6)
        plt.plot([0, 2.00e6], [0, 2.00e6], color='red', linestyle='--')
        plt.title('True vs Predicted Values')
        if self.show_plots:
            plt.show()

    def run(self):
        self.train_model()
        self.predict()
        self.evaluate_model()

        print("Performing hyperparameter tuning...")
        self.hyperparameter_tuning()

        print("Training optimal model...")
        self.train_model(max_depth=self.best_depth)
        self.evaluate_model()
        self.plot_predictions()

