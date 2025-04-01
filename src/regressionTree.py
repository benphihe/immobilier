import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, median_absolute_error, r2_score, explained_variance_score
from sklearn.tree import DecisionTreeRegressor
import numpy as np
from .model import load_and_prepare_data

class RegressionTree:
    def __init__(self):
        self.df = load_and_prepare_data()
        self.x = self.df.drop("price", axis=1)
        self.y = self.df["price"]

    def split_data(self):
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.x, self.y, test_size=0.2, random_state=42
        )

    def train_initial_model(self):
        self.tree = DecisionTreeRegressor(random_state=42)
        self.tree.fit(self.X_train, self.y_train)

    def evaluate_initial_model(self):
        self.y_train_pred = self.tree.predict(self.X_train)
        self.y_test_pred = self.tree.predict(self.X_test)

        self.train_rmse = np.sqrt(mean_squared_error(self.y_train, self.y_train_pred))
        self.test_rmse = np.sqrt(mean_squared_error(self.y_test, self.y_test_pred))
        self.train_mae = mean_absolute_error(self.y_train, self.y_train_pred)
        self.test_mae = mean_absolute_error(self.y_test, self.y_test_pred)
        self.train_r2 = r2_score(self.y_train, self.y_train_pred)
        self.test_r2 = r2_score(self.y_test, self.y_test_pred)

        results = pd.DataFrame({
            'Métrique': ['RMSE', 'MAE', 'R2'],
            'Train': [self.train_rmse, self.train_mae, self.train_r2],
            'Test': [self.test_rmse, self.test_mae, self.test_r2]
        })
        print(results)

    def hyperparameter_tuning(self):
        param_grid = {'max_depth': np.arange(1, 21)}
        grid_tree = GridSearchCV(self.tree, param_grid, cv=5, scoring='neg_mean_squared_error')
        grid_tree.fit(self.X_train, self.y_train)

        self.best_depth = grid_tree.best_params_['max_depth']
        print("Profondeur optimale:", self.best_depth)

        # Visualisation de la courbe de validation
        plt.figure(figsize=(10, 6))
        plt.plot(param_grid['max_depth'], np.sqrt(-grid_tree.cv_results_['mean_test_score']))
        plt.xlabel('Max Depth')
        plt.ylabel('RMSE')
        plt.title('Validation Curve')

        min_rmse = np.min(np.sqrt(-grid_tree.cv_results_['mean_test_score']))
        plt.axhline(y=min_rmse, color='r', linestyle='--')
        plt.axvline(x=self.best_depth, color='r', linestyle='--')
        plt.show()

    def train_optimal_model(self):
        self.tree_optimal = DecisionTreeRegressor(max_depth=self.best_depth, random_state=42)
        self.tree_optimal.fit(self.X_train, self.y_train)

    def evaluate_optimal_model(self):
        self.y_train_pred = self.tree_optimal.predict(self.X_train)
        self.y_test_pred = self.tree_optimal.predict(self.X_test)

        self.train_rmse = np.sqrt(mean_squared_error(self.y_train, self.y_train_pred))
        self.test_rmse = np.sqrt(mean_squared_error(self.y_test, self.y_test_pred))
        self.train_mae = mean_absolute_error(self.y_train, self.y_train_pred)
        self.test_mae = mean_absolute_error(self.y_test, self.y_test_pred)
        self.train_r2 = r2_score(self.y_train, self.y_train_pred)
        self.test_r2 = r2_score(self.y_test, self.y_test_pred)

        results = pd.DataFrame({
            'Métrique': ['RMSE', 'MAE', 'R2'],
            'Train': [self.train_rmse, self.train_mae, self.train_r2],
            'Test': [self.test_rmse, self.test_mae, self.test_r2]
        })
        print(results)

    def plot_predictions(self):
        plt.figure(figsize=(10, 6))
        plt.scatter(self.y_test, self.y_test_pred)
        plt.xlabel('True Values')
        plt.ylabel('Predictions')
        plt.xlim(0, 2.00e6)
        plt.ylim(0, 2.00e6)
        plt.plot([0, 2.00e6], [0, 2.00e6], color='red', linestyle='--')
        plt.show()

    def run(self):
        self.split_data()
        self.train_initial_model()
        self.evaluate_initial_model()
        self.plot_feature_importances()
        self.hyperparameter_tuning()
        self.train_optimal_model()
        self.evaluate_optimal_model()
        self.plot_predictions()