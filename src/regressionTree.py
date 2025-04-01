import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.tree import DecisionTreeRegressor
import numpy as np
from sklearn.metrics import median_absolute_error, r2_score, explained_variance_score


df = pd.read_csv("kc_house_data.csv")

print(df.describe())

df["price_per_sqft_living"] = df["price"] / df["sqft_living"]
df["AsBeenRenovated"] = df["yr_renovated"].apply(lambda x: 0 if x == 0 else 1)
df = df.drop(['date'], axis=1)

df = df[df['price'] < df['price'].quantile(0.99)]


X = df.drop("price", axis=1)
y = df["price"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
tree = DecisionTreeRegressor(random_state=42)
tree.fit(X_train, y_train) 

importances = pd.Series(tree.feature_importances_, index=X_train.columns)
importances_sorted = importances.sort_values()
plt.barh(importances_sorted.index, importances_sorted)
plt.title("Feature importances")
plt.xlabel("Importance")
plt.ylabel("Features")
# plt.savefig('feature_importances.png')

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

param_grid = { 'max_depth': np.arange(1,21) }

tree = DecisionTreeRegressor(random_state=42)

grid_tree = GridSearchCV(tree, param_grid, cv=5, scoring='neg_mean_squared_error')

grid_tree.fit(X_train, y_train)

print("Profondeur optimale:", grid_tree.best_params_['max_depth'])

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

tree_optimal = DecisionTreeRegressor(max_depth=grid_tree.best_params_['max_depth'], random_state=42)
tree_optimal.fit(X_train, y_train)

y_train_pred = tree_optimal.predict(X_train)
y_test_pred = tree_optimal.predict(X_test)

train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred)) # RMSE : Root Mean Squared Error (Erreur quadratique moyenne)
test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
train_mae = mean_absolute_error(y_train, y_train_pred) # MAE : Mean Absolute Error (Erreur absolue moyenne)
test_mae = mean_absolute_error(y_test, y_test_pred)
train_medae = median_absolute_error(y_train, y_train_pred) # MedAE : Median Absolute Error (Erreur absolue médiane)
test_medae = median_absolute_error(y_test, y_test_pred)
train_r2 = r2_score(y_train, y_train_pred) # R2 : Coefficient de détermination
test_r2 = r2_score(y_test, y_test_pred)
train_evs = explained_variance_score(y_train, y_train_pred) # EVS : Explained Variance Score (Score de variance expliquée)
test_evs = explained_variance_score(y_test, y_test_pred)

results = pd.DataFrame({
    'Métrique': ['RMSE', 'MAE', 'MedAE', 'R2', 'EVS'],
    'Train': [train_rmse, train_mae, train_medae, train_r2, train_evs],
    'Test': [test_rmse, test_mae, test_medae, test_r2, test_evs]
})

print(results)

plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_test_pred)
plt.xlabel('True Values')
plt.ylabel('Predictions')

plt.xlim(0, 2.00e6)
plt.ylim(0, 2.00e6)

plt.plot([0, 2.00e6], [0, 2.00e6], color='red', linestyle='--')

plt.show()



