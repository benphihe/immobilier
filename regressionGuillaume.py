import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error, mean_absolute_error


df = pd.read_csv("kc_house_data.csv")

df["price_per_sqft_living"] = df["price"] / df["sqft_living"]
df["AsBeenRenovated"] = df["yr_renovated"].apply(lambda x: 0 if x == 0 else 1)

df = df[df['price'] < df['price'].quantile(0.99)]

X = df.drop("price", axis=1)
y = df["price"]

for col in X.select_dtypes(include=['object']).columns:
    if col == 'date':  
        X[col] = pd.to_datetime(X[col]).astype(int) / 10**9 
    else:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

reg = LinearRegression()
reg.fit(X_train, y_train)

y_pred = reg.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
rmse = mse ** 0.5
mae = mean_absolute_error(y_test, y_pred)
print("RMSE : ", rmse, "MAE : ", mae)

plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, alpha=0.5, label="Prédictions")
plt.xlabel('True Values')
plt.ylabel('Predictions')

plt.xlim(0, 2.00e6)
plt.ylim(0, 2.00e6)

plt.plot([0, 2.00e6], [0, 2.00e6], color='red', linestyle='--', label="Ligne parfaite")

plt.savefig('true_vs_predictedGuillaume.png')
