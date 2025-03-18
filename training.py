from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, accuracy_score, f1_score
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("kc_house_data.csv")
df = df[df['price'] < df['price'].quantile(0.99)]
df = df.drop(['date'], axis=1)

X = df.drop('price', axis=1)
y = df['price']

X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.2,random_state=42)

reg = LinearRegression()
reg.fit(X_train, y_train)

y_pred = reg.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
rmse = mse ** 0.5
mae = mean_absolute_error(y_test, y_pred)
print("RMSE:", rmse, "MAE:", mae)

plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred)
plt.xlabel('True Values')
plt.ylabel('Predictions')

plt.xlim(0, 2.00e6)
plt.ylim(0, 2.00e6)

plt.plot([0, 2.00e6], [0, 2.00e6], color='red', linestyle='--')

plt.savefig('true_vs_predicted.png')
plt.show()