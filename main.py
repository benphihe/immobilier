import sys
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from ydata_profiling import ProfileReport
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error, mean_absolute_error


# Load the dataset
df = pd.read_csv("kc_house_data.csv")
print("Dimensions : ", df.shape)

df["price_per_sqft_living"] = df["price"] / df["sqft_living"]
df["AsBeenRenovated"] = df["yr_renovated"].apply(lambda x: 0 if x == 0 else 1)

print("Pourcentage de maison avec 3 chambres ou plus : ", df[df["bedrooms"] >= 3].shape[0] / df.shape[0] * 100)

print(df.describe())

print(df.head(100))

df = df[df['price'] < df['price'].quantile(0.99)]


plt.figure(figsize=(10, 6))
sns.boxplot(x='grade', y='price', data=df)
plt.title('Price Distribution by Grade')
plt.xlabel('Grade')
plt.ylabel('Price')
plt.savefig('price_by_grade_boxplot.png')

plt.figure(figsize=(10, 6))
sns.scatterplot(x='sqft_living', y='price', data=df)
plt.title('Price vs Square Foot Living Space')
plt.xlabel('Square Foot Living Space')
plt.ylabel('Price')
plt.savefig('price_vs_sqft_living_scatter.png')

plt.figure(figsize=(10, 6))
sns.barplot(x='grade', y='price', data=df)
plt.title('Price Distribution by Grade')
plt.xlabel('Grade')
plt.ylabel('Price')
plt.savefig('price_by_grade_barplot.png')

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

comparison = pd.DataFrame({'Réel': y_test[:10].values, 'Prédit': y_pred[:10]})
print(comparison)

mse = mean_squared_error(y_test, y_pred)
rmse = mse ** 0.5
mae = mean_absolute_error(y_test, y_pred)
print("RMSE : ", rmse)
print("MAE : ", mae)