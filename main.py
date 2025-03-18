import sys
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from ydata_profiling import ProfileReport

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