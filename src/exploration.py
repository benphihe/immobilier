import pandas as pd
from ydata_profiling import ProfileReport
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, accuracy_score, f1_score
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("kc_house_data.csv")

df["price_per_sqft_living"] = df["price"] / df["sqft_living"]
df["AsBeenRenovated"] = df["yr_renovated"].apply(lambda x: 0 if x == 0 else 1)
df = df.drop(['date'], axis=1)
df = df[df['price'] < df['price'].quantile(0.99)]






