import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("kc_house_data.csv")

plt.figure(figsize=(10, 6))
plt.hist(df["price"], bins=100, color="skyblue")
plt.axvline(df["price"].mean(), color="red", linestyle="dashed", linewidth=1)
plt.axvline(df["price"].median(), color="green", linestyle="dashed", linewidth=1)
plt.axvline(df["price"].mode()[0], color="yellow", linestyle="dashed", linewidth=1)
plt.legend(["Mean", "Median", "Mode"])
plt.xlabel("Price")
plt.ylabel("grade")
plt.title("Price distribution")
plt.show()
