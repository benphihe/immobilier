import pandas as pd
df = pd.read_csv("kc_house_data.csv")
print("Dimensions :", df.shape) # (lignes, colonnes)
print(df.head(7))
