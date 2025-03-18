import pandas as pd
from ydata_profiling import ProfileReport

df = pd.read_csv("kc_house_data.csv")

df["isReno"] = df["yr_renovated"].apply(lambda x: 0 if x == 0 else 1)

print("Pourcentage de maison avec 3 chambres ou plus : ", df[df["bedrooms"] >= 3].shape[0] / df.shape[0] * 100)

 
profile = ProfileReport(df, title="Profiling Report")
profile.to_file("your_report.html")

