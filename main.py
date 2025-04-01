import sys
import pandas as pd
import os

# Change to the directory where the script is located
os.chdir(os.path.dirname(os.path.abspath(__file__)))

df = pd.read_csv("kc_house_data.csv")
print("Dimensions : ", df.shape)
print(df.head(100))