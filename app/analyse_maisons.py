import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('kc_house_data.csv')

print("Aperçu des données :")
print(df.head())
print("\nInformations sur le dataset :")
print(df.info())

print("\nStatistiques descriptives :")
print(df.describe()) 