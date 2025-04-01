import sys
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from ydata_profiling import ProfileReport
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error, mean_absolute_error
from src.regressionGuillaume import regressionGuillaume


def main():
    print("Hello World")
    regression_Guillaume = regressionGuillaume()
    regression_Guillaume.run()



if __name__ == "__main__":
    main()