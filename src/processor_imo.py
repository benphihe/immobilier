import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

class HouseDataAnalyzer:
    def __init__(self, file_path):
        self.file_path = file_path
        self.df = None

    def load_and_prepare_data(self):
        self.df = pd.read_csv(self.file_path)
        
        self.df["price_per_sqft_living"] = self.df["price"] / self.df["sqft_living"]
        self.df["AsBeenRenovated"] = self.df["yr_renovated"].apply(lambda x: 0 if x == 0 else 1)
        
        self.df = self.df.drop(['date'], axis=1)
        self.df = self.df[self.df['price'] < self.df['price'].quantile(0.99)]

    def visualize_data(self):
        if self.df is None:
            raise ValueError("Dataframe is not loaded. Please call load_and_prepare_data() first.")
        
        plt.figure(figsize=(10, 6))
        sns.boxplot(x='grade', y='price', data=self.df)
        plt.title('Price Distribution by Grade')
        plt.xlabel('Grade')
        plt.ylabel('Price')
        plt.show()

        plt.figure(figsize=(10, 6))
        sns.scatterplot(x='sqft_living', y='price', data=self.df)
        plt.title('Price vs Square Foot Living Space')
        plt.xlabel('Square Foot Living Space')
        plt.ylabel('Price')
        plt.show()

        plt.figure(figsize=(10, 6))
        sns.barplot(x='grade', y='price', data=self.df)
        plt.title('Price Distribution by Grade')
        plt.xlabel('Grade')
        plt.ylabel('Price')
        plt.show()

def main():
    analyzer = HouseDataAnalyzer('../data/kc_house_data.csv')
    analyzer.load_and_prepare_data()
    # analyzer.visualize_data()
