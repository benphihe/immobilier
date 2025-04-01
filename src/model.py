import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def load_and_prepare_data():
    df = pd.read_csv('data/kc_house_data.csv')
    
    df["price_per_sqft_living"] = df["price"] / df["sqft_living"]
    df["AsBeenRenovated"] = df["yr_renovated"].apply(lambda x: 0 if x == 0 else 1)
    
    df = df.drop(['date'], axis=1)
    
    df = df[df['price'] < df['price'].quantile(0.99)]
    
    return df

def visualize_data(df):
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='grade', y='price', data=df)
    plt.title('Price Distribution by Grade')
    plt.xlabel('Grade')
    plt.ylabel('Price')
    plt.show()

    plt.figure(figsize=(10, 6))
    sns.scatterplot(x='sqft_living', y='price', data=df)
    plt.title('Price vs Square Foot Living Space')
    plt.xlabel('Square Foot Living Space')
    plt.ylabel('Price')
    plt.show()

    plt.figure(figsize=(10, 6))
    sns.barplot(x='grade', y='price', data=df)
    plt.title('Price Distribution by Grade')
    plt.xlabel('Grade')
    plt.ylabel('Price')
    plt.show()

def main():
    df = load_and_prepare_data()
    
    visualize_data(df)

if __name__ == "__main__":
    main()