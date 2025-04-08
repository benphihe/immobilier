from src.processor_imo import HouseDataAnalyzer
import pandas as pd


def test_load_and_prepare_data():
    prep = HouseDataAnalyzer('data/kc_house_data.csv')
    prep.load_and_prepare_data()
    assert prep.df is not None, "Dataframe should not be None after loading data."
    assert 'price_per_sqft_living' in prep.df.columns, "price_per_sqft_living column should be created."
    assert 'AsBeenRenovated' in prep.df.columns, "AsBeenRenovated column should be created."
    assert 'date' not in prep.df.columns, "date column should be dropped."
    assert prep.df['price'].max() < prep.df['price'].quantile(0.99), "Price should be filtered to remove outliers."
    assert prep.df['price'].min() >= 0, "Price should not contain negative values."