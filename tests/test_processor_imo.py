from src.processor_imo import HouseDataAnalyzer
import pandas as pd
import pytest

@pytest.fixture
def sample_df():
    return pd.DataFrame({
        'id': [1234567890, 1234567891, 1234567892],
        'date': ['2014-05-02', '2015-03-12', '2013-07-15'],
        'price': [450000, 350000, 200000],
        'bedrooms': [3, 2, 4],
        'bathrooms': [2.5, 1.0, 3.0],
        'sqft_living': [1500, 1200, 800],
        'sqft_lot': [5000, 4000, 3000],
        'floors': [2, 1, 1],
        'waterfront': [0, 0, 1],
        'view': [0, 1, 0],
        'condition': [3, 4, 5],
        'grade': [7, 6, 5],
        'sqft_above': [1000, 800, 600],
        'sqft_basement': [500, 400, 200],
        'yr_built': [1990, 1985, 2000],
        'yr_renovated': [2000, 0, 2015],
        'zipcode': [98178, 98125, 98028],
        'lat': [47.5112, 47.7210, 47.7379],
        'long': [-122.257, -122.319, -122.233],
        'sqft_living15': [1340, 1690, 2720],
        'sqft_lot15': [5650, 7639, 8062]
    })


def test_load_and_prepare_data(sample_df):
    prep = HouseDataAnalyzer(df=sample_df)
    prep.load_and_prepare_data()
    assert 'price_per_sqft_living' in prep.df.columns
    assert 'AsBeenRenovated' in prep.df.columns
    assert 'date' not in prep.df.columns
    assert len(prep.df.columns) == 22
