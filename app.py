from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
from src.processor_imo import HouseDataAnalyzer
from src.regressionTree import RegressionTree

app = FastAPI()

model = RegressionTree()
model.split_data(test_size=0.2, random_state=42)
model.train_model(max_depth=10)

class HouseData(BaseModel):
    id: int =1
    date: str="2014-10-13T00:00:00Z"
    price: float= 1500000.0
    bedrooms: int = 3
    bathrooms: float = 2.5
    sqft_living: float = 2500.0
    sqft_lot: float
    floors: float
    waterfront: int
    view: int
    condition: int
    grade: int
    sqft_above: float
    sqft_basement: float
    yr_built: int
    yr_renovated: int
    zipcode: int
    lat: float
    long: float
    sqft_living15: float
    sqft_lot15: float

@app.post("/predict")
def predict_price(house_data: HouseData):
    data = pd.DataFrame([house_data.dict()])
    df = HouseDataAnalyzer(df=data).load_and_prepare_data()
    prediction = model.tree.predict(df.drop(columns=['id', 'price']))
    return {"predicted_price": prediction[0]}


