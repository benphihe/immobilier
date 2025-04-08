import pandas as pd
from src.regressionLinear import RegressionLinear
import pytest

@pytest.fixture
def regression():
    reg = RegressionLinear()
    reg.split_data(0.2, 42)
    return reg

def test_split_data(regression):
    assert len(regression.X_train) > 0, "X_train est vide."
    assert len(regression.X_test) > 0, "X_test est vide."
    assert len(regression.y_train) > 0, "y_train est vide."
    assert len(regression.y_test) > 0, "y_test est vide."

def test_train_model(regression):
    regression.train_model()
    assert hasattr(regression, 'reg'), "Le modèle n'a pas été entraîné."
    assert regression.reg is not None, "Le modèle est None."

def test_predict(regression):
    regression.train_model()
    regression.predict()
    assert len(regression.y_pred) == len(regression.y_test), "y_pred n'est pas égal a y_test"

def test_evaluate_model(regression):
    regression.train_model()
    regression.predict()
    regression.evaluate_model()
    assert regression.mse > 0, "MSE est incorrect."
    assert regression.rmse > 0, "RMSE est incorrect."
    assert regression.mae > 0, "MAE est incorrect."

def test_run():
    regression = RegressionLinear()
    regression.split_data(0.2, 42)
    regression.run()
    assert regression.reg is not None, "le modèle n'a pas été entrainé correctement"
    assert len(regression.y_pred) == len(regression.y_test), "y_pred n'est pas égal a y_test"
    assert regression.mse > 0, "MSE est incorrect."