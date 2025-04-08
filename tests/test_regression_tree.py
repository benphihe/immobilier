import pytest
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg') 
from src.regressionTree import RegressionTree

@pytest.fixture
def regression_tree():
    """Fixture pour créer une instance de RegressionTree"""
    return RegressionTree(show_plots=False)

def test_split_data(regression_tree):
    """Test de la méthode split_data"""
    regression_tree.split_data(test_size=0.2, random_state=42)
    
   
    assert len(regression_tree.X_train) + len(regression_tree.X_test) == len(regression_tree.x)
    assert len(regression_tree.y_train) + len(regression_tree.y_test) == len(regression_tree.y)
    

    assert pytest.approx(len(regression_tree.X_test) / len(regression_tree.x), 0.1) == 0.2

def test_train_model(regression_tree):
    """Test de la méthode train_model"""
    regression_tree.split_data()
    regression_tree.train_model(max_depth=3)
    
 
    assert regression_tree.tree is not None
    assert regression_tree.tree.max_depth == 3

def test_evaluate_model(regression_tree, capsys):
    """Test de la méthode evaluate_model"""
    regression_tree.split_data()
    regression_tree.train_model(max_depth=3)
   
    regression_tree.evaluate_model()
    captured = capsys.readouterr()
    
    assert 'Train RMSE' in captured.out
    assert 'Test RMSE' in captured.out
    assert 'Train R2' in captured.out
    assert 'Test R2' in captured.out

def test_hyperparameter_tuning(regression_tree, capsys):
    """Test de la méthode hyperparameter_tuning"""
    regression_tree.split_data()
    regression_tree.train_model()
    
    regression_tree.hyperparameter_tuning()
    captured = capsys.readouterr()
    
    assert 'Profondeur optimale:' in captured.out
    assert regression_tree.best_depth is not None
    assert isinstance(regression_tree.best_depth, (int, np.integer))

def test_plot_predictions(regression_tree):
    """Test de la méthode plot_predictions"""
    regression_tree.split_data()
    regression_tree.train_model(max_depth=3)
    
    regression_tree.plot_predictions()
 
    assert regression_tree.y_test is not None
    assert regression_tree.X_test is not None

def test_run(regression_tree, capsys):
    """Test de la méthode run complète"""
  
    regression_tree.run()
    captured = capsys.readouterr()
    
    assert 'Training initial model...' in captured.out
    assert 'Performing hyperparameter tuning...' in captured.out
    assert 'Training optimal model...' in captured.out
