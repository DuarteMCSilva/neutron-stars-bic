import pytest
import numpy as np
import utils.statistics as component

def test_generate_size():
    generated = component.generate_normal(100, 20, 2)
    assert generated.size == 100

def test_generate_mean():
    generated = component.generate_normal(100, 20, 2)
 
    assert generated.mean() == pytest.approx(20, abs=1)

def test_normalize_array():
    initial = component.generate_normal(100, 20, 2)
    
    normalized = component.normalize(initial)
    
    assert normalized.std() == pytest.approx(1, abs=0.1)

def test_inv_normalization_array():
    initial = np.random.randn(100)
    mean = 20
    std = 1

    normalized = component.inv_normalize(initial, mean, std)
    
    assert normalized.std() == pytest.approx(std, abs=1)
    assert normalized.mean() == pytest.approx(20, abs=1)
