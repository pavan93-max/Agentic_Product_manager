"""Unit tests for user simulation module."""
import pytest
import numpy as np
from engine.simulator import simulate_users


def test_simulator_output_shape():
    """Test that simulator returns correct shape."""
    variant = {"cta_color": "blue", "discount": 0}
    n = 1000
    result = simulate_users(variant, n)
    
    assert len(result) == n
    assert all(x in [0, 1] for x in result)


def test_simulator_reproducibility():
    """Test that simulator is reproducible with seed."""
    variant = {"cta_color": "green", "discount": 10}
    n = 100
    
    result1 = simulate_users(variant, n, seed=42)
    result2 = simulate_users(variant, n, seed=42)
    
    np.testing.assert_array_equal(result1, result2)


def test_simulator_green_cta_effect():
    """Test that green CTA increases conversion rate."""
    variant_blue = {"cta_color": "blue", "discount": 0}
    variant_green = {"cta_color": "green", "discount": 0}
    n = 10000
    
    blue_rate = simulate_users(variant_blue, n, seed=42).mean()
    green_rate = simulate_users(variant_green, n, seed=42).mean()
    
    assert green_rate > blue_rate


def test_simulator_discount_effect():
    """Test that discount increases conversion rate."""
    variant_no_discount = {"cta_color": "blue", "discount": 0}
    variant_discount = {"cta_color": "blue", "discount": 15}
    n = 10000
    
    no_discount_rate = simulate_users(variant_no_discount, n, seed=42).mean()
    discount_rate = simulate_users(variant_discount, n, seed=42).mean()
    
    assert discount_rate > no_discount_rate


def test_simulator_invalid_n():
    """Test that invalid n raises ValueError."""
    variant = {"cta_color": "blue"}
    
    with pytest.raises(ValueError):
        simulate_users(variant, -1)
    
    with pytest.raises(ValueError):
        simulate_users(variant, 0)

