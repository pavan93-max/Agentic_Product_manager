"""Unit tests for Bayesian A/B testing module."""
import pytest
import numpy as np
from engine.bayesian import bayesian_ab_test


def test_bayesian_output_structure():
    """Test that bayesian_ab_test returns expected structure."""
    np.random.seed(42)
    c = np.random.binomial(1, 0.1, 1000)
    t = np.random.binomial(1, 0.12, 1000)
    result = bayesian_ab_test(c, t)
    
    assert "lift_mean" in result
    assert "prob_treatment_better" in result
    assert "ci_95" in result
    assert len(result["ci_95"]) == 2
    assert isinstance(result["lift_mean"], float)
    assert isinstance(result["prob_treatment_better"], float)
    assert 0.0 <= result["prob_treatment_better"] <= 1.0


def test_bayesian_ci_order():
    """Test that credible interval is ordered correctly."""
    np.random.seed(42)
    c = np.random.binomial(1, 0.1, 1000)
    t = np.random.binomial(1, 0.12, 1000)
    result = bayesian_ab_test(c, t)
    
    ci_low, ci_high = result["ci_95"]
    assert ci_low <= ci_high


def test_bayesian_empty_input():
    """Test that empty inputs raise ValueError."""
    with pytest.raises(ValueError):
        bayesian_ab_test(np.array([]), np.array([1, 0, 1]))
    
    with pytest.raises(ValueError):
        bayesian_ab_test(np.array([1, 0, 1]), np.array([]))


def test_bayesian_list_input():
    """Test that list inputs are converted to numpy arrays."""
    c = [1, 0, 1, 0, 1]
    t = [1, 1, 1, 0, 1]
    result = bayesian_ab_test(c, t)
    
    assert "lift_mean" in result
    assert isinstance(result["lift_mean"], float)
