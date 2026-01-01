"""Unit tests for decision rule module."""
import pytest
from engine.decision_rule import decide, SHIP_THRESHOLD, ROLLBACK_THRESHOLD


def test_ship_decision():
    """Test SHIP decision for high probability."""
    result = {
        "prob_treatment_better": 0.97,
        "lift_mean": 0.05,
        "ci_95": [0.02, 0.08]
    }
    assert decide(result) == "SHIP"


def test_rollback_decision():
    """Test ROLLBACK decision for low probability."""
    result = {
        "prob_treatment_better": 0.55,
        "lift_mean": -0.01,
        "ci_95": [-0.03, 0.01]
    }
    assert decide(result) == "ROLLBACK"


def test_iterate_decision():
    """Test ITERATE decision for medium probability."""
    result = {
        "prob_treatment_better": 0.75,
        "lift_mean": 0.02,
        "ci_95": [-0.01, 0.05]
    }
    assert decide(result) == "ITERATE"


def test_custom_thresholds():
    """Test decision with custom thresholds."""
    result = {
        "prob_treatment_better": 0.90,
        "lift_mean": 0.03,
        "ci_95": [0.01, 0.05]
    }
    assert decide(result, ship_threshold=0.85, rollback_threshold=0.50) == "SHIP"


def test_missing_key():
    """Test that missing key raises KeyError."""
    result = {
        "lift_mean": 0.05,
        "ci_95": [0.02, 0.08]
    }
    with pytest.raises(KeyError):
        decide(result)


def test_invalid_thresholds():
    """Test that invalid thresholds raise ValueError."""
    result = {
        "prob_treatment_better": 0.75,
        "lift_mean": 0.02,
        "ci_95": [-0.01, 0.05]
    }
    with pytest.raises(ValueError):
        decide(result, ship_threshold=0.50, rollback_threshold=0.60)

