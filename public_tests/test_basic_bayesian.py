"""
Public Test Suite - Part 2: Bayesian Networks

These tests check BASIC functionality only. Passing all tests here does NOT
guarantee full credit. Hidden tests will be more comprehensive and strict.

Run with: pytest public_tests/test_basic_bayesian.py -v
"""

from __future__ import annotations

import sys
from pathlib import Path

# Add code directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / "code"))

import pandas as pd
import pytest

import bayesian_network


@pytest.fixture(scope="module")
def network() -> bayesian_network.BayesianNetwork:
    """Load the Bayesian network."""
    return bayesian_network.load_network()


@pytest.fixture(scope="module")
def expected_values() -> pd.DataFrame:
    """Load expected values computed from solution."""
    expected_path = Path(__file__).parent / "expected_bayesian.csv"
    return pd.read_csv(expected_path)


def get_expected(expected_values: pd.DataFrame, test_name: str) -> tuple[float, float]:
    """Helper to extract expected value and tolerance."""
    row = expected_values[expected_values['test_name'] == test_name].iloc[0]
    return row['expected_value'], row['tolerance']


def test_network_loads(network: bayesian_network.BayesianNetwork) -> None:
    """
    Test that network loads successfully from JSON.
    
    Expected: Network should have 4 nodes in correct topological order.
    """
    assert hasattr(network, "nodes"), "Network should have 'nodes' attribute"
    assert hasattr(network, "order"), "Network should have 'order' attribute"
    assert hasattr(network, "cpt"), "Network should have 'cpt' attribute"
    
    assert len(network.nodes) == 4, f"Expected 4 nodes, got {len(network.nodes)}"
    assert set(network.nodes) == {"A", "B", "C", "D"}, f"Expected nodes A, B, C, D, got {network.nodes}"
    
    # Order should be topological (provided in JSON)
    assert network.order == ["A", "B", "C", "D"], f"Expected order [A,B,C,D], got {network.order}"
    
    print("✓ Network loaded successfully with 4 nodes")


def test_get_probability(network: bayesian_network.BayesianNetwork) -> None:
    """
    Test CPT lookup functionality.
    
    Tests:
    - Root node with no parents: P(A=True) = 0.5
    - Node with parents: P(C=True | A=True, B=False) = 0.8
    - Complement: P(C=False | A=True, B=False) = 0.2
    """
    # Test root node
    prob_a = network.get_probability("A", True, {})
    assert abs(prob_a - 0.5) <= 0.001, f"P(A=True) should be 0.5, got {prob_a:.4f}"
    
    # Test node with parents
    prob_c_true = network.get_probability("C", True, {"A": True, "B": False})
    assert abs(prob_c_true - 0.8) <= 0.001, f"P(C=True | A=True, B=False) should be 0.8, got {prob_c_true:.4f}"
    
    # Test complement
    prob_c_false = network.get_probability("C", False, {"A": True, "B": False})
    assert abs(prob_c_false - 0.2) <= 0.001, f"P(C=False | A=True, B=False) should be 0.2, got {prob_c_false:.4f}"
    
    # Test probabilities sum to 1
    assert abs(prob_c_true + prob_c_false - 1.0) <= 0.001, "P(True) + P(False) should equal 1.0"
    
    print("✓ get_probability works correctly")


def test_marginal_c(network: bayesian_network.BayesianNetwork, expected_values: pd.DataFrame) -> None:
    """
    Test marginal probability P(C=True).
    
    Expected: Should sum over all possible values of parents A and B.
    
    Hint: Use enumerate_all to marginalize
    """
    result = network.query("C")
    prob = result[True]
    
    expected, tolerance = get_expected(expected_values, 'test_marginal_c')
    
    assert abs(prob - expected) <= tolerance, (
        f"P(C=True) = {prob:.6f}, expected {expected:.6f} ± {tolerance}. "
        "Check your marginalization logic in enumerate_all."
    )
    
    # Also check that probabilities sum to 1
    assert abs(result[True] + result[False] - 1.0) <= 0.001, (
        f"Probabilities should sum to 1.0, got {result[True] + result[False]:.6f}"
    )
    
    print(f"✓ P(C=True) = {prob:.6f} (expected {expected:.6f})")


def test_marginal_d(network: bayesian_network.BayesianNetwork, expected_values: pd.DataFrame) -> None:
    """
    Test marginal probability P(D=True).
    
    Expected: Should marginalize over entire network.
    """
    result = network.query("D")
    prob = result[True]
    
    expected, tolerance = get_expected(expected_values, 'test_marginal_d')
    
    assert abs(prob - expected) <= tolerance, (
        f"P(D=True) = {prob:.6f}, expected {expected:.6f} ± {tolerance}."
    )
    print(f"✓ P(D=True) = {prob:.6f} (expected {expected:.6f})")


def test_causal_reasoning(network: bayesian_network.BayesianNetwork, expected_values: pd.DataFrame) -> None:
    """
    Test causal reasoning: P(C=True | A=True).
    
    Expected: Given a cause, predict the effect.
    This is forward/predictive inference.
    """
    result = network.query("C", {"A": True})
    prob = result[True]
    
    expected, tolerance = get_expected(expected_values, 'test_causal_c_given_a')
    
    assert abs(prob - expected) <= tolerance, (
        f"P(C=True | A=True) = {prob:.6f}, expected {expected:.6f} ± {tolerance}. "
        "Check causal reasoning (cause → effect)."
    )
    print(f"✓ P(C=True | A=True) = {prob:.6f} (expected {expected:.6f})")


def test_diagnostic_reasoning(network: bayesian_network.BayesianNetwork, expected_values: pd.DataFrame) -> None:
    """
    Test diagnostic reasoning: P(A=True | C=True).
    
    Expected: Given an effect, infer the cause.
    This is backward/abductive inference.
    """
    result = network.query("A", {"C": True})
    prob = result[True]
    
    expected, tolerance = get_expected(expected_values, 'test_diagnostic_a_given_c')
    
    assert abs(prob - expected) <= tolerance, (
        f"P(A=True | C=True) = {prob:.6f}, expected {expected:.6f} ± {tolerance}. "
        "Check diagnostic reasoning (effect → cause)."
    )
    print(f"✓ P(A=True | C=True) = {prob:.6f} (expected {expected:.6f})")


def test_explaining_away(network: bayesian_network.BayesianNetwork, expected_values: pd.DataFrame) -> None:
    """
    Test explaining away: P(A=True | C=True, B=True).
    
    Expected: When B is known to cause C, A becomes less likely.
    This is intercausal reasoning - a key feature of Bayesian networks.
    
    P(A | C, B) < P(A | C) because B "explains away" C.
    """
    result = network.query("A", {"C": True, "B": True})
    prob = result[True]
    
    expected, tolerance = get_expected(expected_values, 'test_explaining_away')
    
    assert abs(prob - expected) <= tolerance, (
        f"P(A=True | C=True, B=True) = {prob:.6f}, expected {expected:.6f} ± {tolerance}. "
        "Check explaining away logic (intercausal reasoning)."
    )
    
    # Also check that explaining away actually happens
    prob_without_b = network.query("A", {"C": True})[True]
    assert prob < prob_without_b, (
        f"Explaining away failed: P(A|C,B)={prob:.4f} should be < P(A|C)={prob_without_b:.4f}. "
        "When B is true, it 'explains away' C, making A less likely."
    )
    
    print(f"✓ P(A=True | C=True, B=True) = {prob:.6f} (expected {expected:.6f})")
    print(f"  P(A=True | C=True) = {prob_without_b:.6f}")
    print(f"  → Difference: {prob_without_b - prob:.6f} (B explains away C)")


if __name__ == "__main__":
    # Allow running directly for quick debugging
    pytest.main([__file__, "-v"])
