"""
Public Test Suite - Part 1: Probabilistic Inference

These tests check BASIC functionality only. Passing all tests here does NOT
guarantee full credit. Hidden tests will be more comprehensive and strict.

Run with: pytest public_tests/test_basic_probabilistic.py -v
"""

from __future__ import annotations

import sys
from pathlib import Path

# Add code directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / "code"))

import pandas as pd
import pytest

import probabilistic_inference


@pytest.fixture(scope="module")
def dataframe() -> pd.DataFrame:
    """Load the probabilistic inference dataset."""
    data_path = Path(__file__).parent.parent / "data" / "probabilistic_inference.csv"
    return pd.read_csv(data_path)


@pytest.fixture(scope="module")
def expected_values() -> pd.DataFrame:
    """Load expected values computed from solution."""
    expected_path = Path(__file__).parent / "expected_probabilistic.csv"
    return pd.read_csv(expected_path)


def test_marginal_probability(dataframe: pd.DataFrame, expected_values: pd.DataFrame) -> None:
    """
    Test basic marginal probability computation.
    
    Expected: P(Ailment=Flu) should be around 0.5 (50% of patients have flu)
    
    Hint: Count rows where Ailment='Flu' and divide by total rows
    """
    prob = probabilistic_inference.marginal_probability(dataframe, "Ailment", "Flu")
    
    # Get expected value from CSV
    expected_row = expected_values[expected_values['test_name'] == 'test_marginal_probability'].iloc[0]
    expected = expected_row['expected_value']
    tolerance = expected_row['tolerance']
    
    assert abs(prob - expected) <= tolerance, (
        f"P(Ailment=Flu) = {prob:.6f}, expected {expected:.6f} ± {tolerance}. "
        "Check your counting logic."
    )
    print(f"✓ P(Ailment=Flu) = {prob:.6f} (expected {expected:.6f})")


def test_joint_probability(dataframe: pd.DataFrame, expected_values: pd.DataFrame) -> None:
    """
    Test basic joint probability computation.
    
    Expected: P(Fever=Yes, Cough=Yes) should be 0.4
    
    Hint: Filter rows where BOTH conditions are true, count, then divide by total
    """
    prob = probabilistic_inference.joint_probability(dataframe, {"Fever": "Yes", "Cough": "Yes"})
    
    # Get expected value from CSV
    expected_row = expected_values[expected_values['test_name'] == 'test_joint_probability'].iloc[0]
    expected = expected_row['expected_value']
    tolerance = expected_row['tolerance']
    
    assert abs(prob - expected) <= tolerance, (
        f"P(Fever=Yes, Cough=Yes) = {prob:.6f}, expected {expected:.6f} ± {tolerance}. "
        "Make sure you're filtering for rows matching ALL conditions."
    )
    print(f"✓ P(Fever=Yes, Cough=Yes) = {prob:.6f} (expected {expected:.6f})")


def test_conditional_probability(dataframe: pd.DataFrame, expected_values: pd.DataFrame) -> None:
    """
    Test basic conditional probability.
    
    Expected: P(TestPositive=Positive | Fever=Yes) ≈ 0.722
    
    Hint: P(A|B) = P(A,B) / P(B)
    """
    prob = probabilistic_inference.conditional_probability(
        dataframe,
        query={"TestPositive": "Positive"},
        given={"Fever": "Yes"},
    )
    
    # Get expected value from CSV
    expected_row = expected_values[expected_values['test_name'] == 'test_conditional_probability'].iloc[0]
    expected = expected_row['expected_value']
    tolerance = expected_row['tolerance']
    
    assert abs(prob - expected) <= tolerance, (
        f"P(TestPositive=Positive | Fever=Yes) = {prob:.6f}, expected {expected:.6f} ± {tolerance}. "
        "Formula: P(A|B) = P(A,B) / P(B)"
    )
    print(f"✓ P(TestPositive=Positive | Fever=Yes) = {prob:.6f} (expected {expected:.6f})")


def test_bayes_posterior(dataframe: pd.DataFrame, expected_values: pd.DataFrame) -> None:
    """
    Test Bayes' rule for posterior probability.
    
    Expected: P(Ailment=Flu | TestPositive=Positive) ≈ 0.857
    
    Hint: P(H|E) = P(E|H) * P(H) / P(E)
    """
    prob = probabilistic_inference.bayes_posterior(
        dataframe,
        hypothesis=("Ailment", "Flu"),
        evidence=("TestPositive", "Positive"),
    )
    
    # Get expected value from CSV
    expected_row = expected_values[expected_values['test_name'] == 'test_bayes_posterior'].iloc[0]
    expected = expected_row['expected_value']
    tolerance = expected_row['tolerance']
    
    assert abs(prob - expected) <= tolerance, (
        f"P(Ailment=Flu | TestPositive=Positive) = {prob:.6f}, expected {expected:.6f} ± {tolerance}. "
        "Bayes' rule: P(H|E) = P(E|H) * P(H) / P(E)"
    )
    print(f"✓ P(Ailment=Flu | TestPositive=Positive) = {prob:.6f} (expected {expected:.6f})")


def test_probability_summary_structure(dataframe: pd.DataFrame) -> None:
    """
    Test that probability_summary returns a properly formatted DataFrame.
    
    Expected: DataFrame with columns ['Query', 'Probability'] and at least 4 rows
    """
    summary = probabilistic_inference.probability_summary(dataframe)
    
    # Check it's a DataFrame
    assert isinstance(summary, pd.DataFrame), (
        f"probability_summary should return a DataFrame, got {type(summary)}"
    )
    
    # Check columns
    assert list(summary.columns) == ["Query", "Probability"], (
        f"Expected columns ['Query', 'Probability'], got {list(summary.columns)}"
    )
    
    # Check we have some queries
    assert len(summary) >= 4, (
        f"Expected at least 4 probability queries, got {len(summary)}"
    )
    
    # Check probabilities are valid (between 0 and 1)
    probs = summary["Probability"].values
    assert all(0 <= p <= 1 for p in probs), (
        "All probabilities must be between 0 and 1. "
        f"Found: {probs}"
    )
    
    print(f"✓ probability_summary returns valid DataFrame with {len(summary)} queries")


if __name__ == "__main__":
    # Allow running directly for quick debugging
    pytest.main([__file__, "-v"])

