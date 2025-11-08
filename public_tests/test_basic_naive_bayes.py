"""
Public Test Suite - Part 3: Naive Bayes Classifiers

These tests check BASIC functionality only. Passing all tests here does NOT
guarantee full credit. Hidden tests will be more comprehensive and strict.

Run with: pytest public_tests/test_basic_naive_bayes.py -v
"""

from __future__ import annotations

import sys
from pathlib import Path

# Add code directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / "code"))

import pandas as pd
import pytest

import naive_bayes


@pytest.fixture(scope="module")
def gaussian_data() -> pd.DataFrame:
    """Load Gaussian Naive Bayes dataset."""
    data_path = Path(__file__).parent.parent / "data" / "gaussian_nb.csv"
    return pd.read_csv(data_path)


@pytest.fixture(scope="module")
def categorical_data() -> pd.DataFrame:
    """Load Categorical Naive Bayes dataset."""
    data_path = Path(__file__).parent.parent / "data" / "categorical_nb.csv"
    return pd.read_csv(data_path)


def test_gaussian_fit_basic(gaussian_data: pd.DataFrame) -> None:
    """
    Test that Gaussian Naive Bayes fit() computes reasonable parameters.
    
    Expected: Should store means, variances, and priors for each class.
    
    Hint: Group by class label, compute mean and variance for each feature
    """
    model = naive_bayes.GaussianNaiveBayes()
    features = ["Duration", "Temperature", "WhiteBloodCellCount"]
    model.fit(gaussian_data, features, "Diagnosis")
    
    # Check model has necessary attributes
    assert hasattr(model, "classes_"), "Model should have 'classes_' attribute after fit"
    
    # Check both classes are present
    assert "Viral" in model.classes_, "Viral class missing from model"
    assert "Bacterial" in model.classes_, "Bacterial class missing from model"
    
    # Check priors sum to 1
    prior_sum = sum(stats.prior for stats in model.classes_.values())
    assert 0.99 <= prior_sum <= 1.01, (
        f"Priors should sum to 1.0, got {prior_sum:.4f}"
    )
    
    # Check means are reasonable (duration should be 2-6 days)
    viral_duration_mean = model.classes_["Viral"].mean["Duration"]
    assert 1.0 <= viral_duration_mean <= 4.0, (
        f"Viral duration mean {viral_duration_mean:.2f} seems unrealistic. "
        "Expected between 1.0 and 4.0 days."
    )
    
    print(f"✓ Gaussian model fitted successfully")
    print(f"  Viral mean[Duration] = {viral_duration_mean:.2f}")
    priors_dict = {label: stats.prior for label, stats in model.classes_.items()}
    print(f"  Priors: {priors_dict}")


def test_gaussian_prediction_basic(gaussian_data: pd.DataFrame) -> None:
    """
    Test basic Gaussian Naive Bayes prediction.
    
    Expected: Should classify a clearly viral case correctly.
    
    Hint: Compute log P(class) + sum of log P(feature|class) for each class
    """
    model = naive_bayes.GaussianNaiveBayes()
    features = ["Duration", "Temperature", "WhiteBloodCellCount"]
    model.fit(gaussian_data, features, "Diagnosis")
    
    # Clearly viral case: short duration, low temp, low WBC
    sample = {"Duration": 2.5, "Temperature": 99.0, "WhiteBloodCellCount": 6.5}
    prediction = model.predict(sample)
    
    assert prediction == "Viral", (
        f"Expected 'Viral' for {sample}, got '{prediction}'. "
        "A short duration with low temperature and WBC should indicate viral infection."
    )
    
    print(f"✓ Correctly predicted '{prediction}' for clearly viral case")


def test_gaussian_log_proba_basic(gaussian_data: pd.DataFrame) -> None:
    """
    Test that log probabilities are reasonable.
    
    Expected: Log probabilities should be negative and higher for correct class.
    
    Hint: Log probabilities are always <= 0 (since probabilities are <= 1)
    """
    model = naive_bayes.GaussianNaiveBayes()
    features = ["Duration", "Temperature", "WhiteBloodCellCount"]
    model.fit(gaussian_data, features, "Diagnosis")
    
    sample = {"Duration": 2.5, "Temperature": 99.0, "WhiteBloodCellCount": 6.5}
    log_probs = model.predict_log_proba(sample)
    
    # Check structure
    assert "Viral" in log_probs, "Log probabilities should include 'Viral' class"
    assert "Bacterial" in log_probs, "Log probabilities should include 'Bacterial' class"
    
    # Check values are negative (log of probability < 1)
    assert log_probs["Viral"] <= 0, "Log probability must be <= 0"
    assert log_probs["Bacterial"] <= 0, "Log probability must be <= 0"
    
    # Check Viral has higher log prob (less negative) for this sample
    assert log_probs["Viral"] > log_probs["Bacterial"], (
        f"For clearly viral case, expected log P(Viral) > log P(Bacterial). "
        f"Got Viral={log_probs['Viral']:.2f}, Bacterial={log_probs['Bacterial']:.2f}"
    )
    
    print(f"✓ Log probabilities: Viral={log_probs['Viral']:.2f}, Bacterial={log_probs['Bacterial']:.2f}")


def test_categorical_fit_basic(categorical_data: pd.DataFrame) -> None:
    """
    Test that Categorical Naive Bayes fit() computes frequency tables.
    
    Expected: Should count occurrences of each feature value per class.
    
    Hint: Use Laplace smoothing (add alpha to all counts)
    """
    model = naive_bayes.CategoricalNaiveBayes(alpha=1.0)
    features = ["Color", "Shape", "Texture"]
    model.fit(categorical_data, features, "Label")
    
    # Check model has necessary structures
    assert hasattr(model, "class_priors_"), "Model should have 'class_priors_' after fit"
    assert hasattr(model, "conditional_counts_"), "Model should have 'conditional_counts_' after fit"
    assert hasattr(model, "feature_values_"), "Model should have 'feature_values_' after fit"
    
    # Check classes
    assert "Edible" in model.class_priors_, "Edible class missing"
    assert "Poisonous" in model.class_priors_, "Poisonous class missing"
    
    # Check priors sum to 1
    prior_sum = sum(model.class_priors_.values())
    assert 0.99 <= prior_sum <= 1.01, f"Priors should sum to 1.0, got {prior_sum}"
    
    # Check feature values were recorded
    assert "Color" in model.feature_values_, "Color feature values missing"
    assert len(model.feature_values_["Color"]) >= 3, (
        f"Expected at least 3 colors, got {len(model.feature_values_['Color'])}"
    )
    
    print(f"✓ Categorical model fitted successfully")
    print(f"  Classes: {list(model.class_priors_.keys())}")
    print(f"  Feature 'Color' has {len(model.feature_values_['Color'])} unique values")


def test_categorical_prediction_basic(categorical_data: pd.DataFrame) -> None:
    """
    Test basic Categorical Naive Bayes prediction.
    
    Expected: Should classify based on feature value frequencies.
    
    Hint: Compute log P(class) + sum of log P(feature=value|class) for each class
    """
    model = naive_bayes.CategoricalNaiveBayes(alpha=1.0)
    features = ["Color", "Shape", "Texture"]
    model.fit(categorical_data, features, "Label")
    
    # Test a sample (exact prediction depends on data, just check it returns valid label)
    sample = {"Color": "Red", "Shape": "Rounded", "Texture": "Smooth"}
    prediction = model.predict(sample)
    
    assert prediction in ["Edible", "Poisonous"], (
        f"Prediction should be 'Edible' or 'Poisonous', got '{prediction}'"
    )
    
    print(f"✓ Predicted '{prediction}' for {sample}")


def test_categorical_smoothing_effect(categorical_data: pd.DataFrame) -> None:
    """
    Test that Laplace smoothing prevents zero probabilities.
    
    Expected: Even unseen feature values should have non-zero probability.
    
    Hint: With alpha=1.0, all probabilities should be > 0
    """
    model = naive_bayes.CategoricalNaiveBayes(alpha=1.0)
    features = ["Color", "Shape", "Texture"]
    model.fit(categorical_data, features, "Label")
    
    # Use a known value that might be rare
    sample = {"Color": "Brown", "Shape": "Elongated", "Texture": "Rough"}
    log_probs = model.predict_log_proba(sample)
    
    # Check both classes have valid (finite) log probabilities
    assert log_probs["Edible"] > float("-inf"), (
        "Smoothing should prevent -inf log probability. Check Laplace smoothing implementation."
    )
    assert log_probs["Poisonous"] > float("-inf"), (
        "Smoothing should prevent -inf log probability. Check Laplace smoothing implementation."
    )
    
    # Both should be negative but finite
    assert log_probs["Edible"] < 0, "Log probability should be negative"
    assert log_probs["Poisonous"] < 0, "Log probability should be negative"
    
    print(f"✓ Smoothing working: log P(Edible)={log_probs['Edible']:.2f}, "
          f"log P(Poisonous)={log_probs['Poisonous']:.2f}")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

