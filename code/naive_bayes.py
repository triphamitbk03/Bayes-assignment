from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List

import pandas as pd

DATA_ROOT = Path(__file__).resolve().parents[1] / "data"


@dataclass
class GaussianClassStats:
    mean: Dict[str, float]
    variance: Dict[str, float]
    prior: float


class GaussianNaiveBayes:
    def __init__(self, epsilon: float = 1e-6) -> None:
        self.epsilon = epsilon
        self.classes_: Dict[str, GaussianClassStats] = {}
        self.features_: List[str] = []

    def fit(self, df: pd.DataFrame, feature_cols: Iterable[str], label_col: str) -> None:
        """
        Estimate per-class means, variances, and priors.

        TODO: Group rows by the class label and compute:
          - Mean for each feature
          - Variance (add epsilon to avoid division by zero)
          - Prior probability
        Store the results in self.classes_ and keep track of feature columns.
        """
        raise NotImplementedError("Implement GaussianNaiveBayes.fit.")

    def _log_gaussian(self, x: float, mean: float, variance: float) -> float:
        """Utility to compute log N(x | mean, variance)."""
        return -0.5 * math.log(2 * math.pi * variance) - ((x - mean) ** 2) / (2 * variance)

    def predict_log_proba(self, sample: Dict[str, float]) -> Dict[str, float]:
        """
        Return the log-probability for each class given a numeric sample.

        TODO: Sum the log prior with the log-likelihood contribution from every feature.
        """
        raise NotImplementedError("Implement GaussianNaiveBayes.predict_log_proba.")

    def predict(self, sample: Dict[str, float]) -> str:
        """Return the most probable class label."""
        # TODO: Use predict_log_proba to determine the most likely class label.
        raise NotImplementedError("Implement GaussianNaiveBayes.predict.")


class CategoricalNaiveBayes:
    def __init__(self, alpha: float = 1.0) -> None:
        self.alpha = alpha
        self.feature_values_: Dict[str, List[str]] = {}
        self.class_priors_: Dict[str, float] = {}
        self.conditional_counts_: Dict[str, Dict[str, Dict[str, int]]] = {}
        self.class_totals_: Dict[str, int] = {}

    def fit(self, df: pd.DataFrame, feature_cols: Iterable[str], label_col: str) -> None:
        """
        Estimate class priors and smoothed frequency tables.

        TODO:
          - Track the unique values for each feature
          - Count occurrences of each value per class
          - Store class counts for smoothing
        """
        raise NotImplementedError("Implement CategoricalNaiveBayes.fit.")

    def predict_log_proba(self, sample: Dict[str, str]) -> Dict[str, float]:
        """
        Return the log-probability for each class given a categorical sample.

        TODO: Apply Laplace smoothing with parameter alpha when computing
        conditional probabilities.
        """
        raise NotImplementedError("Implement CategoricalNaiveBayes.predict_log_proba.")

    def predict(self, sample: Dict[str, str]) -> str:
        """Return the most probable class label."""
        # TODO: Use predict_log_proba to determine the most likely class label.
        raise NotImplementedError("Implement CategoricalNaiveBayes.predict.")


def gaussian_demo() -> pd.DataFrame:
    """
    Train the Gaussian model and evaluate three test cases.

    TODO: Implement the same workflow described in the solution overview:
      - Load gaussian_nb.csv
      - Fit the model
      - Predict three sample cases (include their feature values)
      - Return a dataframe with predictions and log-probabilities
    """
    raise NotImplementedError("Implement gaussian_demo.")


def categorical_demo() -> pd.DataFrame:
    """
    Train the categorical model and evaluate three test cases.

    TODO: Follow the instructions from the project brief using categorical_nb.csv.
    """
    raise NotImplementedError("Implement categorical_demo.")


def main() -> None:
    gaussian_results = gaussian_demo()
    categorical_results = categorical_demo()
    print("=== Gaussian Naive Bayes Predictions ===")
    print(gaussian_results.to_string(index=False))
    print("\n=== Categorical Naive Bayes Predictions ===")
    print(categorical_results.to_string(index=False))


if __name__ == "__main__":
    main()

