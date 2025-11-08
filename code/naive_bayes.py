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
        #raise NotImplementedError("Implement GaussianNaiveBayes.fit.")
        self.features_ = list(feature_cols)
        classes = df[label_col].unique()

        # Calculate stats for each class
        for c in classes:
            subset = df[df[label_col] == c]

            # Calculate mean and variance for each feature
            mean = subset[self.features_].mean().to_dict()
            variance = (subset[self.features_].var(ddof=0) + self.epsilon).to_dict()
            #prior:
            prior = len(subset) / len(df)
            # Store in classes_
            self.classes_[c] = GaussianClassStats(mean=mean, variance=variance, prior=prior)



    def _log_gaussian(self, x: float, mean: float, variance: float) -> float:
        """Utility to compute log N(x | mean, variance)."""
        return -0.5 * math.log(2 * math.pi * variance) - ((x - mean) ** 2) / (2 * variance)

    def predict_log_proba(self, sample: Dict[str, float]) -> Dict[str, float]:
        """
        Return the log-probability for each class given a numeric sample.

        TODO: Sum the log prior with the log-likelihood contribution from every feature.
        """
        #raise NotImplementedError("Implement GaussianNaiveBayes.predict_log_proba.")
        log_probs = {}
        for c, stats in self.classes_.items():
            # Start with log prior
            log_prob = math.log(stats.prior)

            #log-likelihood from each feature
            for feature in self.features_:
                x = sample[feature]
                mean = stats.mean[feature]
                variance = stats.variance[feature]
                log_prob += self._log_gaussian(x, mean, variance)
            
            log_probs[c] = log_prob
        return log_probs

    def predict(self, sample: Dict[str, float]) -> str:
        """Return the most probable class label."""
        # TODO: Use predict_log_proba to determine the most likely class label.
        log_probs = self.predict_log_proba(sample)
        return max(log_probs, key=log_probs.get)


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
        """
        # Lưu các giá trị có thể của từng feature (ví dụ {"Fever": ["Yes","No"], ...})
        for f in feature_cols:
            self.feature_values_[f] = sorted(df[f].unique())

        # Các lớp (class labels)
        classes = df[label_col].unique()

        for cls in classes:
            # Subset theo từng class
            subset = df[df[label_col] == cls]

            # Lưu số lượng mẫu của class này
            self.class_totals_[cls] = len(subset)

            # Tính prior P(y)
            self.class_priors_[cls] = len(subset) / len(df)

            # Khởi tạo dict đếm
            self.conditional_counts_[cls] = {}

            # Với mỗi feature: đếm tần suất từng giá trị
            for f in feature_cols:
                counts = subset[f].value_counts().to_dict()  # đếm từng giá trị
                self.conditional_counts_[cls][f] = counts

        

    def predict_log_proba(self, sample: Dict[str, str]) -> Dict[str, float]:
        """
        Return the log-probability for each class given a categorical sample.
        Apply Laplace smoothing with parameter alpha.
        """
        log_probs = {}

        for cls in self.class_priors_:
            # log P(y)
            logp = math.log(self.class_priors_[cls])

            # Với từng feature: cộng log P(x_i|y)
            for f, value in sample.items():
                # Các giá trị có thể có của feature này
                V = len(self.feature_values_[f])
                N_y = self.class_totals_[cls]

                # Số lần feature = value trong class y
                counts = self.conditional_counts_[cls][f]
                count_val = counts.get(value, 0)

                # Áp dụng smoothing
                prob = (count_val + self.alpha) / (N_y + self.alpha * V)

                logp += math.log(prob)

            log_probs[cls] = logp

        return log_probs


    def predict(self, sample: Dict[str, str]) -> str:
        """Return the most probable class label."""
        # TODO: Use predict_log_proba to determine the most likely class label.
        log_probs = self.predict_log_proba(sample)
        return max(log_probs, key=log_probs.get)


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

