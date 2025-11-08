from __future__ import annotations

import pandas as pd
from pathlib import Path
from typing import Dict, Tuple


DATA_ROOT = Path(__file__).resolve().parents[1] / "data"
DATA_PATH = DATA_ROOT / "probabilistic_inference.csv"


def load_dataset(path: Path = DATA_PATH) -> pd.DataFrame:
    """Load the observational dataset used for probability queries."""
    return pd.read_csv(path)


def joint_probability(df: pd.DataFrame, assignment: Dict[str, str]) -> float:
    """
    Compute the empirical joint probability for a set of variable assignments.

    TODO: Filter the dataframe to rows matching every key/value pair and
    return the relative frequency. Remember to handle the empty assignment case.
    """
    raise NotImplementedError("Implement joint_probability.")


def marginal_probability(df: pd.DataFrame, variable: str, value: str) -> float:
    """
    Compute the marginal probability P(variable = value).

    TODO: Reuse joint_probability to keep the implementation consistent.
    """
    raise NotImplementedError("Implement marginal_probability.")


def conditional_probability(
    df: pd.DataFrame,
    query: Dict[str, str],
    given: Dict[str, str],
) -> float:
    """
    Compute P(query | given) using empirical frequencies.

    TODO: Use the joint probability for the combined assignment divided by the
    probability of the conditioning event. Validate that the denominator is not zero.
    """
    raise NotImplementedError("Implement conditional_probability.")


def bayes_posterior(
    df: pd.DataFrame,
    hypothesis: Tuple[str, str],
    evidence: Tuple[str, str],
) -> float:
    """
    Compute P(hypothesis | evidence) using Bayes' rule.

    TODO: Combine the helper functions above:
      1. Estimate the likelihood P(evidence | hypothesis)
      2. Estimate the prior P(hypothesis)
      3. Estimate the evidence probability P(evidence)
      4. Return (likelihood * prior) / evidence_probability
    """
    raise NotImplementedError("Implement bayes_posterior.")


def probability_summary(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build a summary table with at least three probability queries and one posterior.

    TODO: Choose the queries listed in the project brief (or similar) and
    compute each using the helper functions. Return a dataframe with columns
    ['Query', 'Probability'] rounded to four decimals.
    """
    raise NotImplementedError("Implement probability_summary.")


def main() -> None:
    df = load_dataset()
    summary = probability_summary(df)
    print("=== Probability Summary ===")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()

