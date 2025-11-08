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
   #raise NotImplementedError("Implement joint_probability.")
    if not assignment:
        return 1.0
    
    filtered = df.copy()
    for var, val in assignment.items():
        filtered = filtered[filtered[var] == val]

    count = len(filtered)
    total = len(df)

    return count / total


def marginal_probability(df: pd.DataFrame, variable: str, value: str) -> float:
    """
    Compute the marginal probability P(variable = value).

    TODO: Reuse joint_probability to keep the implementation consistent.
    """
    #raise NotImplementedError("Implement marginal_probability.")
    return joint_probability(df, {variable: value})


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
    #raise NotImplementedError("Implement conditional_probability.")
    """ 
    P (query | given) = P(query ∩ given) / P(given)

    steps: 
    1. Combine query and given -> joint event
    2. Compute P(joint event) using joint_probability
    3. Compute P(given) using joint_probability
    4. Return P(joint event) / P(given)
    """
    numerator = joint_probability(df, {**query, **given})
    denominator = joint_probability(df, given)
    if denominator == 0:
        raise ValueError("Denominator in conditional probability is zero.")
    return numerator / denominator



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
    #raise NotImplementedError("Implement bayes_posterior.")
        # formula: P(H|E) = P(E|H) * P(H) / P(E)
    # 1. Likelihood P(E|H)
    e_var, e_val = evidence
    h_var, h_val = hypothesis
    likelihood = conditional_probability(
        df,
        query={e_var: e_val},
        given={h_var: h_val}
    )
    # 2. Prior P(H)
    prior = marginal_probability(df, h_var, h_val)
    # 3. Evidence P(E)
    evidence_prob = marginal_probability(df, e_var, e_val)
    if evidence_prob == 0:
        raise ValueError("Evidence probability is zero.")
    # 4. Posterior P(H|E)
    posterior = (likelihood * prior) / evidence_prob
    return posterior


def probability_summary(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build a summary table with at least three probability queries and one posterior.

    TODO: Choose the queries listed in the project brief (or similar) and
    compute each using the helper functions. Return a dataframe with columns
    ['Query', 'Probability'] rounded to four decimals.
    """
    #raise NotImplementedError("Implement probability_summary.")
    results = []

    # 1. Marginal Probability P(Fever = Yes)
    p_fever_yes = marginal_probability(df, "Fever", "Yes")
    results.append(("P(Fever=Yes)", p_fever_yes))

    # 2. Joint Probability P(Fever = Yes, Cough = Yes)
    p_fever_cough = joint_probability(df, {"Fever": "Yes", "Cough": "Yes"})
    results.append(("P(Fever=Yes, Cough=Yes)", p_fever_cough))

    # 3. Conditional Probability P(Cough = Yes | Fever = Yes)
    p_cough_given_fever = conditional_probability(
        df,
        query={"Cough": "Yes"},
        given={"Fever": "Yes"}
    )
    results.append(("P(Cough=Yes | Fever=Yes)", p_cough_given_fever))

    # 4. Posterior Probability P(Fever = Yes | TestPositive = Positive)
    p_fever_given_test = bayes_posterior(
        df,
        hypothesis=("Fever", "Yes"),
        evidence=("TestPositive", "Positive")
    )
    results.append(("P(Fever=Yes | TestPositive=Positive)", p_fever_given_test))

    #5. Return as DataFrame
    summary_df = pd.DataFrame(results, columns=["Query", "Probability"])
    summary_df["Probability"] = summary_df["Probability"].round(4)

    return summary_df


def main() -> None:
    df = load_dataset()
    summary = probability_summary(df)
    print("=== Probability Summary ===")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()

