# Bayesian Inference Assignment

**Course:** Introduction to Artificial Intelligence  
**Type:** Major assignment  
**Estimated effort:** 15–20 hours

---

## 1. Learning goals

You will practise how to:

1. Compute joint, marginal, conditional, and posterior probabilities from data.
2. Perform exact inference on a fixed Bayesian network using enumeration.
3. Implement Gaussian and Categorical Naive Bayes classifiers from scratch.
4. Validate your code systematically with the provided public tests.

---

## 2. Package contents

```
assignment/
├── README.md                # This document
├── requirements.txt         # Python dependencies (pandas, pytest)
├── data/                    # Datasets – do NOT edit
│   ├── probabilistic_inference.csv
│   ├── gaussian_nb.csv
│   └── categorical_nb.csv
├── code/                    # Your implementation area
│   ├── probabilistic_inference.py
│   ├── bayesian_network.py
│   └── naive_bayes.py
├── local_test.py            # Convenience wrapper around public tests
└── public_tests/            # Basic validation (read-only)
    ├── test_basic_probabilistic.py
    ├── test_basic_bayesian.py
    └── test_basic_naive_bayes.py
```

---

## 3. Environment setup

Requires **Python 3.10 or newer**.

```bash
# from the assignment/ directory
python3 -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
```

All starter code assumes you run commands from the `assignment/` directory.

---

## 4. Working process

1. Open the files in `code/` and read every docstring and TODO. Each function already exposes the exact signature we will import during grading.
2. Implement the TODO blocks. Do **not** change function names, parameters, or return types.
3. After finishing a part, execute:

   ```bash
   # quick wrapper
   python local_test.py --part <n>

   # or run pytest directly
   pytest public_tests/test_basic_<part>.py -v
   ```

   Public tests cover basic behaviour only. Hidden grading tests use stricter tolerances, additional datasets, and property-based checks.

4. Before submission, ensure there are no remaining `raise NotImplementedError` statements and that public tests pass.

---

## 5. Part-by-part requirements

### Part 1 – Probabilistic Inference (`code/probabilistic_inference.py`, 30 pts)

Dataset: `data/probabilistic_inference.csv` (symptom observations). Implement:

- `joint_probability(df, assignment)` – relative frequency of simultaneous assignments.
- `marginal_probability(df, variable, value)` – reuse the joint probability helper.
- `conditional_probability(df, query, given)` – compute `P(query|given)` safely.
- `bayes_posterior(df, hypothesis, evidence)` – apply Bayes’ rule.
- `probability_summary(df)` – build a tidy `DataFrame` of key queries.

### Part 2 – Bayesian Network Inference (`code/bayesian_network.py`, 40 pts)

Network: fixed converging structure A→C←B and C→D defined in `data/bayesian_network.json`.

- `get_probability(node, value, evidence)` – CPT lookup with parent alignment.
- `enumerate_all(variables, evidence)` – recursive enumeration based on AIMA pseudocode.
- `query(variable, evidence)` – compute `P(variable | evidence)` by enumeration and normalisation.

### Part 3 – Naive Bayes Classifiers (`code/naive_bayes.py`, 30 pts)

#### 3A Gaussian Naive Bayes (continuous features)

Dataset: `data/gaussian_nb.csv`

- `GaussianNaiveBayes.fit` – estimate feature means, variances (with epsilon), and class priors.
- `GaussianNaiveBayes.predict_log_proba` – log prior + log Gaussian likelihood per feature.
- `GaussianNaiveBayes.predict` – return argmax class label.
- `gaussian_demo` – optional helper for generating example predictions.

#### 3B Categorical Naive Bayes (discrete features)

Dataset: `data/categorical_nb.csv`

- `CategoricalNaiveBayes.fit` – record value vocabularies, counts, and priors with Laplace smoothing (α = 1.0).
- `CategoricalNaiveBayes.predict_log_proba` – compute log probabilities with smoothing.
- `CategoricalNaiveBayes.predict` – return argmax class label.
- `categorical_demo` – optional helper for generating example predictions.

---

## 6. Testing and grading

- `python local_test.py` – runs all public tests and prints a summary.
- `pytest public_tests -v` – full verbose output.
- Public suite: 18 tests (5 + 7 + 6). Hidden suite: larger coverage, very small tolerances, additional datasets.
- Partial credit is awarded per test case; implement defensively.

---

## 7. Submission checklist

1. `probabilistic_inference.py`, `bayesian_network.py`, and `naive_bayes.py` contain no `NotImplementedError`.
2. `pytest public_tests -v` passes.
3. Code is readable (docstrings, comments where needed, no dead code or print debugging).
4. Zip the following into `{StudentID}_bayesian.zip`:

   ```
   {StudentID}_bayesian/
   ├── probabilistic_inference.py
   ├── bayesian_network.py
   └── naive_bayes.py
   ```

6. Upload the ZIP to the course LMS before the deadline (late policy: -10% per day, up to 2 days).

Do **not** include data files, public tests, or virtual environments.

---

## 8. Academic integrity

This is an individual assignment. Discussing concepts is permitted; sharing code or using external implementations is not. We automatically compare submissions against peers, past cohorts, and public repositories.

---

## 9. Getting help

 - Discuss questions directly with the course lecturer during class sessions.

Good luck, and enjoy exploring probabilistic reasoning! 🎯

---

*Last updated: November 2025*