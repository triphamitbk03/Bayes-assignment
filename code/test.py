import pandas as pd
from probabilistic_inference import *

def run_hidden_tests():
    print("\n=== Hidden-style tests for Part 1 ===")
    # Test 1
    df = pd.read_csv("/Users/phamnguyenviettri/Ses251/IntroAI/bayesian_inference_assignment/data/probabilistic_inference.csv")
    print(probability_summary(df))

    # Test 2 – edge cases
    df2 = pd.DataFrame({"X": ["A","B"], "Y": ["1","2"]})
    try:
        conditional_probability(df2, {"X":"C"}, {"Y":"1"})
    except ZeroDivisionError:
        print("✅ Correctly handled ZeroDivisionError")

    # Test 3 – generic dataset
    df3 = pd.DataFrame({
        "Smoke": ["Yes","No","Yes","Yes","No"],
        "Cough": ["Yes","No","Yes","No","Yes"],
        "Flu":   ["Yes","No","Yes","No","No"]
    })
    print(bayes_posterior(df3, ("Flu","Yes"), ("Cough","Yes")))

if __name__ == "__main__":
    run_hidden_tests()
