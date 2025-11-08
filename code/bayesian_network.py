from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

DATA_ROOT = Path(__file__).resolve().parents[1] / "data"


class BayesianNetwork:
    """
    A Bayesian network for probabilistic inference.
    
    Network structure: A→C←B, C→D (converging structure with causal chain)
    """

    def __init__(self, json_file: str = "bayesian_network.json"):
        """Initialize by loading structure and CPT from JSON."""
        json_path = DATA_ROOT / json_file
        
        with open(json_path, 'r') as f:
            data = json.load(f)
        
        self.nodes = data["structure"]["nodes"]
        self.order = data["structure"]["order"]
        self.cpt = data["cpt"]

    def get_probability(self, node: str, value: bool, evidence: Dict[str, bool]) -> float:
        """Look up conditional probability from CPT given parent values."""
        # TODO: Implement CPT lookup
        # Hint 1: Get node_cpt = self.cpt[node] and parents = node_cpt["parents"]
        # Hint 2: If no parents, key = "()", else build tuple from evidence
        # Hint 3: Look up p_true = node_cpt["probabilities"][key]
        # Hint 4: Return p_true if value else (1.0 - p_true)
        #raise NotImplementedError("TODO: Implement get_probability")
        node_cpt = self.cpt[node]
        parents = node_cpt["parents"]
        if not parents:
            key = "()"
        else:
            parent_values = tuple(evidence[parent] for parent in parents)
            key = str(parent_values)
        p_true = node_cpt["probabilities"][key]
        return p_true if value else (1.0 - p_true)

    def enumerate_all(self, variables: List[str], evidence: Dict[str, bool]) -> float:
        """Recursive enumeration algorithm for computing joint probability."""
        # TODO: Implement enumeration algorithm
        # Hint 1: Base case - if not variables: return 1.0
        # Hint 2: Get first = variables[0], rest = variables[1:]
        # Hint 3: If first in evidence:
        #           prob = self.get_probability(first, evidence[first], evidence)
        #           return prob * self.enumerate_all(rest, evidence)
        # Hint 4: Else (marginalize):
        #           Sum over value in [True, False]:
        #             extended = dict(evidence); extended[first] = value
        #             prob = self.get_probability(first, value, extended)
        #             total += prob * self.enumerate_all(rest, extended)
        #           return total
        #raise NotImplementedError("TODO: Implement enumerate_all")
        if not variables:
            return 1.0
        
        y = variables[0]
        rest = variables[1:]

        if y in evidence:
            prob = self.get_probability(y, evidence[y], evidence)
            return prob * self.enumerate_all(rest, evidence)
        else:
            total = 0.0
            for value in [True, False]:
                new_evidence = evidence.copy()
                new_evidence[y] = value
                prob = self.get_probability(y, value, new_evidence)
                total += prob * self.enumerate_all(rest, new_evidence)
            return total

    def query(self, variable: str, evidence: Dict[str, bool] = None) -> Dict[bool, float]:
        """Compute conditional probability P(variable | evidence) via exact enumeration."""
        # TODO: Implement query using enumeration
        # Hint 1: If evidence is None: evidence = {}
        # Hint 2: For each value in [True, False]:
        #           extended = dict(evidence); extended[variable] = value
        #           dist[value] = self.enumerate_all(self.order, extended)
        # Hint 3: Normalize: total = dist[True] + dist[False]
        #         Check if total == 0 (raise ValueError)
        #         Return {value: dist[value] / total for value in dist}
        #raise NotImplementedError("TODO: Implement query")
        dist = {}
        if evidence is None:
            evidence = {}
        for value in [True, False]:
            extended = evidence.copy()
            extended[variable] = value
            dist[value] = self.enumerate_all(self.order, extended)
        total = dist[True] + dist[False]
        if total == 0:
            raise ValueError("Total probability is zero.")
        
        for val in dist:
            dist[val] /= total
            
        return dist

def load_network() -> BayesianNetwork:
    """Load the pre-defined Bayesian network from JSON."""
    return BayesianNetwork("bayesian_network.json")
