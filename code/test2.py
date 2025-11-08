import json
from bayesian_network import BayesianNetwork

# ============================================================
# 1️⃣ Tải mô hình mạng Bayesian từ file JSON
# ============================================================

print("\n=== LOADING NETWORK ===")
bn = BayesianNetwork("/Users/phamnguyenviettri/Ses251/IntroAI/bayesian_inference_assignment/data/bayesian_network.json")

print("Nodes order:", bn.order)
print("Available CPT keys:", list(bn.cpt.keys()))
print("Structure loaded successfully!\n")

# ============================================================
# 2️⃣ Kiểm tra hàm get_probability()
# ============================================================

print("=== TESTING get_probability() ===")

# Node A: không có parent
p_a_true = bn.get_probability("A", True, {})
p_a_false = bn.get_probability("A", False, {})
print(f"P(A=True) = {p_a_true:.2f}")
print(f"P(A=False) = {p_a_false:.2f}")

# Node C: có cha là A,B
p_c_true_ab_tf = bn.get_probability("C", True, {"A": True, "B": False})
p_c_false_ab_tf = bn.get_probability("C", False, {"A": True, "B": False})
print(f"P(C=True | A=True, B=False) = {p_c_true_ab_tf:.2f}")
print(f"P(C=False | A=True, B=False) = {p_c_false_ab_tf:.2f}")

# Node D: có cha là C
p_d_true_c_true = bn.get_probability("D", True, {"C": True})
p_d_false_c_true = bn.get_probability("D", False, {"C": True})
print(f"P(D=True | C=True) = {p_d_true_c_true:.2f}")
print(f"P(D=False | C=True) = {p_d_false_c_true:.2f}")

# ============================================================
# 3️⃣ Kiểm tra hàm enumerate_all()
# ============================================================

print("\n=== TESTING enumerate_all() ===")

# Tính xác suất toàn phần của mạng không có bằng chứng
p_all = bn.enumerate_all(bn.order, {})
print(f"Total probability (should be close to 1.0): {p_all:.4f}")

# Tính P(C=True) = Σ_{A,B} P(C|A,B)P(A)P(B)
evidence_c_true = {"C": True}
p_c_true_total = bn.enumerate_all(bn.order, evidence_c_true)
print(f"P(C=True) (from enumeration) = {p_c_true_total:.4f}")

# ============================================================
# 4️⃣ Kiểm tra hàm query()
# ============================================================

print("\n=== TESTING query() ===")

# Marginal: P(C)
result_c = bn.query("C", {})
print("P(C):", result_c)

# Causal: P(C | A=True)
result_c_given_a = bn.query("C", {"A": True})
print("P(C | A=True):", result_c_given_a)

# Diagnostic: P(A | C=True)
result_a_given_c = bn.query("A", {"C": True})
print("P(A | C=True):", result_a_given_c)

# Explaining away: P(A | C=True, B=True)
result_a_given_c_b = bn.query("A", {"C": True, "B": True})
print("P(A | C=True, B=True):", result_a_given_c_b)

# ============================================================
# 5️⃣ Tổng hợp kết quả
# ============================================================

print("\n=== SUMMARY ===")
print(f"P(C=True) ≈ {result_c[True]:.2f}")
print(f"P(C=True | A=True) ≈ {result_c_given_a[True]:.2f}")
print(f"P(A=True | C=True) ≈ {result_a_given_c[True]:.2f}")
print(f"P(A=True | C=True, B=True) ≈ {result_a_given_c_b[True]:.2f}")
print("\n✅ Manual tests for Part 2 completed successfully!")
