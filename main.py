import numpy as np
from scipy.optimize import minimize, Bounds

N_MAG = np.sqrt(1.0 / 3.0)

def objective(X):
    a, b, c, d, e, f = X
    L1 = -a + b - c + d - e + f
    L2 = -5*a + 4*b - 3*c + 2*d - e
    return L1**2 + L2**2

# -a + b - c + d - e + f = 0
def constraint_l1(X):
    a, b, c, d, e, f = X
    return -a + b - c + d - e + f

# -5a + 4b - 3c + 2d - e = 0
def constraint_l2(X):
    a, b, c, d, e, f = X
    return -5*a + 4*b - 3*c + 2*d - e

def constraint_col_norm_1(X):
    a, _, _, _, _, f = X
    return 2*a**2 + f**2 - 1.0

def constraint_col_norm_2(X):
    _, b, _, _, e, _ = X
    return 2*b**2 + e**2 - 1.0

def constraint_col_norm_3(X):
    _, _, c, d, _, _ = X
    return 2*c**2 + d**2 - 1.0

def constraint_a_plus_f_zero(X):
    return X[0] + X[5]

constraints = [
    {'type': 'eq', 'fun': constraint_l1},
    {'type': 'eq', 'fun': constraint_l2},
    {'type': 'eq', 'fun': constraint_col_norm_1},
    {'type': 'eq', 'fun': constraint_col_norm_2},
    {'type': 'eq', 'fun': constraint_col_norm_3},
    {'type': 'eq', 'fun': constraint_a_plus_f_zero},
]

x0 = np.array([N_MAG, N_MAG, N_MAG, N_MAG, N_MAG, -N_MAG])

bounds = Bounds([-1.0] * 6, [1.0] * 6)

result = minimize(
    objective,
    x0,
    method='trust-constr',
    constraints=constraints,
    bounds=bounds,
    tol=1e-15,
    options={'maxiter': 5000}
)

if not result.success:
    print("Optimization failed.")
    print("Reason:", result.message)
    V = result.x
else:
    V = result.x

a, b, c, d, e, f = V
print("\n--- Numerical Estimation Results ---")
print(f"Optimization Status: {result.message}")
print(f"Total Residual Error (Objective Value): {result.fun:.10e} (Expected near 0)")
print(f"Required Magnitude (1/sqrt(3)): {N_MAG:.10f} (Approximate if a=-f etc.)")
print("Coefficients found:")
print(f"a = {a:.10f}")
print(f"b = {b:.10f}")
print(f"c = {c:.10f}")
print(f"d = {d:.10f}")
print(f"e = {e:.10f}")
print(f"f = {f:.10f}")

L1_check = constraint_l1(V)
L2_check = constraint_l2(V)
print(f"L1 Check (-a + b - c + d - e + f): {L1_check:.2e}")
print(f"L2 Check (-5a + 4b - 3c + 2d - e): {L2_check:.2e}")
print(f"Norm Check 1: {constraint_col_norm_1(V):.2e}")
print(f"Norm Check 2: {constraint_col_norm_2(V):.2e}")
print(f"Norm Check 3: {constraint_col_norm_3(V):.2e}")
print(f"Rank check: {constraint_a_plus_f_zero(V):.2e}")

M = np.array([
    [V[0], V[1], V[2], V[3], V[4], V[5]],
    [V[5], -V[4], V[3], -V[2], V[1], -V[0]],
    [V[0], -V[1], V[2], -V[3], V[4], -V[5]]
])

rank_M = np.linalg.matrix_rank(M)

print(np.round(M, 10))

column_norms = np.linalg.norm(M, axis=0)
print("Column Norms:")
print(np.round(column_norms, 10))


if rank_M == 3 and np.allclose(column_norms, 1.0):
    print("All constraints are satisfied.")
else:
    if rank_M != 3:
        print(f"The matrix M has a rank of {rank_M}.")
    if not np.allclose(column_norms, 1.0):
        print("One or more column norms are not close to 1.0.")