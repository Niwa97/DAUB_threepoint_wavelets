import numpy as np
from scipy.optimize import minimize, Bounds
#Putting f = 1

def objective(X):
    """Calculates the sum of squares of the coefficients [a, b, c, d, e]."""
    return np.sum(X**2)


def constraint_c3(X):
    """Constraint C3: -a + b - c + d - e + 1 = 0"""
    a, b, c, d, e = X
    return -a + b - c + d - e + 1

def constraint_c4(X):
    """Constraint C4: -5a + 4b - 3c + 2d - e = 0"""
    a, b, c, d, e = X
    return -5*a + 4*b - 3*c + 2*d - e

constraints = [
    {'type': 'eq', 'fun': constraint_c3},
    {'type': 'eq', 'fun': constraint_c4}
]

x0 = np.array([1.0, 1.0, 1.0, 1.0, 1.0])

result = minimize(
    objective,
    x0,
    method='SLSQP',
    constraints=constraints,
    tol=1e-10
)

if not result.success:
    print("\nOptimization failed")
    print("Reason:", result.message)
    exit()

X_optimal = result.x
a, b, c, d, e = X_optimal
f = 1.0

V = np.array([a, b, c, d, e, f])
print("Numerical Estimation Results")
print(f"Optimization Status: {result.message}")
print(f"Minimized Sum of Squares (a^2+...+e^2): {result.fun:.6f}")
print("Coefficients found:")
print(f"a = {a:.6f}")
print(f"b = {b:.6f}")
print(f"c = {c:.6f}")
print(f"d = {d:.6f}")
print(f"e = {e:.6f}")
print(f"f = {f:.6f} (Fixed)")


print("Constraint Verification")
C3_check = constraint_c3(X_optimal)
C4_check = constraint_c4(X_optimal)
print(f"C3 Check (-a + b - c + d - e + f): {C3_check:.2e}")
print(f"C4 Check (-5a + 4b - 3c + 2d - e): {C4_check:.2e}")

M = np.array([
    [V[0], V[1], V[2], V[3], V[4], V[5]],
    [V[5], -V[4], V[3], -V[2], V[1], -V[0]],
    [V[0], -V[1], V[2], -V[3], V[4], -V[5]]
])

rank_M = np.linalg.matrix_rank(M)

print("Matrix Rank Verification")
print(np.round(M, 10))
print(f"Calculated Rank(M): {rank_M}")

if rank_M == 3:
    print("The matrix M has the rank equal to 3")
else:
    print(f"The matrix M has a rank of {rank_M}")

total_sum_sq = np.sum(V**2)
print(f"Total sum of squares (a^2 + ... + f^2): {total_sum_sq:.6f}")
