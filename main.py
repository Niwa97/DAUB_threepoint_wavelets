import numpy as np
from scipy.optimize import minimize, Bounds

N_MAG = np.sqrt(1.0 / 3.0)

def objective(X):
    a, b, c, d, e, f = X
    L1 = -a + b - c + d - e + f
    L2 = -5*a + 4*b - 3*c + 2*d - e
    return L1**2 + L2**2

def constraint_l1(X):
    a, b, c, d, e, f = X
    return -a + b - c + d - e + f

def constraint_l2(X):
    a, b, c, d, e, f = X
    return -5*a + 4*b - 3*c + 2*d - e


def constraint_row_norm(X):
    return np.sum(X**2) - 1.0

def constraint_ortho_h0_h2(X):
    a, b, c, d, e, f = X
    return a**2 - b**2 + c**2 - d**2 + e**2 - f**2

def constraint_ortho_h1_h2(X):
    a, b, c, d, e, f = X
    return a*f + b*e + c*d

def constraint_a_plus_f_zero(X):
    return X[0] + X[5]

constraints = [
    {'type': 'eq', 'fun': constraint_l1},
    {'type': 'eq', 'fun': constraint_l2},
    {'type': 'eq', 'fun': constraint_row_norm},
    {'type': 'eq', 'fun': constraint_ortho_h0_h2},
    {'type': 'eq', 'fun': constraint_ortho_h1_h2},
    {'type': 'eq', 'fun': constraint_a_plus_f_zero},
]

np.random.seed(42)
x0 = np.array([N_MAG, N_MAG, N_MAG, N_MAG, N_MAG, -N_MAG]) + np.random.uniform(-0.1, 0.1, 6)

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

a, b, c, d, e, f = V
print("\n--- Numerical Estimation Results ---")
print(f"Optimization Status: {result.message}")
print(f"Total Residual Error (Objective Value): {result.fun:.10e}")
print("Coefficients found:")
print(f"a = {a:.10f}")
print(f"b = {b:.10f}")
print(f"c = {c:.10f}")
print(f"d = {d:.10f}")
print(f"e = {e:.10f}")
print(f"f = {f:.10f}")

# Rebuild your user-defined matrix
M = np.array([
    [ a,  b,  c,  d,  e,  f],
    [ f, -e,  d, -c,  b, -a],
    [ a, -b,  c, -d,  e, -f]
])

row_ortho = M @ M.T
print("\nM @ M.T (Checking Row Orthogonality, Should be Identity):")
print(np.round(row_ortho, 10))

if np.allclose(row_ortho, np.eye(3)):
    print("\nSuccess: The 3x6 block has orthonormal rows!")
else:
    print("\nThe rows are not orthonormal.")