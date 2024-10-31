import numpy as np
from scipy.integrate import simps

# Define Hermite polynomials H_n(x) using recursion
def hermite_poly(n, x):
    """Compute the Hermite polynomial H_n(x) recursively."""
    if n == 0:
        return np.ones_like(x)  # H_0(x) = 1
    elif n == 1:
        return 2 * x  # H_1(x) = 2x
    else:
        H_nm1 = hermite_poly(n - 1, x)  # H_{n-1}(x)
        H_nm2 = hermite_poly(n - 2, x)  # H_{n-2}(x)
        return 2 * x * H_nm1 - 2 * (n - 1) * H_nm2  # Recurrence relation

# Set parameters
L = 4  # Domain limit for x
xspan = np.linspace(-L, L, 81)  # x values for numerical solutions

# Compute exact Gauss-Hermite eigenfunctions
exact_funcs = []
for n in range(5):
    Hn = hermite_poly(n, xspan)  # Compute H_n(x)
    norm = np.sqrt(simps(Hn**2, xspan))  # Normalize the function
    exact_funcs.append(np.abs(Hn) / norm)
exact_funcs = np.array(exact_funcs).T

# Exact eigenvalues for comparison (n + 0.5 for Hermite polynomials)
exact_eigenvalues = np.array([n + 0.5 for n in range(5)])

# Load numerical solutions (from part (a) and (b))
A1_a = np.load('A1_a.npy')  # Eigenfunctions from part (a)
A2_a = np.load('A2_a.npy')  # Eigenvalues from part (a)

A1_b = np.load('A1_b.npy')  # Eigenfunctions from part (b)
A2_b = np.load('A2_b.npy')  # Eigenvalues from part (b)

# Error computation for eigenfunctions
err_funcs_a = np.zeros(5)
err_funcs_b = np.zeros(5)

for j in range(5):
    diff_a = np.abs(A1_a[:, j] - exact_funcs[:, j])
    diff_b = np.abs(A1_b[:, j] - exact_funcs[:, j])
    
    # Compute L2 norm of the difference
    err_funcs_a[j] = np.sqrt(simps(diff_a**2, xspan))
    err_funcs_b[j] = np.sqrt(simps(diff_b**2, xspan))

# Error computation for eigenvalues (relative percent error)
err_vals_a = 100 * np.abs(A2_a - exact_eigenvalues) / exact_eigenvalues
err_vals_b = 100 * np.abs(A2_b - exact_eigenvalues) / exact_eigenvalues

# Display results
print("Eigenfunction Errors (a):", err_funcs_a)
print("Eigenfunction Errors (b):", err_funcs_b)

print("Eigenvalue Errors (a):", err_vals_a)
print("Eigenvalue Errors (b):", err_vals_b)

# Optional: Plotting the exact and numerical eigenfunctions for comparison
import matplotlib.pyplot as plt

plt.figure(figsize=(12, 6))
for i in range(5):
    plt.plot(xspan, exact_funcs[:, i], label=f'Exact H_{i}(x)')
    plt.plot(xspan, A1_a[:, i], '--', label=f'Numerical (a) Mode {i+1}')
    plt.plot(xspan, A1_b[:, i], ':', label=f'Numerical (b) Mode {i+1}')

plt.legend()
plt.xlabel('x')
plt.ylabel('Normalized |ϕ_n(x)|')
plt.title('Comparison of Exact and Numerical Eigenfunctions')
plt.show()
