# error_analysis.py
# Error Analysis for Homework 2

import numpy as np
from scipy.special import hermite
from scipy.integrate import simps

# Load numerical eigenvalues and eigenfunctions from parts (a) and (b)
# Part (a)
A1_a = np.load('A1_a.npy')      # Eigenfunctions from part (a)
A2_a = np.load('A2_a.npy')      # Eigenvalues from part (a)
xspan_a = np.load('xspan_a.npy')

# Part (b)
A1_b = np.load('A1_b.npy')      # Eigenfunctions from part (b)
A2_b = np.load('A2_b.npy')      # Eigenvalues from part (b)
xspan_b = np.load('xspan_b.npy')

# Parameters
K = 1
n_values = np.arange(5)  # n = 0 to 4

# Compute exact eigenvalues
beta_exact = 2 * n_values + 1  # Since K = 1

# Function to compute exact eigenfunctions
def compute_exact_eigenfunctions(xspan):
    eigenfunctions_exact = []
    for n in n_values:
        Hn = hermite(n)
        psi_n = Hn(xspan) * np.exp(-xspan**2 / 2)
        # Normalize the eigenfunction over the interval
        norm = np.sqrt(simps(psi_n**2, xspan))
        psi_n_normalized = np.abs(psi_n) / norm
        eigenfunctions_exact.append(psi_n_normalized)
    return np.array(eigenfunctions_exact).T  # Shape: [number of x points, 5]

# Compute exact eigenfunctions for parts (a) and (b)
eigenfunctions_exact_a = compute_exact_eigenfunctions(xspan_a)
eigenfunctions_exact_b = compute_exact_eigenfunctions(xspan_b)

# Function to compute the norm of the difference between numerical and exact eigenfunctions
def compute_eigenfunction_error(numerical, exact, xspan):
    error = []
    for i in range(5):
        diff = numerical[:, i] - exact[:, i]
        norm = np.sqrt(simps(diff**2, xspan))
        error.append(norm)
    return np.array(error)

# Compute errors for part (a)
eigenfunction_error_a = compute_eigenfunction_error(A1_a, eigenfunctions_exact_a, xspan_a)
eigenvalue_error_a = 100 * np.abs(A2_a - beta_exact) / beta_exact

# Compute errors for part (b)
eigenfunction_error_b = compute_eigenfunction_error(A1_b, eigenfunctions_exact_b, xspan_b)
eigenvalue_error_b = 100 * np.abs(A2_b - beta_exact) / beta_exact

# Display the errors
print("Eigenfunction Errors for Part (a):", eigenfunction_error_a)
print("Eigenvalue Errors for Part (a) (%):", eigenvalue_error_a)
print("\nEigenfunction Errors for Part (b):", eigenfunction_error_b)
print("Eigenvalue Errors for Part (b) (%):", eigenvalue_error_b)
