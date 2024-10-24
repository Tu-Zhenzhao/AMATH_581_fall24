import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import eigh_tridiagonal

def print_step(description, variables=None):
    """Helper function to print descriptions and current variable states."""
    print("\n" + "=" * 50)
    print(description)
    if variables:
        for var_name, value in variables.items():
            print(f"{var_name}: {value}")
    print("=" * 50)

# Step 1: Define parameters and the spatial grid
L = 4  # Domain limit
dx = 0.1  # Grid spacing

print_step(
    "Step 1: Defining Parameters and the Spatial Grid", 
    {"L (Domain limit)": L, "dx (Grid spacing)": dx}
)

xspan = np.arange(-L, L + dx, dx)  # Grid points from -L to L
N = len(xspan)  # Total number of grid points

print_step(
    "Step 2: Creating the Spatial Grid", 
    {"xspan": xspan, "Number of grid points (N)": N}
)

# Step 3: Define the potential term V(x) = K * x^2
K = 1  # Constant in the equation
V = K * xspan**2  # Potential term

print_step(
    "Step 3: Calculating the Potential Term V(x) = K * x^2", 
    {"K": K, "V": V}
)

# Step 4: Construct the tridiagonal matrix H
# Main diagonal elements (filled later)
d = np.zeros(N)

# Off-diagonal elements
e = np.ones(N - 1) * (-1) / dx**2

print_step(
    "Step 4: Preparing Tridiagonal Matrix H", 
    {"Off-diagonal elements (e)": e}
)

# Step 5: Fill in the main diagonal using boundary and interior conditions
for i in range(N):
    if i == 0 or i == N - 1:
        # Forward and backward differencing at the boundaries
        approx_epsilon = K * L**2
        d[i] = (3 + 2 * dx * np.sqrt(K * L**2 - approx_epsilon)) / dx**2 + V[i]
    else:
        # Interior points
        d[i] = 2 / dx**2 + V[i]

print_step(
    "Step 5: Filling in the Main Diagonal", 
    {"Main diagonal elements (d)": d}
)

# Step 6: Remove boundary points to create the interior matrix
d_interior = d[1:-1]  # Interior main diagonal
e_interior = e[1:-1]  # Interior off-diagonal

print_step(
    "Step 6: Creating the Interior Matrix", 
    {"d_interior": d_interior, "e_interior": e_interior}
)

# Step 7: Solve the eigenvalue problem using eigh_tridiagonal
eigenvalues, eigenvectors = eigh_tridiagonal(d_interior, e_interior)

print_step(
    "Step 7: Solving the Eigenvalue Problem", 
    {"Eigenvalues": eigenvalues[:5], "Eigenvectors (first column)": eigenvectors[:, 0]}
)

# Step 8: Select the first five eigenvalues and eigenvectors
epsilon_n = eigenvalues[:5]
psi_n = eigenvectors[:, :5]

print_step(
    "Step 8: Selecting the First Five Eigenvalues and Eigenvectors", 
    {"First five eigenvalues (epsilon_n)": epsilon_n}
)

# Step 9: Include boundary points and normalize eigenfunctions
psi_n_full = np.zeros((N, 5))
psi_n_full[1:-1, :] = psi_n

# Normalize each eigenfunction
phi_n = np.zeros_like(psi_n_full)
for i in range(5):
    norm = np.sqrt(np.trapz(np.abs(psi_n_full[:, i])**2, xspan))
    phi_n[:, i] = psi_n_full[:, i] / norm

print_step(
    "Step 9: Normalizing the Eigenfunctions", 
    {"Normalized eigenfunctions (phi_n)": phi_n}
)

# Step 10: Take the absolute value of the eigenfunctions
phi_n_abs = np.abs(phi_n)

print_step(
    "Step 10: Taking the Absolute Value of the Eigenfunctions", 
    {"Absolute value of eigenfunctions (phi_n_abs)": phi_n_abs}
)

# Step 11: Save the eigenfunctions and eigenvalues
A1 = phi_n_abs  # Eigenfunctions matrix
A2 = epsilon_n  # Eigenvalues vector

print_step(
    "Step 11: Saving the Results", 
    {"Eigenfunctions matrix (A1)": A1, "Eigenvalues vector (A2)": A2}
)

# Step 12: Plot the first five normalized eigenfunctions
plt.figure(figsize=(10, 6))
for i in range(5):
    plt.plot(xspan, phi_n[:, i], label=f'ϕ_{i+1} (ε={epsilon_n[i]:.4f})')
plt.title('First Five Normalized Eigenfunctions of the Harmonic Oscillator')
plt.xlabel('x')
plt.ylabel('ϕₙ(x)')
plt.legend()
plt.grid()
plt.show()

print_step("Step 12: Plotting the Eigenfunctions", {"Plot": "Displayed"})
