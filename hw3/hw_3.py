import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import eigh_tridiagonal

# Parameters
L = 4  # Domain limit
dx = 0.1  # Grid spacing
K = 1  # Constant in the differential equation

# Spatial grid
xspan = np.arange(-L, L + dx, dx)
N = len(xspan)
x = xspan

# Potential term
V = K * x**2

# Construct the tridiagonal matrix H
# Main diagonal
d = np.zeros(N)
# Off-diagonal
e = np.ones(N - 1) * (-1) / dx**2

# Fill the main diagonal with appropriate values
for i in range(N):
    if i == 0:
        # Forward differencing at the first point
        # Adjust diagonal element using the hint
        approx_epsilon = K * L**2
        d[i] = (3 + 2 * dx * np.sqrt(K * L**2 - approx_epsilon)) / dx**2 + V[i]
    elif i == N - 1:
        # Backward differencing at the last point
        # Adjust diagonal element using the hint
        approx_epsilon = K * L**2
        d[i] = (3 + 2 * dx * np.sqrt(K * L**2 - approx_epsilon)) / dx**2 + V[i]
    else:
        # Interior points
        d[i] = 2 / dx**2 + V[i]

# Adjust the first and last elements due to boundary conditions
# Since ψ(−L) = ψ(L) = 0, we exclude the first and last points
d_interior = d[1:-1]
e_interior = e[1:-1]

# Solve the eigenvalue problem
eigenvalues, eigenvectors = eigh_tridiagonal(d_interior, e_interior)

# Select the first five eigenvalues and eigenvectors
epsilon_n = eigenvalues[:5]
psi_n = eigenvectors[:, :5]

# Include boundary points (set to zero due to boundary conditions)
psi_n_full = np.zeros((N, 5))
psi_n_full[1:-1, :] = psi_n

# Normalize the eigenfunctions
phi_n = np.zeros_like(psi_n_full)
for i in range(5):
    norm = np.sqrt(np.trapz(np.abs(psi_n_full[:, i])**2, x))
    phi_n[:, i] = psi_n_full[:, i] / norm

# Take the absolute value of the eigenfunctions
phi_n_abs = np.abs(phi_n)

# Save the eigenfunctions and eigenvalues
A1 = phi_n_abs  # Eigenfunctions matrix (columns correspond to φₙ)
A2 = epsilon_n  # Eigenvalues vector

# Print the eigenvalues
print("Eigenvalues (εₙ):", A2)
# print the matrix
print("Matrix e", d)
print("D interior", d_interior)
print("E interior", e_interior)

# Plot the eigenfunctions
plt.figure(figsize=(10, 6))
for i in range(5):
    plt.plot(x, phi_n[:, i], label=f'ϕ_{i+1} (ε={epsilon_n[i]:.4f})')
plt.title('First Five Normalized Eigenfunctions of the Harmonic Oscillator')
plt.xlabel('x')
plt.ylabel('ϕₙ(x)')
plt.legend()
plt.grid()
plt.show()
