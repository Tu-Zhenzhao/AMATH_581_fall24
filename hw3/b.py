# b.py
# Homework 2 - Part (b)

import numpy as np
import matplotlib.pyplot as plt

# Parameters
L = 4
dx = 0.1
K = 1
xspan = np.arange(-L, L + dx, dx)
N_total = len(xspan)
N = N_total - 2  # Number of interior points
x_i = xspan[1:-1]  # Interior points

# Initialize the matrix A
A = np.zeros((N, N))

# Off-diagonal entries
E = 1 * np.ones(N - 1)

# Main diagonal entries
D = -2 - dx**2 * K * x_i**2

# Construct the tridiagonal matrix A
A = np.diag(D) + np.diag(E, k=-1) + np.diag(E, k=1)

# Modify the first and last rows (boundary conditions)
A[0, 0] += 4/3
A[0, 1] -= 1/3
A[N-1, N-2] -= 1/3
A[N-1, N-1] += 4/3

# Solve the eigenvalue problem
eigenvalues, eigenvectors = np.linalg.eigh(-A)
final_eigenvalues = eigenvalues[:5] / (dx**2)

# Extract the first five eigenvalues and eigenvectors
eigenvalues = eigenvalues[:5]
eigenvectors = eigenvectors[:, :5]

# Include boundary points in eigenfunctions
eigenfunctions = np.zeros([N_total, 5])

# Normalize the eigenfunctions and take absolute values
for i in range(5):
    eigenvector = eigenvectors[:, i]
    # Adding boundary conditions
    phi0 = 4/3 * eigenvector[0] - 1/3 * eigenvector[1]
    phiN = 4/3 * eigenvector[N-1] - 1/3 * eigenvector[N-2]
    # Construct the full eigenfunction
    eigenfunctions[:, i] = np.concatenate(([phi0], eigenvector, [phiN]))
    # Normalize
    norm = np.trapz(eigenfunctions[:, i]**2, xspan)
    eigenfunctions[:, i] /= np.sqrt(norm)
    eigenfunctions[:, i] = np.abs(eigenfunctions[:, i])

# Save the eigenfunctions and eigenvalues
A1 = eigenfunctions  # Eigenfunctions (shape: [number of x points, 5])
A2 = final_eigenvalues  # Eigenvalues (shape: [5])

# Plot the eigenfunctions
for i in range(5):
    plt.plot(xspan, eigenfunctions[:, i], label=f'Eigenvalue {i+1}: {A2[i]:.2f}')
plt.legend()
plt.xlabel('x')
plt.ylabel('Normalized |ϕ_n(x)|')
plt.title('First Five Normalized Eigenfunctions with Modified Boundary Conditions')
plt.show()

# Save the eigenfunctions and eigenvalues
np.save('A1_b.npy', A1)
np.save('A2_b.npy', A2)
np.save('xspan_b.npy', xspan)
