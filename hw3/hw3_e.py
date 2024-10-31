import numpy as np
from scipy.integrate import simps



# Parameters
K = 1
L = 4                     
xp = [-L, L] 
xshoot = np.linspace(xp[0], xp[1], 81)
n = len(xshoot)

# hardcode H
h = np.array([np.ones_like(xshoot), 2*xshoot, 4*xshoot**2-2 , 8*xshoot**3-12*xshoot, 16 * xshoot**4 - 48 * xshoot**2 + 12])


# Calculate exact eigenfunctions
phi = np.zeros([n, 5])
for j in range(5):
    # Fixed: Use j instead of n in calculate_hermite
    phi[:, j] = (np.exp(-xshoot**2/2) * h[j] / np.sqrt(2**j * np.math.factorial(j) * np.sqrt(np.pi))).T


# Initialize error arrays                                                 
err_psi_a = np.zeros(5)
err_psi_b = np.zeros(5)
err_a = np.zeros(5)
err_b = np.zeros(5)

# Load numerical solutions (from part (a) and (b))
eigvec_a = np.load('A1_a.npy')  # Eigenfunctions from part (a)
eigval_a = np.load('A2_a.npy')  # Eigenvalues from part (a)

eigvec_b = np.load('A1_b.npy')  # Eigenfunctions from part (b)
eigval_b = np.load('A2_b.npy')  # Eigenvalues from part (b)

# Calculate errors
for j in range(5):
    # Fixed: Changed xspan to xshoot
    err_psi_a[j] = simps((np.abs(eigvec_a[:, j]) - np.abs(phi[:, j]))**2, xshoot)
    err_psi_b[j] = simps((np.abs(eigvec_b[:, j]) - np.abs(phi[:, j]))**2, xshoot)
    err_a[j] = 100 * np.abs(eigval_a[j] - (2*j + 1)) / (2*j + 1)  # Fixed: Changed (2*j-1) to (2*j+1)
    err_b[j] = 100 * np.abs(eigval_b[j] - (2*j + 1)) / (2*j + 1)  # Fixed: Changed (2*j-1) to (2*j+1)

A10 = err_psi_a
A11 = err_a
A12 = err_psi_b  # Fixed: Changed err_psi_a to err_psi_b
A13 = err_b

print("Error in wavefunctions (method a):", A10)
print("Error in eigenvalues (method a):", A11)
print("Error in wavefunctions (method b):", A12)
print("Error in eigenvalues (method b):", A13)
