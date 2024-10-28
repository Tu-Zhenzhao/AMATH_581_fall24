# a.py
# Homework 2 - Part (a)

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

# Parameters
y0 = 0
xspan = np.linspace(-4, 4, 81)
K = 1
tol = 1e-5
col = ['r', 'b', 'g', 'c', 'm']
init_beta = 1

# Define the differential equation
def func(y, x, K, beta):
    return [y[1], (K * x**2 - beta) * y[0]]

# Lists to store eigenvalues and eigenfunctions
eigvals = []
eigfuncs = []

# Loop through different beta values to find the first 5 eigenvalues and eigenfunctions
beta_start = init_beta

for modes in range(5):
    beta = beta_start
    dbeta = 1  # Initial step size for eigenvalue adjustment
    # Convergence loop for each beta
    for _ in range(1000):
        x0 = [1, np.sqrt(K * 4**2 - beta)]
        # Solve the ODE
        y = odeint(func, x0, xspan, args=(K, beta))
        # Check convergence
        err = y[-1, 1] + np.sqrt(K * 4**2 - beta) * y[-1, 0]
        if np.abs(err) < tol:
            eigvals.append(beta)
            break
        # Adjust beta using the shooting method
        if (-1) ** (modes + 1) * err > 0:
            beta -= dbeta
        else:
            beta += dbeta
            dbeta *= 0.5
    # Update beta_start for the next mode
    beta_start = beta + 0.1
    # Normalize the eigenfunction
    norm = np.trapz(y[:, 0]**2, xspan)
    eigfuncs.append(np.abs(y[:, 0]) / np.sqrt(norm))
    # Plot the eigenfunction
    plt.plot(xspan, y[:, 0] / np.sqrt(norm), col[modes], label=r'$\beta$ = ' + f'{beta:.4f}')

# Convert lists to arrays
A1 = np.array(eigfuncs).T  # Eigenfunctions (shape: [number of x points, 5])
A2 = np.array(eigvals)     # Eigenvalues (shape: [5])

# Plot settings
plt.legend()
plt.title("Eigenfunctions from Part (a)")
plt.xlabel('x')
plt.ylabel('Normalized |ϕ_n(x)|')
plt.show()

# Save the eigenfunctions and eigenvalues
np.save('A1_a.npy', A1)
np.save('A2_a.npy', A2)
np.save('xspan_a.npy', xspan)
