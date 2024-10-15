import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

# Parameters
xspan = np.linspace(-4, 4, 100)
K = 1
tol = 1e-5
col = ['r', 'b', 'g', 'c', 'm']
x0 = [1, 0]  # Start with ϕ_n(0) = 1, dϕ_n(0)/dx = 0
init_beta = 1  # Initial guess for eigenvalue
eigvals = []
eigfuncs = []

# Define the differential equation system
def func(y, x, K, beta):
    return [y[1], (K * x**2 - beta) * y[0]]

# Shooting method to find eigenvalues
for modes in range(5):
    beta = init_beta
    dbeta = 1
    for i in range(1000):
        y = odeint(func, x0, xspan, args=(K, beta))
        if np.abs(y[-1, 0]) < tol:  # Converged to boundary condition
            eigvals.append(beta)
            break
        # Adjust beta based on boundary condition
        if (-1)**(modes + 1) * y[-1, 0] > 0:
            beta -= dbeta
        else:
            beta += dbeta
            dbeta *= 0.5

    # Normalize the eigenfunction
    norm = np.trapz(y[:, 0]**2, xspan)
    eigfuncs.append(y[:, 0] / np.sqrt(norm))
    init_beta = beta + 0.1  # Move to next eigenvalue region

    # Plot the normalized eigenfunction
    plt.plot(xspan, eigfuncs[-1], col[modes], label=f'Eigenvalue {modes+1} = {beta:.4f}')

plt.legend()
plt.show()

# Display eigenvalues
print("Eigenvalues:", eigvals)
