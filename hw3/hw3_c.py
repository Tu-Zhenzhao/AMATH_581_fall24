import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

# Parameters
L = 2  # Domain limit
dx = 0.1  # Grid spacing
xspan = np.arange(-L, L + dx, dx)
K = 1
tol = 1e-5
gamma_values = [0.05, -0.05]  # Values of gamma

# Function to define the ODE system
def func(y, x, K, epsilon, gamma):
    phi = y[0]
    phi_prime = y[1]
    phi_abs_sq = phi**2  # Since we have |phi|^2
    phi_double_prime = (gamma * phi_abs_sq + K * x**2 - epsilon) * phi
    return [phi_prime, phi_double_prime]

# Arrays to store results
A5 = []  # Eigenfunctions for gamma = 0.05
A6 = []  # Eigenvalues for gamma = 0.05
A7 = []  # Eigenfunctions for gamma = -0.05
A8 = []  # Eigenvalues for gamma = -0.05

# Loop over gamma values
for gamma in gamma_values:
    eigvals = []
    eigfuncs = []
    epsilon_start = 0  # Initial guess for epsilon
    for mode in range(2):  # First two modes
        epsilon = epsilon_start
        depsilon = 1  # Initial step size for epsilon adjustment

        # Convergence loop for each epsilon
        for iteration in range(1000):
            # Initial conditions
            y0 = [1, 0]

            # Solve the ODE
            sol = odeint(func, y0, xspan, args=(K, epsilon, gamma))

            # Compute the error at x = L
            phi_L = sol[-1, 0]
            phi_prime_L = sol[-1, 1]
            err = phi_L  # We want phi(L) = 0

            if np.abs(err) < tol:
                eigvals.append(epsilon)
                break

            # Adjust epsilon based on the error
            if (-1) ** (mode) * err > 0:
                epsilon -= depsilon
            else:
                epsilon += depsilon
                depsilon *= 0.5

        # Update epsilon_start for the next mode
        epsilon_start = epsilon + 0.5

        # Normalize the eigenfunction
        phi_n = sol[:, 0]
        norm = np.sqrt(np.trapz(phi_n**2, xspan))
        phi_n_normalized = np.abs(phi_n) / norm

        eigfuncs.append(phi_n_normalized)

        # Plotting the solution
        plt.plot(xspan, phi_n_normalized, label=f'Mode {mode+1}, ε={epsilon:.4f}, γ={gamma}')

    plt.title(f'Eigenfunctions for γ = {gamma}')
    plt.xlabel('x')
    plt.ylabel('|ϕ_n(x)|')
    plt.legend()
    plt.grid()
    plt.show()

    # Save results to appropriate arrays
    eigenfunctions_matrix = np.column_stack(eigfuncs)
    eigenvalues_vector = np.array(eigvals)

    if gamma == 0.05:
        A5 = eigenfunctions_matrix
        A6 = eigenvalues_vector
    else:
        A7 = eigenfunctions_matrix
        A8 = eigenvalues_vector

# Print the eigenvalues
print("Eigenvalues for γ = 0.05 (A6):", A6)
print("Eigenvalues for γ = -0.05 (A8):", A8)
