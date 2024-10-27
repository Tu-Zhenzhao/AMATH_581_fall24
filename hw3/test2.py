import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint, simps

# Parameters
K = 1
gamma_values = [0.05, -0.05]
L = 2
xspan = np.linspace(-L, L, int(2 * L / 0.1) + 1)  # x from -L to L with step size 0.1
modes = [0, 1]  # First two modes

# Define the differential equation
def func(y, x, K, gamma, epsilon):
    return [y[1], (gamma * y[0]**2 + K * x**2 - epsilon) * y[0]]

# Eigenvalue and eigenfunction lists
eigvals = []
eigfuncs = []

# Loop over gamma values
for gamma in gamma_values:
    for mode in modes:
        A = 0.1  # Initial guess for A
        dA = 1  # Initial step size for A
        epsilon = 1.0  # Initial guess for epsilon
        depsilon = 1  # Initial step size for epsilon adjustment

        # Convergence loop for epsilon
        for i in range(1000):
            try:
                y0 = [A, A * np.sqrt(K * L**2 - epsilon)]
                sol = odeint(func, y0, xspan, args=(K, gamma, epsilon))
            except ValueError:
                print(f"Skipping invalid initial guess with epsilon = {epsilon}")
                break  # Skip to the next iteration if initial conditions are invalid

            # Error at x = L
            phi_L, phi_prime_L = sol[-1]
            err = phi_prime_L + np.sqrt(K * L**2 - epsilon) * phi_L

            if np.abs(err) < 1e-5:
                eigvals.append(epsilon)
                break  # Converged

            # Adjust epsilon
            if (-1) ** mode * err > 0:
                epsilon -= depsilon
            else:
                epsilon += depsilon
                depsilon *= 0.5

            # Adjust A based on the integral of the solution
            A = simps(sol[:, 0]**2, xspan)
            print(f"epsilon = {epsilon}, A = {A}")
            print(f"eigvals = {eigvals}")
            if (np.abs(A) - 1) < 1e-5:
                break  # Stop if norm is too small

            if A < 1:
                A += dA
            else:
                A -= dA / 2
                dA /= 2

        epsilon += 0.2

        # Only plot and store if an eigenvalue was found
        if len(eigvals) > 0:
            norm = np.trapz(sol[:, 0]**2, xspan)
            eigfuncs.append(sol[:, 0] / np.sqrt(norm))

            plt.plot(xspan, eigfuncs[-1], label=f'{mode} Mode, ε = {eigvals[-1]:.2f}')

    plt.title(f'Eigenfunctions for γ = {gamma}')
    plt.xlabel('x')
    plt.ylabel('Normalized |ϕ|')
    plt.legend()
    plt.grid(True)
    plt.show()
