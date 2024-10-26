import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

# parameters 
# Parameters
K = 1
gamma_values = [0.05, -0.05]
L = 2
xspan = np.linspace(-L, L, int(2*L/0.1)+1)  # x from -L to L with step size 0.1
modes = [0, 1]  # First two modes

# Define the differential equation
def func(y, x, K, gamma, epsilon):
    return [y[1], (gamma * y[0]**2 + K * x**2 - epsilon) * y[0]]

# Eigenvalue list
eigvals = []

# Eigenfunction list
eigfuncs = []

# Loop over gamma values
for gamma in gamma_values:
    # Loop over modes
    for mode in modes:
        A = 0.1  # Initial guess for A
        epsilon = 1.0  # Initial guess for epsilon
        depsilon = 1  # Initial step size for epsilon adjustment

        # Convergence loop for each epsilon
        for i in range(1000):
            y0 = [A, A * np.sqrt(K * L**2 - epsilon)]

            # Solve the ODE system
            sol = odeint(func, y0, xspan, args=(K, gamma, epsilon))

            # Compute the error at x = L
            phi_L = sol[-1, 0]
            phi_prime_L = sol[-1, 1]
            err = phi_prime_L + np.sqrt(K * L**2 - epsilon) * phi_L

            if np.abs(err) < 1e-5:
                break

            # Adjust epsilon based on the error
            if (-1) ** mode * err > 0:
                epsilon -= depsilon
            else:
                epsilon += depsilon
                depsilon *= 0.5

        epsilon += 0.1

        # Normalization for eigenfunction
        norm = np.trapz(sol[:, 0] * sol[:, 0], xspan)
        eigfuncs.append(np.abs(sol[:, 0]) / np.sqrt(norm))

        # Determine if the eigenfunction is normalized
        if (np.sum(eigfuncs[-1]) - 1)<1e-5:
            eigvals.append(epsilon)
            print(f'Sum of eigenfunction for γ = {gamma}: {np.sum(eigfuncs[-1])}')
        else:
            print(f'Sum of eigenfunction for γ = {gamma}: {np.sum(eigfuncs[-1])}')
            


    # print the first two eigenvalues
    print(f'Eigenvalues for γ = {gamma}: {eigvals[-2:]}')
    # Plotting the eigenfunction
    plt.plot(xspan, eigfuncs[-1])
    plt.title(f'Eigenfunction for γ = {gamma}')
    plt.xlabel('x')
    plt.ylabel('y(x)')
    plt.grid(True)
    plt.show()


