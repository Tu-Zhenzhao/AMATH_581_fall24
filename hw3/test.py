import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import root_scalar
import matplotlib.pyplot as plt

# Parameters
L = 2
K = 1
xspan = np.linspace(0, L, 41)  # From 0 to L with 41 points (step size 0.1)
gammas = [0.05, -0.05]  # Gamma values
modes = ['even', 'odd']  # Modes to compute

def nonlinear_ode(x, y, gamma, epsilon):
    """
    Defines the nonlinear differential equation.
    """
    y1, y2 = y
    dy1dx = y2
    dy2dx = - (epsilon - gamma * np.abs(y1)**2 - K * x**2) * y1
    return [dy1dx, dy2dx]

def shoot(gamma, mode, epsilon_guess):
    """
    Performs the shooting method to find the eigenvalue epsilon.
    """
    # Initial conditions based on mode
    if mode == 'even':
        y0 = [1.0, 0.0]  # y(0) = 1, y'(0) = 0
    elif mode == 'odd':
        y0 = [0.0, 1.0]  # y(0) = 0, y'(0) = 1
    else:
        raise ValueError("Mode must be 'even' or 'odd'.")

    # Objective function for root finding
    def objective(epsilon):
        sol = solve_ivp(
            nonlinear_ode, [0, L], y0, args=(gamma, epsilon),
            t_eval=xspan, rtol=1e-6, atol=1e-8
        )
        y_end = sol.y[0, -1]  # y at x = L
        return y_end

    # Find epsilon using root finding
    sol = root_scalar(
        objective, bracket=[epsilon_guess - 5, epsilon_guess + 5],
        method='bisect', xtol=1e-6
    )

    if not sol.converged:
        raise RuntimeError("Root finding did not converge.")

    epsilon_found = sol.root

    # Solve ODE with found epsilon
    sol = solve_ivp(
        nonlinear_ode, [0, L], y0, args=(gamma, epsilon_found),
        t_eval=xspan, rtol=1e-6, atol=1e-8
    )

    # Construct full solution using symmetry
    if mode == 'even':
        x_full = np.concatenate((-sol.t[::-1], sol.t[1:]))
        y_full = np.concatenate((sol.y[0][::-1], sol.y[0][1:]))
    else:
        x_full = np.concatenate((-sol.t[::-1], sol.t[1:]))
        y_full = np.concatenate((-sol.y[0][::-1], sol.y[0][1:]))

    # Normalize the eigenfunction
    norm = np.trapz(np.abs(y_full)**2, x_full)
    y_normalized = np.abs(y_full) / np.sqrt(norm)

    return epsilon_found, x_full, y_normalized

# Main computation
for gamma in gammas:
    print(f"Computing for gamma = {gamma}")
    eigenvalues = []
    eigenfunctions = []

    for mode, epsilon_guess in zip(modes, [1.0, 4.0]):
        epsilon, x, y_norm = shoot(gamma, mode, epsilon_guess)
        eigenvalues.append(epsilon)
        eigenfunctions.append(y_norm)

        # Plotting the eigenfunction
        plt.plot(x, y_norm, label=f'{mode.capitalize()} Mode, ε = {epsilon:.6f}')

    plt.title(f'Normalized Modes for γ = {gamma}')
    plt.xlabel('x')
    plt.ylabel('Normalized |ϕ|')
    plt.legend()
    plt.grid(True)
    plt.show()

    # Save the eigenfunctions and eigenvalues
    eigenfunctions_matrix = np.column_stack(eigenfunctions)
    eigenvalues_vector = np.array(eigenvalues)

    print("Eigenvalues:", eigenvalues_vector)
    print("Eigenfunctions matrix shape:", eigenfunctions_matrix.shape)
