import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# Parameters
K = 1
E = 1   # Given energy level εn = 1
L = 2
x_span = [-L, L]
y0 = [1, np.sqrt(K * L**2 - E)]  # Initial conditions: φ = 1, φx = sqrt(KL^2 - 1)

# ODE function representing the Quantum Harmonic Oscillator
def hw1_rhs_a(x, y, E):
    return [y[1], (K * x**2 - E) * y[0]]

# Methods to use
methods = ['RK45', 'RK23', 'Radau', 'BDF']

# Tolerances
tolerances = [1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9, 1e-10]

# Dictionary to store average step sizes for each method
avg_step_sizes = {method: [] for method in methods}

# Loop over methods
for method in methods:
    # Loop over tolerances
    for tol in tolerances:
        # Solve the ODE
        sol = solve_ivp(hw1_rhs_a, x_span, y0, method=method, args=(E,), rtol=tol, atol=tol)
        # Compute average step size
        steps = np.diff(sol.t)
        avg_step = np.mean(steps)
        # Store the average step size
        avg_step_sizes[method].append(avg_step)

# Now, compute logs and fit lines
slopes = []

plt.figure()
for method in methods:
    avg_steps = avg_step_sizes[method]
    log_avg_steps = np.log10(avg_steps)
    log_tols = np.log10(tolerances)
    # Fit a line to the data
    slope, intercept = np.polyfit(log_tols, log_avg_steps, 1)
    slopes.append(slope)
    # Plot the data and the fitted line
    plt.plot(log_tols, log_avg_steps, 'o-', label=f'{method} (slope={slope:.2f})')
    plt.plot(log_tols, slope * log_tols + intercept, '--')

plt.xlabel('log10(Tolerance)')
plt.ylabel('log10(Average Step Size)')
plt.legend()
plt.title('Log-Log plot of Average Step Size vs Tolerance')
plt.grid(True)
plt.show()

# Now, print the slopes and compute the estimated order p
print("Computed Slopes and Estimated Orders:")
for i, method in enumerate(methods):
    m = slopes[i]
    p = (1 / m) - 1
    print(f"Method: {method}, Slope: {m:.4f}, Estimated Order p: {p:.2f}")

# Save the slopes in a 4x1 vector
slopes_array = np.array(slopes).reshape((4, 1))
print("\nSlopes array:")
print(slopes_array)
