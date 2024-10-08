import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp, simps
from scipy.optimize import bisect

# Constants
L = 4
x_span = np.arange(-L, L + 0.1, 0.1)
K = 1

# Function to solve the differential equation for given ε_n using shooting method
def schrodinger_shooting(ε_n, x_span):
    def schrodinger_ode(x, y):
        # y[0] = φ(x), y[1] = dφ/dx
        d2φ_dx2 = (x**2 - ε_n) * y[0]
        return [y[1], d2φ_dx2]
    
    # Initial conditions: start from one side with small φ and a guessed slope for φ'
    φ0 = [1e-5, 1e-5]  # Initial guess for φ and φ'
    sol = solve_ivp(schrodinger_ode, [x_span[0], x_span[-1]], φ0, t_eval=x_span)
    return sol.y[0]  # Return φ(x)

# Function to find the eigenvalue ε_n
def find_eigenvalue(n):
    def boundary_condition(ε_n):
        φ = schrodinger_shooting(ε_n, x_span)
        return φ[-1]  # The value at x=L (should approach 0)
    
    # Use bisection method to find the eigenvalue that satisfies the boundary condition
    ε_n = bisect(boundary_condition, n - 1, n + 1)
    return ε_n

# Calculate the first five normalized eigenfunctions and eigenvalues
eigenvalues = []
eigenfunctions = []

for n in range(1, 6):
    # Find eigenvalue using shooting method
    ε_n = find_eigenvalue(n)
    eigenvalues.append(ε_n)
    
    # Find corresponding eigenfunction
    φ = schrodinger_shooting(ε_n, x_span)
    
    # Normalize the eigenfunction
    norm = simps(np.abs(φ)**2, x_span)
    φ_normalized = φ / np.sqrt(norm)
    
    eigenfunctions.append(np.abs(φ_normalized))

# Plot the first five eigenfunctions
plt.figure(figsize=(10, 6))
for i, φ in enumerate(eigenfunctions, 1):
    plt.plot(x_span, φ, label=f'ϕ_{i} (ε_{i} = {eigenvalues[i-1]:.3f})')

plt.title('First Five Normalized Eigenfunctions')
plt.xlabel('x')
plt.ylabel('ϕ_n(x)')
plt.legend()
plt.grid(True)
plt.show()
