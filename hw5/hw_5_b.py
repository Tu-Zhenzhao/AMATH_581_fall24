import numpy as np
from scipy.sparse import diags, kron, eye, csr_matrix
from scipy.sparse.linalg import spsolve, splu, bicgstab, gmres
from scipy.integrate import solve_ivp
import time
import matplotlib.pyplot as plt

# Parameters
n = 64  # Number of grid points in each dimension
x_min, x_max = -10, 10  # Spatial domain in x
y_min, y_max = -10, 10  # Spatial domain in y
nu = 0.001  # Viscosity
t_span = (0, 4)  # Time span for integration
t_eval = np.arange(0, 4.5, 0.5)  # Time evaluation points

# Spatial grid
x = np.linspace(x_min, x_max, n, endpoint=False)
y = np.linspace(y_min, y_max, n, endpoint=False)
dx = (x_max - x_min) / n
dy = (y_max - y_min) / n
X, Y = np.meshgrid(x, y, indexing='ij')

# Initial vorticity (elliptical Gaussian)
omega0 = np.exp(-X**2 - Y**2 / 20)
omega0_flat = omega0.flatten()

# Construct Laplacian operator A with periodic boundary conditions
def construct_laplacian(n, dx):
    diagonals = [-2 * np.ones(n), np.ones(n - 1), np.ones(n - 1)]
    offsets = [0, -1, 1]
    lap1d = diags(diagonals, offsets, shape=(n, n), format='csr')
    lap1d = lap1d / dx**2

    # Periodic boundary conditions
    lap1d = lap1d.tolil()
    lap1d[0, -1] = 1 / dx**2
    lap1d[-1, 0] = 1 / dx**2
    lap1d = lap1d.tocsr()

    I = eye(n, format='csr')
    A = kron(lap1d, I) + kron(I, lap1d)
    return A

# Laplacian operator
A = construct_laplacian(n, dx)
A = A.tocsr()

# Adjust A(0,0) as per the instructions
A = A.tolil()
A[0, 0] = 2  # Set A(0,0) = 2 instead of -4
A = A.tocsr()

# Precompute LU decomposition
start_time = time.time()
lu = splu(A)
lu_time = time.time() - start_time
print(f"LU decomposition time: {lu_time:.4f} seconds")

# Function to solve for psi using different methods
def solve_streamfunction(omega_flat, method='direct'):
    b = omega_flat.copy()
    if method == 'direct':
        start_time = time.time()
        psi_flat = spsolve(A, b)
        elapsed_time = time.time() - start_time
    elif method == 'LU':
        start_time = time.time()
        psi_flat = lu.solve(b)
        elapsed_time = time.time() - start_time
    elif method == 'BiCGSTAB':
        start_time = time.time()
        psi_flat, info = bicgstab(A, b, tol=1e-6, maxiter=1000)
        elapsed_time = time.time() - start_time
    elif method == 'GMRES':
        start_time = time.time()
        psi_flat, info = gmres(A, b, tol=1e-6, restart=50, maxiter=1000)
        elapsed_time = time.time() - start_time
    else:
        raise ValueError("Unknown method")
    return psi_flat.reshape((n, n)), elapsed_time

# Lists to store computational times
times_direct = []
times_lu = []
times_bicgstab = []
times_gmres = []

# Lists to store residuals
residuals_bicgstab = []
residuals_gmres = []

# Choose method for the simulation
method = 'direct'  # Options: 'direct', 'LU', 'BiCGSTAB', 'GMRES'

def rhs(t, omega_flat):
    """Compute the right-hand side of the vorticity equation."""
    omega = omega_flat.reshape((n, n))

    # Solve for streamfunction ψ
    psi, elapsed_time = solve_streamfunction(omega_flat, method=method)

    # Store computational times
    if method == 'direct':
        times_direct.append(elapsed_time)
    elif method == 'LU':
        times_lu.append(elapsed_time)
    elif method == 'BiCGSTAB':
        times_bicgstab.append(elapsed_time)
    elif method == 'GMRES':
        times_gmres.append(elapsed_time)

    # Compute derivatives using finite differences with periodic boundaries
    omega_roll_xp = np.roll(omega, -1, axis=0)
    omega_roll_xm = np.roll(omega, 1, axis=0)
    omega_roll_yp = np.roll(omega, -1, axis=1)
    omega_roll_ym = np.roll(omega, 1, axis=1)

    psi_roll_xp = np.roll(psi, -1, axis=0)
    psi_roll_xm = np.roll(psi, 1, axis=0)
    psi_roll_yp = np.roll(psi, -1, axis=1)
    psi_roll_ym = np.roll(psi, 1, axis=1)

    omega_x = (omega_roll_xp - omega_roll_xm) / (2 * dx)
    omega_y = (omega_roll_yp - omega_roll_ym) / (2 * dy)
    psi_x = (psi_roll_xp - psi_roll_xm) / (2 * dx)
    psi_y = (psi_roll_yp - psi_roll_ym) / (2 * dy)

    # Compute Jacobian [ψ, ω]
    J = psi_x * omega_y - psi_y * omega_x

    # Compute Laplacian of ω using finite differences
    laplacian_omega = (omega_roll_xp + omega_roll_xm + omega_roll_yp + omega_roll_ym - 4 * omega) / dx**2

    # RHS of the vorticity equation
    domega_dt = -J + nu * laplacian_omega

    return domega_dt.flatten()

# Run the simulation for different methods
methods = ['direct', 'LU', 'BiCGSTAB', 'GMRES']
results = {}
times = {}

for method in methods:
    print(f"\nRunning simulation with method: {method}")
    times_direct = []
    times_lu = []
    times_bicgstab = []
    times_gmres = []
    residuals_bicgstab = []
    residuals_gmres = []

    # Reset LU decomposition if method changes
    if method == 'LU':
        start_time = time.time()
        P_lu, L_lu, U_lu = splu(A).LU
        lu_time = time.time() - start_time
        print(f"LU decomposition time: {lu_time:.4f} seconds")

    # Integrate in time using solve_ivp
    sol = solve_ivp(rhs, t_span, omega0_flat, t_eval=t_eval, method='RK45')

    # Store the results
    results[method] = sol.y
    if method == 'direct':
        times[method] = times_direct
    elif method == 'LU':
        times[method] = times_lu
    elif method == 'BiCGSTAB':
        times[method] = times_bicgstab
        residuals_bicgstab = np.array(residuals_bicgstab)
    elif method == 'GMRES':
        times[method] = times_gmres
        residuals_gmres = np.array(residuals_gmres)

# Extract the solutions
A2 = results['direct']
A3 = results['LU']

# Plot computational times
plt.figure(figsize=(10, 6))
plt.plot(times['direct'], label='Direct Solve (A\\b)')
plt.plot(times['LU'], label='LU Decomposition')
plt.plot(times['BiCGSTAB'], label='BiCGSTAB')
plt.plot(times['GMRES'], label='GMRES')
plt.xlabel('Time Step')
plt.ylabel('Computational Time (s)')
plt.title('Computational Time for Different Methods')
plt.legend()
plt.grid()
plt.show()

# For BiCGSTAB and GMRES, plot residuals if available
# (This requires modifying the solver to store residuals)
