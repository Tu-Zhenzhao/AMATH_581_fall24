import numpy as np
from scipy.fft import fft2, ifft2
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

# Parameters
n = 64  # Number of grid points
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

# Wavenumber grids for FFT
kx = np.fft.fftfreq(n, d=(x_max - x_min) / n) * 2 * np.pi
ky = np.fft.fftfreq(n, d=(y_max - y_min) / n) * 2 * np.pi
kx[0] = 1e-6  # Avoid division by zero
ky[0] = 1e-6
KX, KY = np.meshgrid(kx, ky, indexing='ij')
K2 = KX**2 + KY**2
K2[0, 0] = 1e-6  # Avoid division by zero at (0,0)

def compute_streamfunction(omega):
    """Compute the streamfunction ψ from vorticity ω using FFT."""
    omega_hat = fft2(omega)
    psi_hat = omega_hat / (-K2)
    psi_hat[0, 0] = 0  # Enforce zero mean for ψ
    psi = np.real(ifft2(psi_hat))
    return psi

def rhs(t, omega_flat):
    """Compute the right-hand side of the vorticity equation."""
    omega = omega_flat.reshape((n, n))

    # Compute streamfunction ψ
    psi = compute_streamfunction(omega)
    
    # Compute derivatives using spectral methods
    omega_hat = fft2(omega)
    psi_hat = fft2(psi)

    # Compute derivatives in Fourier space
    omega_x_hat = 1j * KX * omega_hat
    omega_y_hat = 1j * KY * omega_hat
    psi_x_hat = 1j * KX * psi_hat
    psi_y_hat = 1j * KY * psi_hat

    # Transform back to physical space
    omega_x = np.real(ifft2(omega_x_hat))
    omega_y = np.real(ifft2(omega_y_hat))
    psi_x = np.real(ifft2(psi_x_hat))
    psi_y = np.real(ifft2(psi_y_hat))

    # Compute Jacobian [ψ, ω]
    J = psi_x * omega_y - psi_y * omega_x

    # Compute Laplacian of ω using spectral methods
    laplacian_omega_hat = -K2 * omega_hat
    laplacian_omega = np.real(ifft2(laplacian_omega_hat))

    # RHS of the vorticity equation
    domega_dt = -J + nu * laplacian_omega

    return domega_dt.flatten()

# Integrate in time using solve_ivp
sol = solve_ivp(rhs, t_span, omega0_flat, t_eval=t_eval, method='RK45')

# Extract the solution for vorticity at each time step
A1 = sol.y  # A1 has shape (n*n, number of time steps)

# Plotting the vorticity at different time steps
num_plots = len(t_eval)
fig, axes = plt.subplots(2, int(np.ceil(num_plots / 2)), figsize=(15, 6))

for i, ax in enumerate(axes.flatten()):
    if i < num_plots:
        omega_t = A1[:, i].reshape((n, n))
        c = ax.contourf(X, Y, omega_t, levels=50, cmap='jet')
        ax.set_title(f't = {t_eval[i]:.2f}')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        fig.colorbar(c, ax=ax)
    else:
        ax.axis('off')

plt.tight_layout()
plt.show()

print(A1)
