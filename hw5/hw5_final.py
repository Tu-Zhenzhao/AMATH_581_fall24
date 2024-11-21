import numpy as np
from scipy.fft import fft2, ifft2
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from scipy.sparse import diags, kron, eye, csr_matrix, spdiags
from scipy.sparse.linalg import spsolve, bicgstab, gmres, splu
from scipy.linalg import lu, solve_triangular 
import time

import matplotlib.pyplot as plt
import copy



# Define parameters
m = 64
n = m * m # N value in x and y directions
x_min, x_max = -10, 10
dx = (x_max - x_min) / m

# total size of matrix
e0 = np.zeros(n) # vector of zeros
e1 = np.ones(n) # vector of ones
e2 = np.copy(e1) # copy the one vector
e4 = np.copy(e0) # copy the zero vector

for j in range(1, m+1):
    e2[m*j-1] = 0 # overwrite every m^th value with zero
    e4[m*j-1] = 1 # overwirte every m^th value with one

# Shift to correct positions
e3 = np.zeros_like(e2)
e3[1:n] = e2[0:n-1]
e3[0] = e2[n-1]
e5 = np.zeros_like(e4)
e5[1:n] = e4[0:n-1]
e5[0] = e4[n-1]

# Place diagonal elements
diagonals = [e1.flatten(), e1.flatten(), e5.flatten(),e2.flatten(),-4*e1.flatten(), e3.flatten(),e4.flatten(), e1.flatten(), e1.flatten()]
offsets = [-(n-m), -m, -m+1, -1, 0, 1, m-1, m, (n-m)]
A = spdiags(diagonals, offsets, n, n).toarray()
A = A/(dx**2)

# Create matrix B
e = np.ones(n)

data_B = [e, -1*e, e, -1*e]
offsets_B = [-(n-m), -m, m, (n - m)]
B = spdiags(data_B, offsets_B, n, n).toarray()
B = B/(2*dx)

# Create matrix C
e1 = np.zeros(n)
e2 = np.ones(n)
e3 = np.ones(n)
e4 = np.zeros(n)


for i in range(n):
    if (i + 1) % m == 1:
        e1[i] = 1
        e3[i] = 0
    if (i + 1) % m == 0:
        e2[i] = 0
        e4[i] = 1

e2 = -1*e2
e4 = -1*e4

C = spdiags([e1,e2, e3,e4],[-m+1, -1,1,m-1], n,n).toarray()
C = C/(2*dx)


# Parameters
n = 64  # Number of grid points
x_min, x_max = -10, 10  # Spatial domain in x
y_min, y_max = -10, 10  # Spatial domain in y
nu = 0.001  # Viscosity
t_span = (0, 4)  # Time span for integration
#t_span = (0, 10)
t_eval = np.arange(0, 4.5, 0.5)  # Time evaluation points
#t_eval = np.arange(0, 10.5, 0.5)

# Define spatial domain and initial conditions
x2 = np.linspace(x_min, x_max, n + 1)
x = x2[:n]
y2 = np.linspace(y_min, y_max, n + 1)
y = y2[:n]
X, Y = np.meshgrid(x, y)

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

def spc_rhs(t, wt2, n, K, nu, A, B, C):
    
    wt = wt2.reshape((n, n))
    wtfft = fft2(wt)
    psix = np.real(ifft2(-wtfft / K)).flatten()
    rhs = (nu * (A @ wt2) + (B @ wt2) * (C @ psix) - (B @ psix) * (C @ wt2))
    
    return rhs

# Integrate in time using solve_ivp
sol = solve_ivp(spc_rhs, [0,4], omega0_flat, t_eval=t_eval, method='RK45', args=(n, K2, nu,A,B,C))
# Extract the solution for vorticity at each time step
A1 = sol.y  # A1 has shape (n*n, number of time steps)


# set A[0,0]=2
A[0,0] /= -2

def rhs_ode_GE(t, omega, A, B, C):
        
        #solving for A.psi = w using GE
        #psi0 = spsolve(A,omega_0)
        psi = np.linalg.solve(A,omega)
        psi_x = B@psi
        psi_y = C@psi
        omega_x = B@omega
        omega_y = C@omega
        omega = nu*(A@omega) - psi_x*omega_y + psi_y*omega_x
         
        
        return omega

def rhs_ode_LU(t, omega, A, B, C):
        
        #solving for A.psi = w using GE
        #psi0 = spsolve(A,omega_0)
        plu = splu(A)
        psi = plu.solve(omega)
        psi_x = B@psi
        psi_y = C@psi
        omega_x = B@omega
        omega_y = C@omega
        omega = nu*(A)@omega - np.multiply(psi_x,omega_y) + np.multiply(psi_y,omega_x)
        
        return omega
    
#%%time for Direct
start_time = time.time()
sol_GE = solve_ivp(lambda t,omega: rhs_ode_GE(t,omega, A, B, C), [0, 4], 
                                omega0_flat, t_eval = t_eval)
end_time = time.time()

elapsed_time = end_time - start_time
#print(elapsed_time)
A2 = copy.deepcopy(sol_GE.y)


#%%time for LU 
start_time = time.time()
sol_LU = solve_ivp(lambda t,omega: rhs_ode_LU(t,omega, A, B, C), [0, 4], 
                                omega0_flat, t_eval = t_eval)
end_time = time.time()

elapsed_time = end_time - start_time
#print(elapsed_time)
A3 = copy.deepcopy(sol_GE.y)
A3

