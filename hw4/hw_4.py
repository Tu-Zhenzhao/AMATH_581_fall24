import numpy as np
from scipy.sparse import spdiags, kron, eye
import matplotlib.pyplot as plt

# Step 1: Define the Grid Parameters
n = 8  # Number of grid points in each dimension
x = np.linspace(-10, 10, n)  # Spatial domain for x
y = np.linspace(-10, 10, n)  # Spatial domain for y
dx = x[1] - x[0]  # Grid spacing in x-direction
dy = y[1] - y[0]  # Grid spacing in y-direction

# Ensure dx and dy are equal (uniform grid)
dx = (x[-1] - x[0]) / (n - 1)
dy = dx

# Step 2: Create the 1D Derivative Matrices with Periodic Boundary Conditions

# Second Derivative Operator (Dxx and Dyy)
e = np.ones(n)  # Vector of ones
main_diag = -2 * e / dx**2  # Main diagonal
off_diag = e / dx**2  # Off-diagonals

# Create the sparse matrix for Dxx
data = np.array([off_diag, main_diag, off_diag])
offsets = np.array([-1, 0, 1])
Dxx = spdiags(data, offsets, n, n, format='csr')

# Adjust for periodic boundary conditions
Dxx = Dxx.tolil()
Dxx[0, -1] = 1 / dx**2  # Wrap-around from first to last
Dxx[-1, 0] = 1 / dx**2  # Wrap-around from last to first
Dxx = Dxx.tocsr()

# Dyy is identical to Dxx in this case
Dyy = Dxx.copy()

# First Derivative Operator (D_x and D_y)
lower_diag = -e / (2 * dx)
upper_diag = e / (2 * dx)

data = np.array([lower_diag, upper_diag])
offsets = np.array([-1, 1])
D_x = spdiags(data, offsets, n, n, format='csr')

# Adjust for periodic boundary conditions
D_x = D_x.tolil()
D_x[0, -1] = -1 / (2 * dx)  # Wrap-around from first to last
D_x[-1, 0] = 1 / (2 * dx)   # Wrap-around from last to first
D_x = D_x.tocsr()

# D_y is identical to D_x in this case
D_y = D_x.copy()

# Step 3: Construct the 2D Operators Using Kronecker Products

# Identity matrix of size n
I_n = eye(n, format='csr')

# Laplacian operator in 2D (Matrix A1)
A1 = kron(I_n, Dxx) + kron(Dyy, I_n)

# First derivative operator in x-direction (Matrix A2)
A2 = kron(I_n, D_x)

# First derivative operator in y-direction (Matrix A3)
A3 = kron(D_y, I_n)

# The matrices A1, A2, and A3 are now ready to use

plt.spy(A1)
plt.show()

plt.spy(A2)
plt.show()

plt.spy(A3)
plt.show()
