import numpy as np
import scipy
from scipy.fft import fft2, ifft2
import matplotlib.pyplot as plt


# Initialize x-y mesh
x = np.linspace(-10, 10, 64, endpoint=False)
y = np.linspace(-10, 10, 64, endpoint=False)
X, Y = np.meshgrid(x, y)

# Define initial conditions
m = 1
alpha = 0
n = 64
u = (np.tanh(np.sqrt(X**2 + Y**2)) - alpha)*np.cos(m*np.angle(X + 1j*Y) - np.sqrt(X**2 + Y**2))
v = (np.tanh(np.sqrt(X**2 + Y**2)) - alpha)*np.sin(m*np.angle(X + 1j*Y) - np.sqrt(X**2 + Y**2))

# Transform into Fourier domain
u0 = fft2(u)
v0 = fft2(v)


vec0_new = np.hstack([(u0.reshape(n*n), v0.reshape(n*n))])

# Append initial conditions
u0 = u0.reshape(-1, 1, order='F')
v0 = v0.reshape(-1, 1, order='F')
vec0 = np.concatenate((u0, v0))

def rhs1(t, n, vec, beta, KX, KY):
    """Right-hand side function to return Fourier transform of the solution"""
    u_hat = vec[:4096].reshape(n, n, order='F')
    v_hat = vec[4096:].reshape(n, n, order='F')

    # Transform out of Fourier domain
    u = ifft2(u_hat)
    v = ifft2(v_hat)

    u_nl = u - u**3 - u*v**2 + beta*(v*u**2 + v**3)
    v_nl = -beta*(u**3 + u*v**2) + v - v*u**2 - v**3

    u_t = fft2(u_nl) - 0.1*((KX**2)*u_hat + (KY**2)*u_hat)
    v_t = fft2(v_nl) - 0.1*((KX**2)*v_hat + (KY**2)*v_hat)

    u_t = u_t.reshape(n**2, order='F')
    v_t = v_t.reshape(n**2, order='F')
    rhs = np.concatenate((u_t, v_t), axis=0)

    return rhs



t_span = np.linspace(0, 4, 9)
r1 = np.arange(0, n/2, 1)
r2 = np.arange(-n/2, 0, 1)
kx = (2*np.pi/20)*np.concatenate((r1, r2))
ky = kx.copy()
KX, KY = np.meshgrid(kx, ky)
beta = 1

# Timestep using the explicit Runge-Kutta method of order 4(5)
sol1 = scipy.integrate.solve_ivp(lambda t, vec: rhs1(t, n, vec, beta, KX, KY), [0, 4], np.squeeze(vec0), t_eval = t_span)
A1 = sol1.y



# Define Cheb
def cheb(N):
    if N == 0:
        D = 0.
        x = 1.
    else:
        n = np.arange(0, N+1)
        x = np.cos(np.pi*n/N).reshape(N+1, 1)
        c = (np.hstack(([2.], np.ones(N-1), [2.])) * (-1)**n).reshape(N+1, 1)
        X = np.tile(x, (1, N+1))
        dX = X - X.T
        D = np.dot(c, 1./c.T) / (dX + np.eye(N+1))
        D -= np.diag(np.sum(D.T, axis=0))  # Changed sum() to np.sum()
    
    return D, x.reshape(N+1)



# Define initial conditions
m = 1
alpha = 0
n = 30
N2 = (n+1)*(n+1)
# Create the Chebyshev differentiation matrix
D, x = cheb(n)

D[n,:]=0
D[0,:]=0

D2 = (np.dot(D, D))/((10)*(10))
y = x

# Scale Laplacian
I = np.eye(len(D2))
Lap = np.kron(D2, I) + np.kron(I, D2)


# Create the Chebyshev points
X, Y = np.meshgrid(x,y)
X = X*(20/2)
Y = Y*(20/2)


u = (np.tanh(np.sqrt(X**2 + Y**2)) - alpha)*np.cos(m*np.angle(X + 1j*Y) - np.sqrt(X**2 + Y**2))
v = (np.tanh(np.sqrt(X**2 + Y**2)) - alpha)*np.sin(m*np.angle(X + 1j*Y) - np.sqrt(X**2 + Y**2))

# Append initial conditions
u1 = u.flatten()
v1 = v.flatten()
vec1 = np.concatenate([u1, v1])

def rhs2(t, vec, beta,N, Lap):
    """"Right-hand side function to solve our PDE"""
    u = vec[:N]
    v = vec[N:2*N]

    u_nl = u - u**3 - u*v**2 + beta*(v*u**2 + v**3)
    v_nl = -beta*(u**3 + u*v**2) + v - v*u**2 - v**3

    u_t = u_nl + 0.1*(Lap@u)
    v_t = v_nl + 0.1*(Lap@v)

    rhs = np.concatenate((u_t, v_t), axis=0)

    return rhs


# Timestep using the explicit Runge-Kutta method of order 4(5)
sol2 = scipy.integrate.solve_ivp(lambda t, vec: rhs2(t, vec, beta,N2, Lap), [0, 4], np.squeeze(vec1), t_eval = t_span)
A2 = sol2.y