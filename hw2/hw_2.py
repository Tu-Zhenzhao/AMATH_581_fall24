# Homework 2

# given boundary value problem:
# d^2y/dx^2 - [Kx^2-beta]y = 0
# where we expect the solution to be y \to 0 as x \to \infty
# take K = 1, x \in [-4, 4] choose xspan = -4:0.1:4
# goal: find the first 5 eigenvalues and eigenfunctions

# import libraries
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

# initial parameters
y0 = 0
xspan = np.linspace(-4, 4, 81)
K = 1
tol = 1e-5
col = ['r', 'b', 'g', 'c', 'm', 'y', 'k', 'w', 'orange', 'purple', 'brown']
init_beta = 1


# define the differential equation
def func(y, x, K, beta):
    return [y[1], (K * x**2 - beta) * y[0]]

print("Shape of xspan:", xspan.shape)

# eigenvalue list
eigvals = []

# eigenfunction list
eigfuncs = []

# loop through different beta
beta_start = init_beta

for modes in range(5):
    beta = beta_start
    dbeta = 1 # initial step size for eigenvalue adjustment
    # convergence loop for each beta
    for i in range(1000):
        x0 = [1, np.sqrt(K*4**2-beta)*1]
        # solve the ODE
        y = odeint(func, x0, xspan, args=(K, beta))
        # check if the solution is converged

        err = y[-1,1] + np.sqrt(K*4**2-beta)*y[-1,0]
        print("Last value of y:", y[-1, 0])
        if np.abs(err) < tol:
            eigvals.append(beta)
            print('Epsilon =', beta)
            print("Last value of y:", y[-1, 0])
            break

        # shooting scheme: check it is greater than 0
        if (-1) ** (modes + 1) * err > 0:
            beta -= dbeta
        else:
            beta += dbeta
            dbeta *= 0.5

    # finding a eigenvalue then find a new beta
    beta_start = beta + 0.1
    # print 
    print("Shape of y:", y.shape)
    # normalization for eigenfunction
    norm = np.trapz(y[:, 0]*y[:, 0], xspan)
    # append eigenfunction make it to 5 column matrix
    eigfuncs.append(np.abs(y[:, 0])/np.sqrt(norm))
    # plotting the solution
    plt.plot(xspan, y[:, 0]/np.sqrt(norm), col[modes], label=r'$\beta$ = ' + str(beta))

# trans eigenvalue to 1*5 matrix
A2 = np.array(eigvals).reshape(1, 5)
# trans eigenfunction to 100*5 matrix
A1 = np.array(eigfuncs).T
# print
#print("Eigenvalues:", eigvals.shape)
#print("Eigenfunctions:", eigfuncs.shape)
print(A1.shape)
print(A1)
plt.legend()
plt.show()


