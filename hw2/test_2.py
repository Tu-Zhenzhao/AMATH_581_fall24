import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

tol = 1e-4 # define a tolerance level
col = ['r', 'b', 'g', 'c', 'm', 'y', 'k', 'w', 'orange', 'purple', 'brown']
n0 = 100; A = 1; x0 = [0, A]; xp = [-1, 1]
xshoot = np.linspace(xp[0], xp[1],1000)

def shoot2(x, dummy, n0, beta):
    return [x[1], (beta - n0) * x[0]]

beta_start = n0 # beginning value of beta
for modes in range(1, 6): # begin mode loop
    beta = beta_start # initial value of eigenvalue beta
    dbeta = n0 / 100 # default step size in beta
    for _ in range(1000): # begin convergence loop for beta
        y = odeint(shoot2, x0, xshoot, args=(n0,beta))
        if abs(y[-1, 1] - 0) < tol: # check for convergence
            print(beta) # write out eigenvalue
            break # get out of convergence loop
    if (-1) ** (modes + 1) * y[-1, 1] > 0:
        beta -= dbeta
    else:
        beta += dbeta / 2
        dbeta /= 2

    beta_start = beta - 0.1 # after finding eigenvalue, pick new start
    norm = np.trapz(y[:, 0] * y[:, 0], xshoot) # calculate the normalization
    plt.plot(xshoot, y[:, 0] / np.sqrt(norm), col[modes - 1], label=f'Eigenvalue {modes} = {beta:.4f}')


plt.legend()
plt.show()
