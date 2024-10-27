# Problem 3

import numpy as np
import scipy.integrate
import matplotlib.pyplot as plt



def rhsfunc2(t, y, eps, gam):
    f1 = y[1]
    f2 = (gam*abs(y[0])**2 + t**2 - eps)*y[0]
    return np.array([f1, f2])

xp = [-3, 3]
tol = 1e-5
A = 1e-3

x_evals3 = np.linspace(-3, 3, 61)
eigenvalues3 = np.zeros(4)
eigenfunctions3 = np.zeros([61, 4])

for i, gamma in enumerate([0.05, -0.05]):
    eps_start = 0
    
    for mode in range(2):
        eps = eps_start
        deps = 0.01
        
        for j in range(1000):
            y0 = np.array([A, A*np.sqrt(3**2 - eps)])
            sol = scipy.integrate.solve_ivp(lambda x, y: rhsfunc2(x, y, eps, gamma), xp, y0, t_eval = x_evals3)
            
            eig_norm3 = np.trapz(sol.y[0, :]**2, x = x_evals3)
            if ((np.abs(sol.y[1, -1] + np.sqrt(3**2 - eps)*sol.y[0, -1]) < tol) and (np.abs(eig_norm3 - 1) < tol)):
                eigenfunctions3[:, mode + 2*i] = abs(sol.y[0, :])
                eigenvalues3[mode + 2*i] = eps
                break
            else:
                A = A/np.sqrt(eig_norm3)
            
            y0 = np.array([A, A*np.sqrt(3**2 - eps)])
            sol = scipy.integrate.solve_ivp(lambda x, y: rhsfunc2(x, y, eps, gamma), xp, y0, t_eval = x_evals3)
            
            eig_norm3 = np.trapz(sol.y[0, :]**2, x = x_evals3)
            if ((np.abs(sol.y[1, -1] + np.sqrt(3**2 - eps)*sol.y[0, -1]) < tol) and (np.abs(eig_norm3 - 1) < tol)):
                eigenfunctions3[:, mode + 2*i] = abs(sol.y[0, :])
                eigenvalues3[mode + 2*i] = eps
                break           
            elif (-1)**(mode)*(sol.y[1, -1] + np.sqrt(3**2 - eps)*sol.y[0, -1]) > 0:
                eps = eps + deps
            else:
                eps = eps - deps/2
                deps = deps/2
            
        eps_start = eps + 0.1

        plt.plot(x_evals3, abs(eigenfunctions3[:, mode + 2*i]), label=f'{mode} Mode, ε = {eigenvalues3[mode + 2*i]:.2f}')

plt.legend()
plt.xlabel('x')
plt.ylabel('Normalized |ϕ_n(x)|')
plt.title('First Two Normalized Eigenfunctions with Modified Boundary Conditions')
plt.show()
