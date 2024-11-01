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

#print("Shape of xspan:", xspan.shape)

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
        #print("Last value of y:", y[-1, 0])
        if np.abs(err) < tol:
            eigvals.append(beta)
            #print('Epsilon =', beta)
            #print("Last value of y:", y[-1, 0])
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
    #print("Shape of y:", y.shape)
    # normalization for eigenfunction
    norm = np.trapz(y[:, 0]*y[:, 0], xspan)
    # append eigenfunction make it to 5 column matrix
    eigfuncs.append(np.abs(y[:, 0])/np.sqrt(norm))
    # plotting the solution
    plt.plot(xspan, y[:, 0]/np.sqrt(norm), col[modes], label=r'$\beta$ = ' + str(beta))

# trans eigenvalue to 1 by 5 matrix
A2 = np.array(eigvals)
# trans eigenfunction to 100*5 matrix
A1 = np.array(eigfuncs).T

# Parameters
L = 4
dx = 0.1
K = 1
xspan = np.arange(-L, L + dx, dx)
N_total = len(xspan)
print("Total number of points:", N_total)
N = N_total - 2  # Number of interior points
x_i = xspan[1:-1]  # Interior points

# Initialize the matrix A
A = np.zeros((N, N))

# Off-diagonal entries
E = 1 * np.ones(N - 1)

# Main diagonal entries
D = -2 - dx**2 * K * x_i**2

# Construct the tridiagonal matrix A
A = np.diag(D) + np.diag(E, k=-1) + np.diag(E, k=1)
print("Matrix A before modifications:\n", A)
# Modify the first row (forward differencing)
A[0, 0] += 4/3
A[0, 1] -= 1/3

# Modify the last row (backward differencing)
A[N-1, N-2] -=  1/3
A[N-1, N-1] += 4/3
print("Matrix A after modifications:\n", A)
print("Eigenvalue problem size:", A.shape)
# Solve the eigenvalue problem
eigenvalues, eigenvectors = np.linalg.eig(-A)
# Sort eigenvalues and eigenvectors
idx = eigenvalues.argsort()
eigenvalues = eigenvalues[idx]
final_eigenvalues =eigenvalues[:5]/(dx**2)
print("Eigenvalues:\n", final_eigenvalues)


# Extract the first five eigenvalues and eigenvectors
eigenvalues = eigenvalues[:5]
eigenvectors = eigenvectors[:, idx][:, :5]
print("Eigenvectors:\n", eigenvectors.shape)

# Include boundary points in eigenfunctions
eigenfunctions = np.zeros([N_total, 5])

# Normalize the eigenfunctions and take absolute values
for i in range(5):
    eigenvector = eigenvectors[:, i]
    print("Eigenvector:", eigenvector.shape)

    # adding boundary conditions
    phi0 = 4/3 * eigenvector[0]  - 1/3 * eigenvector[1]
    phiN = 4/3 * eigenvector[N-1] - 1/3 * eigenvector[N-2]

    # construct the full eigenfunction
    eigenfunctions[:, i] = np.concatenate(([phi0], eigenvector, [phiN]))

    norm = np.trapz(eigenfunctions[:, i]**2, xspan)
    eigenfunctions[:, i] /= np.sqrt(norm)
    eigenfunctions[:, i] = np.abs(eigenfunctions[:, i])

# Save the eigenfunctions and eigenvalues
A3 = eigenfunctions  # 5-column matrix of eigenfunctions
A4 = final_eigenvalues         # 1x5 vector of eigenvalues


#Q3
import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate


# Define ODE
def rhsfunc1(t, y, beta,gamma):
    f1 = y[1] #f1 = y1'= y2 = phi'
    K = 1
    n0 = K*t*t #n(x) = x*x (here t is the independent variable)
    f2 = (gamma*y[0]*y[0]+n0 - epsilon)*y[0]#this changes #f2 = y2' = phi"
    return np.array([f1, f2])

L = 2 
xp = [-L,L,] # xspan
tol = 1e-5 # We want to find beta such that |y(x=1)| < tol
K = 1
epsilon_start = 0 # This is our initial beta value, we will change it#recommended on piazza to start from epsilon = 0
A_start = 0.001
gamma = 0.05

eigen_values_q3_A = []
eigen_functions_q3_A = []

# Make a loop over beta values to find more eigenvalue-eigenfunction pairs
#modes is another way to say eigenfunction

for modes in range(2): # Try to find 5 modes
    epsilon = epsilon_start 
    depsilon = 0.01 # This is the amount we will decrease beta by each time we don't have an eigenvalue
                 # until we get an eigenvalue
    A =A_start
     
    for j in range(1000):
        x_evals = np.linspace(-L, L, (20*L)+1) #20L + 1 linearly spaced points in between
        
        #update/define y0 again, initial conditions
        y0 = np.array([A, A*np.sqrt(K*L*L-epsilon)])
        
        ##check
        sol = scipy.integrate.solve_ivp(lambda x,y: rhsfunc1(x, y, epsilon,gamma), xp, y0, t_eval = x_evals)
        y_sol = sol.y[0, :] #gives phi
        y_sol_1 =sol.y[1,:] #gives phi'
        
        
        #compute norm and boundary condition
        norm = np.trapz(y_sol**2,x=x_evals)
        BC = y_sol_1[-1]+(np.sqrt(K*L*L-epsilon)*y_sol[-1]) #don't multiply by A boundary condition

            
        #checking both conditions
        if np.abs(BC) < tol and np.abs(norm - 1) < tol :
            #the boundary condition at phi'(x=L) should be limited to be less than here
            #phi'(L) = - sqrt(epsilon)*phi(L) -->given < tol
            #print(r'We got the eigenvalue! $\epsilon = $', epsilon)
            print(f"Norm: {norm}")
            eigen_values_q3_A.append(epsilon)
            print(r'We got the eigenvalue! $\epsilon = $', epsilon, "At j = ", j)
            break
        else:
            #update initial condition with new A
            A = A/np.sqrt(norm)
        

       
        #shooting for BC
        if (-1)**(modes)*(BC) > 0:
            
            #phi'(L) = - sqrt(KL^2 - epsilon)*phi(L)
            epsilon = epsilon + depsilon 
            # Decrease beta if y(1)>0, because we know that y(1)>0 for beta = beta_start
            
        else:
            epsilon = epsilon - depsilon/2  # Increase beta by a smaller amount if y(1)<0
            depsilon = depsilon/2 # Cut dbeta in half to make we converge


            
    epsilon_start = epsilon + 0.1 # increase beta once we have found one mode.

    # normalization for eigenfunction
    eigen_functions_q3_A.append(abs(y_sol))


A5 =  np.array(eigen_functions_q3_A).T
#print("eigen_functions_q3_A Q3 A:", np.array(eigen_functions_q3_A).T.shape)
A6 = eigen_values_q3_A


# Define ODE
def rhsfunc1(t, y, beta,gamma):
    f1 = y[1] #f1 = y1'= y2 = phi'
    K = 1
    n0 = K*t*t #n(x) = x*x (here t is the independent variable)
    f2 = (gamma*y[0]*y[0]+n0 - epsilon)*y[0]#this changes #f2 = y2' = phi"
    return np.array([f1, f2])

# Define some constants
#n0 = 0 #defined inside the function
# Define our initial conditions
#A = 1 # This is the shooting-method parameter that we will change , y1_(-1) = A
#y0 = np.array([A, 1]) # y1_(-1) = A, y2_(-1) = 1 #do I need to keep updating A? yes!
L = 2 
xp = [-L,L] # xspan
tol = 1e-5 # We want to find beta such that |y(x=1)| < tol
K = 1
epsilon_start = 0 # This is our initial beta value, we will change it#recommended on piazza to start from epsilon = 0
A_start = 0.001
gamma = -0.05

eigen_values_q3_B = []
eigen_functions_q3_B = []

# Make a loop over beta values to find more eigenvalue-eigenfunction pairs
#modes is another way to say eigenfunction

for modes in range(2): # Try to find 5 modes
    epsilon = epsilon_start 
    depsilon = 0.01 # This is the amount we will decrease beta by each time we don't have an eigenvalue
                 # until we get an eigenvalue
    A =A_start
     
    for j in range(1000):
        x_evals = np.linspace(-L, L, (20*L)+1) #20L + 1 linearly spaced points in between
        
        #update/define y0 again, initial conditions
        y0 = np.array([A, A*np.sqrt(K*L*L-epsilon)])
        
        ##check
        sol = scipy.integrate.solve_ivp(lambda x,y: rhsfunc1(x, y, epsilon,gamma), xp, y0, t_eval = x_evals)
        y_sol = sol.y[0, :] #gives phi
        y_sol_1 =sol.y[1,:] #gives phi'
        
        
        #compute norm and boundary condition
        norm = np.trapz(y_sol**2,x=x_evals)
        BC = y_sol_1[-1]+(np.sqrt(K*L*L-epsilon)*y_sol[-1]) #don't multiply by A boundary condition

            
        #checking both conditions
        if np.abs(BC) < tol and np.abs(norm - 1) < tol :
            #the boundary condition at phi'(x=L) should be limited to be less than here
            #phi'(L) = - sqrt(epsilon)*phi(L) -->given < tol
            #print(r'We got the eigenvalue! $\epsilon = $', epsilon)
            eigen_values_q3_B.append(epsilon)
            break
        else:
            #update initial condition with new A
            A = A/np.sqrt(norm)
        

       
        #shooting for BC
        if (-1)**(modes)*(BC) > 0:
            
            #phi'(L) = - sqrt(KL^2 - epsilon)*phi(L)
            epsilon = epsilon + depsilon 
            # Decrease beta if y(1)>0, because we know that y(1)>0 for beta = beta_start
            
        else:
            epsilon = epsilon - depsilon/2  # Increase beta by a smaller amount if y(1)<0
            depsilon = depsilon/2 # Cut dbeta in half to make we converge


            
    epsilon_start = epsilon + 0.1 # increase beta once we have found one mode.

    # normalization for eigenfunction
    eigen_functions_q3_B.append(abs(y_sol))

# make eigenfunction to 2 columns matrix
A7 = np.array(eigen_functions_q3_B).T
print(f"A7: {A7.shape}", A7.shape)
# make eigenvalue to 1 by 2 vector
A8 = eigen_values_q3_B

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
    log_avg_steps = np.log(avg_steps)
    log_tols = np.log(tolerances)
    # Fit a line to the data
    slope, intercept = np.polyfit(log_avg_steps ,log_tols,  1)
    slopes.append(slope)
    # Plot the data and the fitted line
    plt.plot( log_avg_steps, log_tols, 'o-', label=f'{method} (slope={slope:.4f})')
    #plt.plot(log_tols, slope * log_tols + intercept, '--')

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
A9 = np.array(slopes)
print("\nSlopes array:")
print(A9.shape)


import numpy as np


def factorial(n):
   result = 1
   for i in range(1, n + 1):
       result *= i
   return result

# Parameters
K = 1
L = 4                     
xp = [-L, L] 
xshoot = np.linspace(xp[0], xp[1], 81)
n = len(xshoot)

# hardcode H
h = np.array([np.ones_like(xshoot), 2*xshoot, 4*xshoot**2-2 , 8*xshoot**3-12*xshoot, 16 * xshoot**4 - 48 * xshoot**2 + 12])


# Calculate exact eigenfunctions
phi = np.zeros([n, 5])
for j in range(5):
    # Fixed: Use j instead of n in calculate_hermite
    phi[:, j] = (np.exp(-xshoot**2/2) * h[j] / np.sqrt(2**j * factorial(j) * np.sqrt(np.pi))).T


# Initialize error arrays                                                 
err_psi_a = np.zeros(5)
err_psi_b = np.zeros(5)
err_a = np.zeros(5)
err_b = np.zeros(5)

# Load numerical solutions (from part (a) and (b))
eigvec_a = A1
eigval_a = A2

eigvec_b = A3
eigval_b = A4



# Calculate errors
for j in range(5):
    # Fixed: Changed xspan to xshoot
    err_psi_a[j] = np.trapz((np.abs(eigvec_a[:, j]) - np.abs(phi[:, j]))**2, xshoot)
    err_psi_b[j] = np.trapz((np.abs(eigvec_b[:, j]) - np.abs(phi[:, j]))**2, xshoot)
    err_a[j] = 100 * np.abs(eigval_a[j] - (2*j - 1)) / (2*j - 1)  # Fixed: Changed (2*j-1) to (2*j+1)
    err_b[j] = 100 * np.abs(eigval_b[j] - (2*j + 1)) / (2*j + 1)  # Fixed: Changed (2*j-1) to (2*j+1)

A10 = err_psi_a
A11 = err_a
A12 = err_psi_b  # Fixed: Changed err_psi_a to err_psi_b
A13 = err_b


print("Error in wavefunctions (method a):", A10)
print("Error in eigenvalues (method a):", A11)
print("Error in wavefunctions (method b):", A12)
print("Error in eigenvalues (method b):", A13)

