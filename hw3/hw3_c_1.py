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

# Define some constants
#n0 = 0 #defined inside the function
# Define our initial conditions
#A = 1 # This is the shooting-method parameter that we will change , y1_(-1) = A
#y0 = np.array([A, 1]) # y1_(-1) = A, y2_(-1) = 1 #do I need to keep updating A? yes!
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

    
    eigen_functions_q3_A.append(y_sol)
    
    # plot
    plt.plot(x_evals, y_sol, label=f'{modes} Mode, $\epsilon = $ {eigen_values_q3_A[-1]:.2f}')

plt.title('First Five Normalized Eigenfunctions of the Harmonic Oscillator')
plt.xlabel('x')
plt.ylabel('ϕₙ(x)')
plt.legend()
plt.grid()
plt.show()

    
