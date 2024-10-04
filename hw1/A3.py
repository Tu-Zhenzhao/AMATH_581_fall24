#####
#Question 3
# Function: f(x)=x*sin(3x)-exp(x)
# Method: Bisection And Newton Raphson
# Bisection: Initial end points, x_right = -0.7 and x_left = -0.4
# Newton Raphson: Initial guess: x(1) = -1.6
# Goal: converge (in absolute value) to 10^-6 comparing the two methods
#####

# packages
import numpy as np
import matplotlib.pyplot as plt

# Newton Raphson Parameters
x0 = -1.6       # initial guess
tol = 10**(-6)  # tolerance
max_iter = 20   # maximum number of iterations

# Bisection Parameters
x_left = -0.7
x_right = -0.4
tol = 10**(-6)                  # tolerance
max_iter = 30                   # maximum number of iterations
init_mid = (x_left + x_right)/2  # initial midpoint

# iterations table of Bisection
acc_bi = []        # absolute value of f(x)
mid = []        # mid point
# iterations table of Newton Raphson
x_j = []        # value of x
acc_nr = []        # absolute value of f(x)


# function
def f(x):
    return x*np.sin(3*x) - np.exp(x)


# derivative
def df(x):
    return np.sin(3*x) + 3*x*np.cos(3*x) - np.exp(x)


# bisection
def bisection(x_left, x_right):
    for i in range(max_iter):
        x_mid = (x_left + x_right)/2
        if f(x_left)*f(x_mid) < 0:
            x_right = x_mid
        else:
            x_left = x_mid
        mid.append(x_mid)
        acc_bi.append(abs(f(x_mid)))
        if abs(f(x_mid)) < tol:
            break
    return x_mid, mid, acc_bi


# Newton Raphson
def newtonRaphson(x0):
    for i in range(max_iter):
        x1 = x0 - f(x0)/df(x0)
        x0 = x1
        x_j.append(x0)
        acc_nr.append(abs(f(x0)))
        if abs(f(x0)) < tol:
            break
    return x1, x_j, acc_nr


# output for bisection
x_mid, mid, acc = bisection(x_left, x_right)
#print('Local minima at: ', x_mid)
#print('Number of iterations: ', len(mid))
#print('Error: ', acc_bi[-1])
#
#print("------------------------------")

# output for Newton Raphson
x1, x_j, acc = newtonRaphson(x0)
#print('Local minima at: ', x1)
#print('Number of iterations: ', len(x_j))
#print('Error: ', acc_nr[-1])
#
A3 = np.array([len(acc_nr), len(acc_bi)])
#print(A3)


## print the two methods iterations in 1 \times 2 vector
#print("------------------------------")
#print('Bisection\t', 'Newton Raphson\t')
#print(len(acc_bi), '\t', len(acc_nr), '\t')
#
#
## plot two methods (be careful acc_bi and acc_nr dimensions are different) comparing the value at each iteration
#plt.plot(acc_bi, label='Bisection')
#plt.plot(acc_nr, label='Newton Raphson')
#plt.xlabel('j')
#plt.ylabel('error')
#plt.legend()
#plt.show()
