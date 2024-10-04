
#####
#Question 1
# Function: f(x)=x*sin(3x)-exp(x)
# Method: Newton Raphson
# Initial guess: x(1) = -1.6
# Goal: converge (in absolute value) to 10^-6
#####

# packages
import numpy as np
import matplotlib.pyplot as plt

# parameters
x0 = -1.6       # initial guess
tol = 10**(-6)  # tolerance
max_iter = 20   # maximum number of iterations

# function
def f(x):
    return x*np.sin(3*x) - np.exp(x)

# derivative
def df(x):
    return np.sin(3*x) + 3*x*np.cos(3*x) - np.exp(x)

# table of iterations
x_j = [x0]          # list of f(x_n) values
acc = [abs(f(x0))]  # list of accuracy values

# Newton Raphson Iteration
def newtonRaphson(x0):
    for i in range(max_iter):
        x1 = x0 - f(x0)/df(x0)
        x0 = x1
        x_j.append(x0)
        acc.append(abs(f(x0)))
        if abs(f(x0)) < tol:
            break
    return x1, x_j, acc



# output
x1, x_n, err = newtonRaphson(x0)
#print('Local minima at: ', x1)
#print('Number of iterations: ', len(x_n))
#print('Error: ', err[-1])
# save to numpy vector
A1 = x_n
print(x_n)
print(len(x_n))

#print("TABLE:")
#print("----------")
## print j, x(j), acc(j) as table    
#print('j\t x(j)\t\t acc(j)')
#for i in range(len(x_n)):
#    print(i, '\t', x_n[i], '\t', err[i])
#
## plot the iterations vs error
#plt.plot(err)
#plt.xlabel('Number of iterations')
#plt.ylabel('Error')
#plt.title('Newton Raphson')
#plt.show()


# parameters
x_left = -0.4
x_right = -0.7
tol = 10**(-6)                      # tolerance
max_iter = 30                       # maximum number of iterations
init_mid = (x_left + x_right)/2     # initial mid point

# iterations table
acc = []        # absolute value of f(x)
mid = []        # mid point



# bisection
def bisection(x_left, x_right):
    for i in range(max_iter):
        x_mid = (x_left + x_right)/2
        if f(x_left)*f(x_mid) < 0:
            x_right = x_mid
        else:
            x_left = x_mid
        mid.append(x_mid)
        acc.append(abs(f(x_mid)))
        if abs(f(x_mid)) < tol:
            break
    return x_mid, mid, acc

# output
x_mid, mid, acc = bisection(x_left, x_right)

#print('Local minima at: ', x_mid)
#print('Number of iterations: ', len(mid))
#print('Error: ', acc[-1])
A2 = mid

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
x1, x_j, acc = newtonRaphson(x0)
A3 = np.array([len(acc_nr), len(acc_bi)])




# matrices
A = np.array([[1, 2], [-1, 1]])
B = np.array([[2, 0], [0, 2]])

# matrix addition
A4 = A + B

# matrices
x = np.array([1, 1])
y = np.array([0, 1])
A5 = 3*x - 4*y


# matrices
x = np.array([[1], [1]])
A = np.array([[1, 2], [-1, 1]])
A6 = (A @ x).reshape(2,)


# matrices
B = np.array([[2, 0], [0, 2]])
x = np.array([[1], [0]])
y = np.array([[0], [1]])
A7 = (B @ (x - y)).reshape(2,)


# matrices
x = np.array([[1], [0]])
D = np.array([[1, 2], [2, 3], [-1, 0]])
A8 = (D @ x).reshape(3,)

# matrices
D = np.array([[1, 2], [2, 3], [-1, 0]])
y = np.array([[0], [1]])
z = np.array([[1], [2], [-1]])
A9 = (D @ y + z).reshape(3,)

# matrices
A = np.array([[1, 2], [-1, 1]])
B = np.array([[2, 0], [0, 2]])
A10 = A @ B

# matrices
B = np.array([[2, 0], [0, 2]])
C = np.array([[2, 0, -3], [0, 0, -1]])
A11 = B @ C

# matrices
C = np.array([[2, 0, -3], [0, 0, -1]])
D = np.array([[1, 2], [2, 3], [-1, 0]])
A12 = C @ D


