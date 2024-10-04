
#####
#Question 2
# Function: f(x)=x*sin(3x)-exp(x)
# Method: Bisection
# Initial end points: x_right = -0.7 and x_left = -0.4
# Goal: converge (in absolute value) to 10^-6
#####

# packages
import numpy as np
import matplotlib.pyplot as plt

# parameters
x_left = -0.4
x_right = -0.7
tol = 10**(-6)                      # tolerance
max_iter = 30                       # maximum number of iterations
init_mid = (x_left + x_right)/2     # initial mid point

# iterations table
acc = []        # absolute value of f(x)
mid = []        # mid point

# function
def f(x):
    return x*np.sin(3*x) - np.exp(x)


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
print(A2)

#
#
#
## table of iterations: j, x_j, f(x_j), error
#print('j\t', 'x_j\t', 'f(x_j)\t', 'error')
#for i in range(len(mid)):
#    print(i, '\t', mid[i], '\t', f(mid[i]), '\t', acc[i])
#
#
## plot iterations vs error
#plt.plot(acc)
#plt.xlabel('j')
#plt.ylabel('error')
#plt.show()

