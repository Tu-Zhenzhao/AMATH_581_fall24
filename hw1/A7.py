# Matrix Calculation

####
# Question 2d
# matrix: B = [[2, 0], [0, 2]], x = [[1], [0]], y = [[0], [1]]
# Calculate: C = B(x - y)
####

# packages
import numpy as np
import matplotlib.pyplot as plt


# matrices
B = np.array([[2, 0], [0, 2]])
x = np.array([[1], [0]])
y = np.array([[0], [1]])
A7 = B @ (x - y)

#print("B(x-y) = \n", C)


