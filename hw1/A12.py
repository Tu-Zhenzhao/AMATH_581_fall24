# Matrix Calculation

####
# Question 2i
# matrix: C = [[2, 0, -3], [0, 0, -1]], D = [[1, 2], [2, 3], [-1, 0]]
# Calculate: z = CD
####

# packages
import numpy as np
import matplotlib.pyplot as plt


# matrices
C = np.array([[2, 0, -3], [0, 0, -1]])
D = np.array([[1, 2], [2, 3], [-1, 0]])
A12 = C @ D

#print("CD = \n", z)


