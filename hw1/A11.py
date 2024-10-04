# Matrix Calculation

####
# Question 2h
# matrix: B = [[2, 0], [0, 2]], C = [[2, 0, -3], [0, 0, -1]]
# Calculate: z = BC
####

# packages
import numpy as np
import matplotlib.pyplot as plt


# matrices
B = np.array([[2, 0], [0, 2]])
C = np.array([[2, 0, -3], [0, 0, -1]])
A11 = B @ C

#print("BC = \n", z)


