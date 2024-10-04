# Matrix Calculation

####
# Question 2f
# matrix: D = [[1, 2], [2, 3], [-1, 0]], y = [[0], [1]], z = [[1], [2], [-1]]
# Calculate: C = Dy + z
####

# packages
import numpy as np
import matplotlib.pyplot as plt


# matrices
D = np.array([[1, 2], [2, 3], [-1, 0]])
y = np.array([[0], [1]])
z = np.array([[1], [2], [-1]])
A9 = D @ y + z

#print("Dy+z = \n", C)

