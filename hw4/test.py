import numpy as np
import scipy.integrate
import scipy.sparse
import matplotlib.pyplot as plt
import time
from matplotlib import animation

# Problem 1

## a

x_evals = np.arange(-10, 10, 0.1)
n = 200
b = np.ones((n))
Bin = np.array([-b, b, -b, b])
d = np.array([-1, 1, n-1, 1-n])
matrix1 = scipy.sparse.spdiags(Bin, d, n, n, format='csc')/(2*0.1)
A1 = matrix1.todense()

m = 64
n = m**2
e1 = np.ones(n)
e0 = e1.copy()
e0[0] = -0.5
Low1 = np.tile(np.concatenate((np.ones(m-1), [0])), (m,))
Low2 = np.tile(np.concatenate(([1], np.zeros(m-1))), (m,))
Up1 = np.roll(Low1, 1)
Up2 = np.roll(Low2, m-1)
matrix2 = scipy.sparse.spdiags([e1, e1, Low2, Low1, -4*e0, Up1, Up2, e1, e1], [-(n-m), -m, -m+1, -1, 0, 1, m-1, m, (n-m)], n, n, format='csc')/((20/64)**2)
A4 = matrix2.todense()

m = 64
n = m**2
e1 = np.ones(n)
matrix3 = scipy.sparse.spdiags([e1, -e1, e1, -e1], [-(n-m), -m, m, (n-m)], n, n, format='csc')/(2*(20/64))
A5 = matrix3.todense()

plt.spy(A1)
plt.show()
plt.spy(A4)
plt.show()
plt.spy(A5)
plt.show()
