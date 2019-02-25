import numpy as np
import pdipm_single

dtype = np.float32
Solver = pdipm_single.KKTSolverChoPartial

Q = np.array([[1,0],[0,0.001]], dtype=dtype)
q = np.array([3,4], dtype=dtype)
G = np.array([[-1,0], [0,-1], [-1,-3], [2,5], [3,4]], dtype=dtype)
h = np.array([0,0,-15,100,80], dtype=dtype)
A = None #np.zeros((0,0), dtype=dtype)
b = None #np.zeros(0, dtype=dtype)
print(pdipm_single.forward(Q, q, G, h, A, b, Solver(Q, G, A), verbose=True))

# https://scaron.info/blog/quadratic-programming-in-python.html
M = np.array([[1., 2., 0.], [-8., 3., 2.], [0., 1., 1.]])
P = np.dot(M.T, M)
q = np.dot(np.array([3., 2., 3.]), M)
G = np.array([[1., 2., 1.], [2., 0., 1.], [-1., 2., -1.]])
h = np.array([3., 2., -2.])
A = None #np.zeros((0,0), dtype=dtype)
b = None #np.zeros(0, dtype=dtype)
Q = P
print(pdipm_single.forward(Q, q, G, h, A, b, Solver(Q, G, A), verbose=True))
