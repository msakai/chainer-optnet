import numpy as np
from chainer_optnet.solvers import pdipm_batch

dtype = np.float32
#dtype = np.float64
#Solver = pdipm_batch.KKTSolverLUFull
Solver = pdipm_batch.KKTSolverLUPartial
#Solver = pdipm_batch.KKTSolverIRUnopt
xp = np
#import cupy
#xp = cupy

Q = xp.array([[[1,0],[0,1]]], dtype=dtype)
q = xp.array([[1,1]], dtype=dtype)
G = xp.array([[[-1,0], [0,-1]]], dtype=dtype)
h = xp.array([[-1,-1]], dtype=dtype)
A = None #xp.zeros((0,0), dtype=dtype)
b = None #xp.zeros(0, dtype=dtype)
print(pdipm_batch.forward(Q, q, G, h, A, b, Solver(Q, G, A), verbose=True))

# https://scaron.info/blog/quadratic-programming-in-python.html
M = xp.array([[1., 2., 0.], [-8., 3., 2.], [0., 1., 1.]])
P = xp.dot(M.T, M)
q = xp.dot(xp.array([3., 2., 3.]), M)
G = xp.array([[1., 2., 1.], [2., 0., 1.], [-1., 2., -1.]])
h = xp.array([3., 2., -2.])
A = None #xp.zeros((0,0), dtype=dtype)
b = None #xp.zeros(0, dtype=dtype)
Q = xp.expand_dims(P, 0)
q = xp.expand_dims(q, 0)
G = xp.expand_dims(G, 0)
h = xp.expand_dims(h, 0)
print(pdipm_batch.forward(Q, q, G, h, A, b, Solver(Q, G, A), verbose=True))

Q = xp.array([[[1,0],[0,0.1]]], dtype=dtype)
q = xp.array([[3,4]], dtype=dtype)
G = xp.array([[[-1,0], [0,-1], [-1,-3], [2,5], [3,4]]], dtype=dtype)
h = xp.array([[0,0,-15,100,80]], dtype=dtype)
A = None #xp.zeros((0,0), dtype=dtype)
b = None #xp.zeros(0, dtype=dtype)
print(pdipm_batch.forward(Q, q, G, h, A, b, Solver(Q, G, A), verbose=True))
