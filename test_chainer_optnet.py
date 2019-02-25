import chainer_optnet

import numpy as xp
#import cupy as xp
dtype = xp.float32

# https://scaron.info/blog/quadratic-programming-in-python.html
M = xp.array([[1., 2., 0.], [-8., 3., 2.], [0., 1., 1.]], dtype=dtype)
P = xp.dot(M.T, M)
q = xp.dot(xp.array([3., 2., 3.], dtype=dtype), M)
G = xp.array([[1., 2., 1.], [2., 0., 1.], [-1., 2., -1.]], dtype=dtype)
h = xp.array([3., 2., -2.], dtype=dtype)
A = None
b = None

print(chainer_optnet.quadprog(P, q, G, h))

import chainer.gradient_check
dl = xp.random.uniform(-1, 1, (3,)).astype(dtype)
chainer.gradient_check.check_backward(chainer_optnet.quadprog, (P, q, G, h), dl, atol=0.1) # XXX
