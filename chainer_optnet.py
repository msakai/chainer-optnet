import chainer
from chainer import function_node
from chainer.utils import type_check
import chainer.functions as F
import pdipm_batch


class QuadProg(function_node.FunctionNode):
    def __init__(self):
        super().__init__()

    def check_type_forward(self, in_types):
        n_in = in_types.size()
        type_check.expect(n_in == 6)
        Q, p, G, h, A, b = in_types

        type_check.expect(
            Q.dtype.kind == 'f',
            Q.ndim == 3,
            Q.shape[1] == Q.shape[2],

            p.dtype.kind == 'f',
            p.ndim == 2,
            p.shape == Q.shape[:2],

            G.dtype.kind == 'f',
            G.ndim == 3,
            G.shape[0] == Q.shape[0],
            G.shape[2] == Q.shape[1],

            h.dtype.kind == 'f',
            h.ndim == 2,
            h.shape[0] == Q.shape[0],
            h.shape[1] == G.shape[1],

            A.dtype.kind == 'f',
            A.ndim == 3,
            A.shape[0] == Q.shape[0],
            A.shape[2] == Q.shape[1],

            b.dtype.kind == 'f',
            b.ndim == 2,
            b.shape[0] == Q.shape[0],
            b.shape[1] == A.shape[1]
        )

    def forward(self, inputs):
        Q, p, G, h, A, b = inputs

        self.kkt_solver = pdipm_batch.KKTSolverLUPartial(Q, G, A)
        zhat, nu, lam, slack = pdipm_batch.forward(Q, p, G, h, A, b, self.kkt_solver)
        if nu is None:
            xp = chainer.backend.get_array_module(*inputs)
            nu = xp.zeros((A.shape[0], A.shape[1]), dtype=Q.dtype)

        self.retain_outputs((0,))
        self.nu = nu
        self.lam = lam
        self.slack = slack
        return zhat,

    def backward(self, indexes, grad_outputs):
        dl_dzhat, = grad_outputs

        zhat, = self.get_retained_outputs()
        nu = self.nu
        lam = self.lam
        slack = self.slack

        xp = chainer.backend.get_array_module(dl_dzhat)
        (nBatch, nz) = zhat.shape
        nineq = slack.shape[1]
        neq = nu.shape[1]

        d = xp.clip(lam, a_min=1e-8, a_max=None) / xp.clip(slack, a_min=1e-8, a_max=None)
        self.kkt_solver.set_d(d)
        dx, _, dlam, dnu = self.kkt_solver.solve(
            dl_dzhat.data,
            xp.zeros((nBatch, nineq), dtype=dl_dzhat.dtype),
            xp.zeros((nBatch, nineq), dtype=dl_dzhat.dtype),
            xp.zeros((nBatch, neq), dtype=dl_dzhat.dtype)
        )
        dx = chainer.Variable(dx)
        dlam = chainer.Variable(dlam)
        if dnu is None:
            dnu = xp.zeros_like(nu.data)
        dnu = chainer.Variable(dnu)

        def bger(x, y):
            return F.expand_dims(x, 2) @ F.expand_dims(y, 1)

        dQ = 0.5 * (bger(dx, zhat) + bger(zhat, dx))
        dp = dx
        dG = bger(dlam, zhat) + bger(lam, dx)
        dh = -dlam
        dA = bger(dnu, zhat) + bger(nu, dx)
        db = -dnu
        return dQ, dp, dG, dh, dA, db


def quadprog(Q, p, G, h, A = None, b = None):
    nBatch = None
    is_batched = False

    def expand(x, ndim):
        nonlocal nBatch
        nonlocal is_batched

        if x.ndim == ndim + 1:
            is_batched = True
            if nBatch is None:
                nBatch = x.shape[0]
            elif nBatch != x.shape[0]:
                raise RuntimeError("unexpected batch size %d (expected %d)" % (x.shape[0], nBatch))
            return x
        elif x.ndim == ndim:
            if nBatch is None:
                nBatch = 1
            elif nBatch != 1:
                raise RuntimeError("unexpected batch size %d (expected %d)" % (1, nBatch))
            x = F.expand_dims(x, 0)
            return x
        else:
            raise RuntimeError("unexpected number of dimensions")

    Q = expand(Q, 2)
    p = expand(p, 1)
    G = expand(G, 2)
    h = expand(h, 1)
    if A is None:
        xp = chainer.backend.get_array_module(Q)
        A = chainer.Variable(xp.zeros((Q.shape[0], 0, Q.shape[1])))
        b = chainer.Variable(xp.zeros((Q.shape[0], 0)))
    else:
        A = expand(A, 2)
        b = expand(b, 1)

    ret = QuadProg().apply((Q, p, G, h, A, b))[0]
    if is_batched:
        return ret
    else:
        return F.squeeze(ret, 0)


if __name__ == '__main__':
    import numpy as np
    dtype = np.float32

    # https://scaron.info/blog/quadratic-programming-in-python.html
    M = np.array([[1., 2., 0.], [-8., 3., 2.], [0., 1., 1.]], dtype=dtype)
    P = np.dot(M.T, M)
    q = np.dot(np.array([3., 2., 3.], dtype=dtype), M)
    G = np.array([[1., 2., 1.], [2., 0., 1.], [-1., 2., -1.]], dtype=dtype)
    h = np.array([3., 2., -2.], dtype=dtype)
    A = None
    b = None

    print(quadprog(P, q, G, h))

    import chainer.gradient_check
    dl = np.random.uniform(-1, 1, (3,)).astype(dtype)
    chainer.gradient_check.check_backward(quadprog, (P, q, G, h), dl, atol=0.1) # XXX