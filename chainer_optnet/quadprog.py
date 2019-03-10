import numpy as np
import chainer
import chainer.cuda
from chainer import function_node
from chainer.utils import type_check
import chainer.functions as F
import chainer_optnet.solvers.pdipm_batch as pdipm_batch

try:
    import chainer_optnet.solvers.cvxpy as cvxpy_solver
    _cvxpy_available = True
except ImportError:
    _cvxpy_available = False


class QuadProg(function_node.FunctionNode):
    def __init__(self, solver):
        super().__init__()
        self.solver = solver

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

        if self.solver == 'pdipm_batch':
            self.kkt_solver = pdipm_batch.KKTSolverLUPartial(Q, G, A)
            zhat, nu, lam, slack = pdipm_batch.quadprog(Q, p, G, h, A, b, self.kkt_solver)
            if nu is None:
                xp = chainer.backend.get_array_module(*inputs)
                nu = xp.zeros((A.shape[0], A.shape[1]), dtype=Q.dtype)
        elif self.solver == 'cvxpy':
            if not _cvxpy_available:
                raise RuntimeError("cvxpy is not available")
            self.kkt_solver = None
            Q = chainer.cuda.to_cpu(Q)
            p = chainer.cuda.to_cpu(p)
            G = chainer.cuda.to_cpu(G)
            h = chainer.cuda.to_cpu(h)
            A = chainer.cuda.to_cpu(A)
            b = chainer.cuda.to_cpu(b)
            n = len(Q)
            nv = G.shape[2]
            nineq = G.shape[1]
            neq = A.shape[1]
            zhat = np.empty((n,nv), dtype=Q.dtype)
            nu = np.empty((n,neq), dtype=Q.dtype)
            lam = np.empty((n,nineq), dtype=Q.dtype)
            slack = np.empty((n,nineq), dtype=Q.dtype)
            for i in range(n):
                zhat[i], nu[i], lam[i], slack[i] = cvxpy_solver.quadprog(Q[i], p[i], G[i], h[i], A[i], b[i])
            xp = chainer.backend.get_array_module(*inputs)
            zhat = xp.asarray(zhat)
            nu = xp.asarray(nu)
            lam = xp.asarray(lam)
            slack = xp.asarray(slack)
            self.retain_inputs((0,2,4)) # Q, G, A
        else:
            raise ValueError("unknown solver")

        self.retain_outputs((0,))
        self.nu = nu
        self.lam = lam
        self.slack = slack
        return zhat,

    def backward(self, indexes, grad_outputs):
        dl_dzhat, = grad_outputs

        if self.solver == "cvxpy":
            Q, G, A = self.get_retained_inputs()
            self.kkt_solver = pdipm_batch.KKTSolverLUPartial(Q.data, G.data, A.data)

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
            dnu = xp.zeros_like(nu)
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


def quadprog(Q, p, G, h, A = None, b = None, solver='pdipm_batch'):
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
        A = chainer.Variable(xp.zeros((Q.shape[0], 0, Q.shape[1]), dtype=Q.dtype))
        b = chainer.Variable(xp.zeros((Q.shape[0], 0), dtype=Q.dtype))
    else:
        A = expand(A, 2)
        b = expand(b, 1)

    ret = QuadProg(solver=solver).apply((Q, p, G, h, A, b))[0]
    if is_batched:
        return ret
    else:
        return F.squeeze(ret, 0)
