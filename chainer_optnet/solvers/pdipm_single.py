from abc import ABCMeta, abstractmethod
import sys
import numpy as np
import scipy.linalg


def get_sizes(G, A=None):
    if A is None:
        neq = 0
    else:
        new = A.shape[0]
    return (G.shape[0], G.shape[1], neq, 1)


def cho_factor_partial(A, C):
    """
    factor(A, C)(D) compute Cholesky factorization of
    X = (A C^T)
        (C D  ).
    """
    n1 = A.shape[0]
    n2 = C.shape[0]
    n = n1 + n2
    assert A.shape == (n1,n1)
    assert C.shape == (n2,n1)

    U = np.zeros((n,n), dtype=A.dtype)
    (U_11, _) = scipy.linalg.cho_factor(A)
    U_12 = np.linalg.solve(U_11.T, C.T)
    U[:n1, :n1] = U_11
    U[:n1, n1:] = U_12
    R = - U_12.T @ U_12

    def factor(D):
        (U_22, _) = scipy.linalg.cho_factor(R + D)
        U[n1:, n1:] = U_22
        def check():
            X = np.concatenate((np.concatenate((A, C.T), 1), np.concatenate((C, D), 1)), 0)
            return np.allclose(X, U.T @ U)
        #assert check()
        return U
    return factor


class KKTSolver(metaclass=ABCMeta):
    """
    Solve the equation system
    K_sym @ np.concatenate(x,s,z,y) = - np.concatenate(rx, rs, rz, ry)
    where
      K_sym = (Q 0     G^T A^T)
              (0 D(d)  I   0  )
              (G I     0   0  )
              (A 0     0   0  )
    """
    def __init__(self, Q, G, A):
        super().__init__()

    @abstractmethod
    def set_d(self, d):
        pass

    @abstractmethod
    def solve(self, rx, rs, rz, ry):
        pass


class KKTSolverChoPartial(KKTSolver):
    def __init__(self, Q, G, A):
        super().__init__(Q, G, A)
        self.G = G
        self.A = A
        (self.U_Q, _) = scipy.linalg.cho_factor(Q)
        nineq, nz, neq, _ = get_sizes(self.G, self.A)

        # S = [ A Q^{-1} A^T        A Q^{-1} G^T          ]
        #     [ G Q^{-1} A^T        G Q^{-1} G^T + D^{-1} ]
        #
        # We compute a partial Cholesky decomposition of the S matrix
        # that can be completed once D^{-1} is known.
        # See https://locuslab.github.io/qpth/#block-lu-factorization
        # for more details.
        self.G_invQ_GT = self.G @ scipy.linalg.cho_solve((self.U_Q,False), self.G.T)
        if neq > 0:
            invQ_AT = scipy.linalg.cho_solve((U_Q,False), A.T)
            A_invQ_AT = A @ invQ_AT
            G_invQ_AT = G @ invQ_AT
        else:
            A_invQ_AT = np.zeros((neq,neq), dtype=Q.dtype)
            G_invQ_AT = np.zeros((nineq,neq), dtype=Q.dtype)
        self.factor_kkt = cho_factor_partial(A_invQ_AT, G_invQ_AT)
        self.d = None
        self.S_U = None

    def set_d(self, d):
        self.d = d
        self.U_S = self.factor_kkt(self.G_invQ_GT + np.diag(1.0 / d))

    def solve(self, rx, rs, rz, ry):
        nineq, nz, neq, _ = get_sizes(self.G, self.A)

        invQ_rx = scipy.linalg.cho_solve((self.U_Q, False), rx)
        if neq > 0:
            h = np.concatenate((self.A.dot(invQ_rx) - ry, self.G.dot(invQ_rx) + rs / d - rz), 0)
        else:
            h = self.G.dot(invQ_rx) + rs / self.d - rz

        w = -scipy.linalg.cho_solve((self.U_S, False), h)

        g1 = -rx - self.G.T.dot(w[neq:])
        if neq > 0:
            g1 -= self.A.T.dot(w[:neq])
        g2 = -rs - w[neq:]

        dx = scipy.linalg.cho_solve((self.U_Q, False), g1)
        ds = g2 / self.d
        dz = w[neq:]
        dy = w[:neq] if neq > 0 else None

        return dx, ds, dz, dy


def quadprog(Q, p, G, h, A, b, kkt_solver: KKTSolver,
             eps=1e-12, verbose=0, maxIter=20):
    nineq, nz, neq, _ = get_sizes(G, A)

    # find initial values
    d = np.ones(nineq, dtype=Q.dtype)
    kkt_solver.set_d(d)

    x, s, z, y = kkt_solver.solve(
        p, np.zeros(nineq, dtype=Q.dtype),
        -h, -b if b is not None else None)

    if np.min(s) < 0:
        s -= np.min(s) - 1
    if np.min(z) < 0:
        z -= np.min(z) - 1

    prev_resid = None
    for i in range(maxIter):
        # affine scaling direction
        rx = (A.T.dot(y) if neq > 0 else 0.) + G.T.dot(z) + Q.dot(x) + p
        rs = z
        rz = G.dot(x) + s - h
        ry = A.dot(x) - b if neq > 0 else np.zeros(0, dtype=Q.dtype)
        mu = s.dot(z) / nineq
        pri_resid = np.linalg.norm(ry) + np.linalg.norm(rz)
        dual_resid = np.linalg.norm(rx)
        resid = pri_resid + dual_resid + nineq * mu

        d = z / s
        kkt_solver.set_d(d)

        if verbose >= 1:
            print(("iter: {}, primal_res = {:.5g}, dual_res = {:.5g}, " +
                   "gap = {:.5g}, kappa(d) = {:.5g}").format(
                    i, pri_resid, dual_resid, mu, min(d) / max(d)),
                  file=sys.stderr)
        # if (pri_resid < 5e-4 and dual_resid < 5e-4 and mu < 4e-4):
        improved = (prev_resid is None) or (resid < prev_resid + 1e-6)
        if not improved or resid < eps:
            break
        prev_resid = resid

        dx_aff, ds_aff, dz_aff, dy_aff = kkt_solver.solve(rx, rs, rz, ry)

        # compute centering directions
        alpha = min(min(get_step(z, dz_aff), get_step(s, ds_aff)), 1.0)
        sig = (np.dot(s + alpha * ds_aff, z +
                         alpha * dz_aff) / (np.dot(s, z)))**3

        rx = np.zeros(nz, dtype=Q.dtype)
        rs = (-mu * sig * np.ones(nineq, dtype=Q.dtype) + ds_aff * dz_aff) / s
        rz = np.zeros(nineq, dtype=Q.dtype)
        ry = np.zeros(neq, dtype=Q.dtype)

        dx_cor, ds_cor, dz_cor, dy_cor = kkt_solver.solve(rx, rs, rz, ry)

        dx = dx_aff + dx_cor
        ds = ds_aff + ds_cor
        dz = dz_aff + dz_cor
        dy = dy_aff + dy_cor if neq > 0 else None
        alpha = min(1.0, 0.999 * min(get_step(s, ds), get_step(z, dz)))
        dx_norm = np.linalg.norm(dx)
        dz_norm = np.linalg.norm(dz)
        if np.isnan(dx_norm) or dx_norm > 1e5 or dz_norm > 1e5:
            # Overflow, return early
            return x, y, z

        x += alpha * dx
        s += alpha * ds
        z += alpha * dz
        y = y + alpha * dy if neq > 0 else None

    return x, y, z, s


def get_step(v, dv):
    #I = dv < 1e-12
    I = dv < 0
    if np.any(I):
        return np.min(-v[I] / dv[I])
    else:
        return 1
