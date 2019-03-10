from abc import ABCMeta, abstractmethod
import sys
import numpy as np
import scipy.linalg
from chainer_optnet import pivots

try:
    import cupy
    from chainer_optnet import cupy_decomp_lu
    cupy_available = True
except ImportError:
    cupy_available = False


def get_array_module(a):
    if cupy_available and isinstance(a, cupy.ndarray):
        return cupy
    else:
        return np
 

def beye(xp, nBatch, n, dtype):
    return xp.broadcast_to(xp.expand_dims(xp.eye(n, dtype=dtype), 0), (nBatch, n, n))


def bdiag(x):
    xp = get_array_module(x)
    n, sz = x.shape
    X = xp.zeros((n, sz, sz), dtype=x.dtype)
    I = beye(xp, n, sz, dtype=np.bool)
    X[I] = x.reshape(-1)
    return X


def btranspose(A):
    return A.transpose(0, 2, 1)


def bmvm(A, x):
    xp = get_array_module(A)
    return (A @ xp.expand_dims(x, 2)).squeeze(2)


def bvmm(x, A):
    xp = get_array_module(x)
    return (xp.expand_dims(x, 1) @ A).squeeze(1)


def batch_lu_factor(A):
    xp = get_array_module(A)
    A = A.copy()
    if cupy_available and xp == cupy:
        Ps = xp.empty((A.shape[0], A.shape[1]), dtype=np.int32)
        for i in range(len(A)):
            A[i], Ps[i] = cupy_decomp_lu.lu_factor(A[i], overwrite_a=True)
        return A, Ps
    else:
        Ps = []
        for i in range(len(A)):
            A[i], piv = scipy.linalg.lu_factor(A[i], overwrite_a=True)
            Ps.append(piv)
        return A, xp.array(Ps)


def batch_lu_solve(lu_and_piv, b):
    LU, piv = lu_and_piv
    xp = get_array_module(LU)
    b = b.copy()
    for i in range(len(LU)):
        if cupy_available and xp == cupy:
            b[i] = cupy_decomp_lu.lu_solve((LU[i], piv[i]), b[i], overwrite_b=True)
        else:
            b[i] = scipy.linalg.lu_solve((LU[i], piv[i]), b[i], overwrite_b=True)
    return b


def batch_lu_unpack(LU):
    xp = get_array_module(LU)
    nBatch, n, _ = LU.shape
    I = xp.eye(n, dtype=LU.dtype)
    L = xp.empty((nBatch, n, n), dtype=LU.dtype)
    U = xp.empty((nBatch, n, n), dtype=LU.dtype)
    for i in range(nBatch):
        L[i] = xp.tril(LU[i], k=-1) + I
        U[i] = xp.triu(LU[i])
    return L, U


def batch_bpermute(a, idx):
    xp = get_array_module(a)
    ret = xp.empty_like(a)
    if a.size != 0:
        for i in range(len(idx)):
            ret[i] = pivots.bpermute(a[i], idx[i])
    return ret


def batch_permute(a, idx):
    xp = get_array_module(a)
    ret = xp.empty_like(a)
    if a.size != 0:
        for i in range(len(idx)):
            ret[i] = pivots.permute(a[i], idx[i])
    return ret


def batch_pivots_to_perm(piv):
    xp = get_array_module(piv)
    if cupy_available and xp == cupy:
        piv = piv.get()
        ret = np.empty_like(piv)
        for i in range(len(piv)):
            ret[i] = pivots.pivots_to_perm(piv[i])
        return cupy.asarray(ret)
    else:
        ret = xp.empty_like(piv)
        for i in range(len(piv)):
            ret[i] = pivots.pivots_to_perm(piv[i])
        return ret


def batch_lu_factor_partial(A, B, C):
    """
    factor(A, B, C)(D) compute LU factorization of
    X = (A B)
        (C D).

    c.f. https://locuslab.github.io/qpth/#block-lu-factorization
    """
    xp = get_array_module(A)
    nBatch = A.shape[0]
    n1 = A.shape[1]
    n2 = B.shape[2]
    n = n1 + n2
    assert A.shape == (nBatch,n1,n1)
    assert B.shape == (nBatch,n1,n2)
    assert C.shape == (nBatch,n2,n1)

    X_LU = xp.zeros((nBatch,n,n), dtype=A.dtype)
    X_piv = xp.zeros((nBatch,n), dtype=np.int32)

    if n1 == 0:
        C_invA_B = xp.zeros((nBatch,n2,n2), dtype=A.dtype)
        X_LU_21_pre = xp.zeros((nBatch,n2,0), dtype=A.dtype)
    else:
        A_LU_and_piv = batch_lu_factor(A)
        A_LU, A_piv = A_LU_and_piv
        A_L, A_U = batch_lu_unpack(A_LU)

        invA_B = batch_lu_solve(A_LU_and_piv, B)
        C_invA_B = C @ invA_B

        # A_U^-1 = A^-1 A_P A_L since A = A_P A_L A_U
        A_U_inv = batch_lu_solve(A_LU_and_piv, batch_permute(A_L, batch_pivots_to_perm(A_piv)))
        X_LU_21_pre = C @ A_U_inv

        X_LU[:, 0:n1, 0:n1] = A_LU
        X_LU[:, 0:n1, n1:n1+n2] = A_U @ invA_B
        X_piv[:, 0:n1] = A_piv

    def f(D):
        assert D.shape == (nBatch,n2,n2)

        if n2 > 0:
            S = D - C_invA_B
            S_LU, S_piv = batch_lu_factor(S)

            X_LU[:, n1:, :n1] = batch_bpermute(X_LU_21_pre, batch_pivots_to_perm(S_piv))
            X_LU[:, n1:, n1:] = S_LU
            X_piv[:, n1:] = S_piv + n1

        def check():
            X_L, X_U = batch_lu_unpack(X_LU)
            X = batch_permute(X_L @ X_U, batch_pivots_to_perm(X_piv))
            X_expected = xp.concatenate((xp.concatenate((A, B), axis=2), xp.concatenate((C, D), axis=2)), axis=1)
            return xp.allclose(X, X_expected)
        #assert check()

        return X_LU, X_piv

    return f


def get_sizes(G, A=None):
    if G.ndim == 2:
        nineq, nz = G.shape
        nBatch = 1
    elif G.ndim == 3:
        nBatch, nineq, nz = G.shape
    if A is not None:
        neq = A.shape[1]
    else:
        neq = 0
    return nineq, nz, neq, nBatch


INACC_ERR = """
--------
qpth warning: Returning an inaccurate and potentially incorrect solution.

Some residual is large.
Your problem may be infeasible or difficult.

You can try using the CVXPY solver to see if your problem is feasible
and you can use the verbose option to check the convergence status of
our solver while increasing the number of iterations.
"""


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


class KKTSolverLUFull(KKTSolver):
    def __init__(self, Q, G, A):
        super().__init__(Q, G, A)
        self.Q = Q
        self.G = G
        self.A = A
        self.D = None

    def set_d(self, d):
        self.D = bdiag(d)

    def solve(self, rx, rs, rz, ry):
        assert self.D is not None
        xp = get_array_module(rx)
        nineq, nz, neq, nBatch = get_sizes(self.G, self.A)

        H_ = xp.zeros((nBatch, nz + nineq, nz + nineq), dtype=self.Q.dtype)
        H_[:, :nz, :nz] = self.Q
        H_[:, -nineq:, -nineq:] = self.D
        # H =
        # (Q 0)
        # (0 D)
        if neq > 0:
            A_ = xp.concatenate([xp.concatenate([self.G, beye(xp, nBatch, nineq, dtype=self.Q.dtype)], 2),
                                 xp.concatenate([self.A, xp.zeros((nBatch, neq, nineq), dtype=self.Q.dtype)], 2)], 1)
            g_ = xp.concatenate([rx, rs], 1)
            h_ = xp.concatenate([rz, ry], 1)
        else:
            A_ = xp.concatenate([self.G, beye(xp, nBatch, nineq, dtype=self.Q.dtype)], 2)
            g_ = xp.concatenate([rx, rs], 1)
            h_ = rz

        H_LU = batch_lu_factor(H_)

        # H^-1 =
        #   (Q^-1 0   )
        #   (0    D^-1)
        # A_^T =
        #   (G^T A^T)
        #   (I   0)
        invH_A_ = batch_lu_solve(H_LU, btranspose(A_))
        #   (Q^-1 G^T  Q^-1 A^T)
        #   (D^-1      0)

        invH_g_ = batch_lu_solve(H_LU, g_)

        # A_ =
        #   (G I)
        #   (A 0)
        S_ = A_ @ invH_A_
        # (G Q^-1 G^T + D^-1  G Q^-1 A^T)
        # (A Q^-1 G^T         A Q^-1 A^T)

        S_LU = batch_lu_factor(S_)
        t_ = bvmm(invH_g_, btranspose(A_)) - h_
        w_ = batch_lu_solve(S_LU, -t_)          # solve_kktのwと同じだが順序が違う
        t_ = -g_ - bvmm(w_, A_)                 # solve_kktのg1とg2に相当
        # A_^T =
        #   (G^T A^T)
        #   (I   0)
        # A_^T w_
        # = (G^T dz + A^T dy, dz)
        v_ = batch_lu_solve(H_LU, t_)           # dxの情報と D^-1
        # dx = v_[:, :nz] = Q^-1 g1
        # ds = v_[:, nz:] = D^-1 g2

        dx = v_[:, :nz]
        ds = v_[:, nz:]
        dz = w_[:, :nineq]
        dy = w_[:, nineq:] if neq > 0 else None

        return dx, ds, dz, dy


class KKTSolverLUPartial(KKTSolver):
    def __init__(self, Q, G, A):
        super().__init__(Q, G, A)
        self.G = G
        self.A = A
        self.Q_LU = batch_lu_factor(Q)
        nineq, nz, neq, nBatch = get_sizes(self.G, self.A)
        xp = get_array_module(Q)

        # S = [ A Q^{-1} A^T        A Q^{-1} G^T          ]
        #     [ G Q^{-1} A^T        G Q^{-1} G^T + D^{-1} ]
        #
        # We compute a partial LU decomposition of the S matrix
        # that can be completed once D^{-1} is known.
        # See https://locuslab.github.io/qpth/#block-lu-factorization
        # for more details.
        self.G_invQ_GT = G @ batch_lu_solve(self.Q_LU, btranspose(self.G))
        if neq > 0:
            invQ_AT = batch_lu_solve(self.Q_LU, btranspose(A))
            A_invQ_AT = A @ invQ_AT # A
            G_invQ_AT = G @ invQ_AT # C
            A_invQ_GT = btranspose(G_invQ_AT) # B
        else:
            A_invQ_AT = xp.zeros((nBatch,neq,neq), dtype=Q.dtype) # A
            G_invQ_AT = xp.zeros((nBatch,nineq,neq), dtype=Q.dtype) # C
            A_invQ_GT = xp.zeros((nBatch,neq,nineq), dtype=Q.dtype) # B
        self.factor_kkt = batch_lu_factor_partial(A_invQ_AT, A_invQ_GT, G_invQ_AT)
        self.factor_kkt_eye = beye(xp, nBatch, nineq, dtype=np.bool)
        self.d = None
        self.S_LU = None

    def set_d(self, d):
        xp = get_array_module(d)
        self.d = d
        S_22 = self.G_invQ_GT.copy()
        S_22[self.factor_kkt_eye] += (1. / d).reshape(-1)
        self.S_LU = self.factor_kkt(S_22)

    def solve(self, rx, rs, rz, ry):
        assert self.S_LU is not None
        xp = get_array_module(rx)
        nineq, nz, neq, nBatch = get_sizes(self.G, self.A)

        invQ_rx = batch_lu_solve(self.Q_LU, rx)
        if neq > 0:
            h = xp.concatenate((bvmm(invQ_rx, btranspose(self.A)) - ry,
                                bvmm(invQ_rx, btranspose(self.G)) + rs / self.d - rz), 1)
        else:
            h = bvmm(invQ_rx, btranspose(self.G)) + rs / self.d - rz

        w = -batch_lu_solve(self.S_LU, h)

        g1 = -rx - bvmm(w[:, neq:], self.G)
        if neq > 0:
            g1 -= bvmm(w[:, :neq], self.A)
        g2 = -rs - w[:, neq:]

        dx = batch_lu_solve(self.Q_LU, g1)
        ds = g2 / self.d
        dz = w[:, neq:]
        dy = w[:, :neq] if neq > 0 else None
        return dx, ds, dz, dy


class KKTSolverIRUnopt(KKTSolver):
    """Inefficient iterative refinement."""

    def __init__(self, Q, G, A):
        super().__init__(Q, G, A)
        self.Q = Q
        self.G = G
        self.A = A
        self.D = None
        self.niter = 1

    def set_d(self, d):
        self.D = bdiag(d)

    def solve(self, rx, rs, rz, ry):
        xp = get_array_module(rx)
        nineq, nz, neq, nBatch = get_sizes(self.G, self.A)

        eps = 1e-7
        Q_tilde = self.Q + eps * beye(xp, nBatch, nz, dtype=self.Q.dtype)
        D_tilde = self.D + eps * beye(xp, nBatch, nineq, dtype=self.Q.dtype)

        dx, ds, dz, dy = self.factor_solve_kkt_reg(
            Q_tilde, D_tilde, rx, rs, rz, ry, eps)
        res = self.kkt_resid_reg(dx, ds, dz, dy, rx, rs, rz, ry, eps)
        resx, ress, resz, resy = res
        res = resx
        for k in range(self.niter):
            ddx, dds, ddz, ddy = self.factor_solve_kkt_reg(
                Q_tilde, D_tilde,
                -resx, -ress, -resz, -resy if resy is not None else None,
                eps)
            dx, ds, dz, dy = [v + dv if v is not None else None
                              for v, dv in zip((dx, ds, dz, dy), (ddx, dds, ddz, ddy))]
            res = self.kkt_resid_reg(dx, ds, dz, dy, rx, rs, rz, ry, eps)
            resx, ress, resz, resy = res
            res = resx

        return dx, ds, dz, dy

    def kkt_resid_reg(self, dx, ds, dz, dy, rx, rs, rz, ry, eps):
        Q_tilde = self.Q # ???
        D_tilde = self.D # ???
        resx = bmvm(Q_tilde, dx) + bmvm(btranspose(self.G), dz) + rx
        if dy is not None:
            resx += bmvm(btranspose(self.A), dy)
        ress = bmvm(D_tilde, ds) + dz + rs
        resz = bmvm(self.G, dx) + ds - eps * dz + rz
        resy = bmvm(self.A, dx) - eps * dy + ry if dy is not None else None
        return resx, ress, resz, resy

    def factor_solve_kkt_reg(self, Q_tilde, D_tilde, rx, rs, rz, ry, eps):
        xp = get_array_module(self.G)
        nineq, nz, neq, nBatch = get_sizes(self.G, self.A)

        H_ = xp.zeros((nBatch, nz + nineq, nz + nineq), dtype=Q_tilde.dtype)
        H_[:, :nz, :nz] = Q_tilde
        H_[:, -nineq:, -nineq:] = D_tilde
        if neq > 0:
            A_ = xp.concatenate([xp.concatenate([self.G, beye(xp, nBatch, nineq, dtype=Q_tilde.dtype)], 2),
                                 xp.concatenate([self.A, xp.zeros((nBatch, neq, nineq), dtype=Q_tilde.dtype)], 2)], 1)
            g_ = xp.concatenate([rx, rs], 1)
            h_ = xp.concatenate([rz, ry], 1)
        else:
            A_ = xp.concatenate(
                [G, beye(xp, nBatch, nineq, dtype=Q_tilde.dtype)], 2)
            g_ = xp.concatenate([rx, rs], 1)
            h_ = rz

        H_LU = batch_lu_factor(H_)

        invH_A_ = batch_lu_solve(H_LU, btranspose(A_))
        invH_g_ = batch_lu_solve(H_LU, g_)

        S_ = A_ @ invH_A_
        S_ -= eps * beye(xp, nBatch, neq + nineq, dtype=Q_tilde.dtype)
        S_LU = batch_lu_factor(S_)
        t_ = bvmm(invH_g_, btranspose(A_)) - h_
        w_ = batch_lu_solve(S_LU, -t_)
        t_ = -g_ - bvmm(w_, A_)
        v_ = batch_lu_solve(H_LU, t_)

        dx = v_[:, :nz]
        ds = v_[:, nz:]
        dz = w_[:, :nineq]
        dy = w_[:, nineq:] if neq > 0 else None

        return dx, ds, dz, dy


def quadprog(Q, p, G, h, A, b, kkt_solver: KKTSolver,
             eps=1e-12, verbose=0, notImprovedLim=3, maxIter=20):
    xp = get_array_module(Q)
    nineq, nz, neq, nBatch = get_sizes(G, A)

    # Find initial values
    d = xp.ones((nBatch, nineq), dtype=Q.dtype)
    kkt_solver.set_d(d)
    x, s, z, y = kkt_solver.solve(
        p, xp.zeros((nBatch, nineq), dtype=Q.dtype),
        -h, -b if b is not None else None)

    # Make all of the slack variables >= 1.
    M = s.min(1)
    I = M < 0
    if xp.any(I):
        s[I] -= M[I] - 1

    # Make all of the inequality dual variables >= 1.
    M = z.min(1)
    I = M < 0
    if xp.any(I):
        z[I] -= M[I] - 1

    best = {'resids': None, 'x': None, 'z': None, 's': None, 'y': None}
    nNotImproved = 0

    for i in range(maxIter):
        # affine scaling direction

        rx = (bvmm(y, A) if neq > 0 else 0.) + \
             bvmm(z, G) + \
             bvmm(x, btranspose(Q)) + \
             p
        rs = z
        rz = bvmm(x, btranspose(G)) + s - h
        ry = bvmm(x, btranspose(A)) - b if neq > 0 else 0.0
        mu = xp.abs((s * z).sum(axis=1) / nineq)
        z_resid = xp.linalg.norm(rz, axis=1)
        y_resid = xp.linalg.norm(ry, axis=1) if neq > 0 else 0
        pri_resid = y_resid + z_resid
        dual_resid = xp.linalg.norm(rx, axis=1)
        resids = pri_resid + dual_resid + nineq * mu

        d = z / s
        kkt_solver.set_d(d)

        if verbose >= 1:
            print('iter: {}, pri_resid: {:.5g}, dual_resid: {:.5g}, mu: {:.5g}'.format(
                i, float(pri_resid.mean()), float(dual_resid.mean()), float(mu.mean())),
                file=sys.stderr)

        if best['resids'] is None:
            best['resids'] = resids
            best['x'] = x.copy()
            best['z'] = z.copy()
            best['s'] = s.copy()
            best['y'] = y.copy() if y is not None else None
            nNotImproved = 0
        else:
            I = resids < best['resids']
            if I.sum() > 0:
                nNotImproved = 0
            else:
                nNotImproved += 1
            I_nz = xp.broadcast_to(I.reshape(nBatch,1), (nBatch,nz))
            I_nineq = xp.broadcast_to(I.reshape(nBatch,1), (nBatch, nineq))
            best['resids'][I] = resids[I]
            best['x'][I_nz] = x[I_nz]
            best['z'][I_nineq] = z[I_nineq]
            best['s'][I_nineq] = s[I_nineq]
            if neq > 0:
                I_neq = xp.broadcast_to(I.reshape(nBatch,1), (nBatch, neq))
                best['y'][I_neq] = y[I_neq]
        if nNotImproved == notImprovedLim or best['resids'].max() < eps or mu.min() > 1e32:
            if best['resids'].max() > 1. and verbose >= 0:
                print(INACC_ERR, file=sys.stderr)
            return best['x'], best['y'], best['z'], best['s']

        dx_aff, ds_aff, dz_aff, dy_aff = kkt_solver.solve(rx, rs, rz, ry)

        # compute centering directions
        alpha = xp.minimum(xp.minimum(get_step(z, dz_aff),
                                      get_step(s, ds_aff)),
                           xp.ones(nBatch, dtype=Q.dtype))
        alpha_nineq = xp.broadcast_to(xp.expand_dims(alpha, 1), (nBatch, nineq))
        t1 = s + alpha_nineq * ds_aff
        t2 = z + alpha_nineq * dz_aff
        t3 = xp.sum(t1 * t2, axis=1)
        t4 = xp.sum(s * z, axis=1)
        sig = (t3 / t4)**3

        rx = xp.zeros((nBatch, nz), dtype=Q.dtype)
        rs = (xp.broadcast_to(xp.expand_dims(-mu * sig, 1), (nBatch,nineq)) + ds_aff * dz_aff) / s
        rz = xp.zeros((nBatch, nineq), dtype=Q.dtype)
        ry = xp.zeros((nBatch, neq), dtype=Q.dtype) if neq > 0 else xp.zeros((), dtype=Q.dtype)

        dx_cor, ds_cor, dz_cor, dy_cor = kkt_solver.solve(rx, rs, rz, ry)

        dx = dx_aff + dx_cor
        ds = ds_aff + ds_cor
        dz = dz_aff + dz_cor
        dy = dy_aff + dy_cor if neq > 0 else None
        alpha = xp.minimum(0.999 * xp.minimum(get_step(z, dz),
                                              get_step(s, ds)),
                           xp.ones(nBatch, dtype=Q.dtype))
        alpha_nineq = xp.broadcast_to(xp.expand_dims(alpha, 1), (nBatch, nineq))
        alpha_neq = xp.broadcast_to(xp.expand_dims(alpha, 1), (nBatch, neq)) if neq > 0 else None
        alpha_nz = xp.broadcast_to(xp.expand_dims(alpha, 1), (nBatch, nz))

        x += alpha_nz * dx
        s += alpha_nineq * ds
        z += alpha_nineq * dz
        y = y + alpha_neq * dy if neq > 0 else None

    if best['resids'].max() > 1. and verbose >= 0:
        print(INACC_ERR, file=sys.stderr)
    return best['x'], best['y'], best['z'], best['s']


def get_step(v, dv):
    a = -v / dv
    a[dv >= 0] = 1.0
    return a.min(axis=1)
