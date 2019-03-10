import cvxpy
import numpy as np

def quadprog(Q, p, G, h, A, b, verbose=False):
    Q = (Q + Q.T) / 2 # XXX

    nz = p.shape[0]
    neq = A.shape[0] if A is not None else 0
    nineq = G.shape[0]

    z = cvxpy.Variable(nz)

    obj = cvxpy.Minimize(0.5 * cvxpy.quad_form(z, Q) + p.T * z)
    eqCon = A * z == b if neq > 0 else None
    if nineq > 0:
        slacks = cvxpy.Variable(nineq)
        ineqCon = G * z + slacks == h
        slacksCon = slacks >= 0
    else:
        ineqCon = slacks = slacksCon = None

    cons = [x for x in [eqCon, ineqCon, slacksCon] if x is not None]
    prob = cvxpy.Problem(obj, cons)
    prob.solve(verbose=verbose)

    zhat = np.array(z.value).ravel()
    nu = np.array(eqCon.dual_value).ravel() if eqCon is not None else None
    if ineqCon is not None:
        lam = np.array(ineqCon.dual_value).ravel()
        slacks = np.array(slacks.value).ravel()
    else:
        lam = slacks = None

    return zhat, nu, lam, slacks
