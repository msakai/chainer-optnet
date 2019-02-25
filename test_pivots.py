import numpy as np
import scipy.linalg
import pivots

A = np.array([[2, 5, 8, 7], [5, 2, 2, 8], [7, 5, 6, 6], [5, 4, 4, 8]])

lu, piv = scipy.linalg.lu_factor(A)
p = pivots.pivots_to_perm(piv)
q = pivots.inv_perm(p)

print(piv)
print(p)
print(pivots.pivots_to_perm(pivots.perm_to_pivots(p)) == p)
print(pivots.perm_to_pivots(pivots.pivots_to_perm(piv)) == piv)

L, U = np.tril(lu, k=-1) + np.eye(4), np.triu(lu)

print(A - pivots.permute(L @ U, p))
print(pivots.bpermute(A, p) - L @ U)

print(pivots.permute(A, q) - L @ U)
print(A - pivots.bpermute(L @ U, q))

print(pivots.comp_perm(pivots.inv_perm(p), p))
print(pivots.comp_perm(p, pivots.inv_perm(p)))
