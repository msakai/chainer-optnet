import numpy as np
import chainer

def pivots_to_perm(piv):
    p = id_perm(len(piv))
    for (i,j) in enumerate(piv):
        p[i], p[j] = p[j], p[i]
    return p

def perm_to_pivots(idx):
    o2c = id_perm(len(idx)) # 元のindexの要素が現在どこにあるか
    c2o = id_perm(len(idx)) # 現在のindexの要素は元のどこの要素か
    piv = np.empty_like(idx, dtype=np.int32)
    for (i,j) in enumerate(idx):
        k = o2c[j]
        assert j == c2o[k]
        assert i == o2c[c2o[i]]
        piv[i] = k # i と k を pivot
        o2c[c2o[i]], o2c[c2o[k]] = o2c[c2o[k]], o2c[c2o[i]]
        c2o[i], c2o[k] = c2o[k], c2o[i]
    return piv

def bpermute(a, idx):
    return a[idx]

def permute(a, idx):
    ret = np.empty_like(a)
    ret[idx] = a
    return ret


def id_perm(n):
    return np.arange(n, dtype=np.int32)

def comp_perm(p1, p2):
    """
    Apply p2 then p1.
    permute(a, comp_perm(p1,p2)) == permute(permute(a, p2), p1)
    """
    return p1[p2]

def inv_perm(idx):
    ret = np.empty_like(idx)
    #for (i,j) in enumerate(idx):
    #    ret[j] = i
    ret[idx] = id_perm(len(idx))
    return ret


if __name__ == '__main__':
    import scipy.linalg

    A = np.array([[2, 5, 8, 7], [5, 2, 2, 8], [7, 5, 6, 6], [5, 4, 4, 8]])

    lu, piv = scipy.linalg.lu_factor(A)
    p = pivots_to_perm(piv)
    q = inv_perm(p)

    print(piv)
    print(p)
    print(pivots_to_perm(perm_to_pivots(p)) == p)
    print(perm_to_pivots(pivots_to_perm(piv)) == piv)

    L, U = np.tril(lu, k=-1) + np.eye(4), np.triu(lu)

    print(A - permute(L @ U, p))
    print(bpermute(A, p) - L @ U)

    print(permute(A, q) - L @ U)
    print(A - bpermute(L @ U, q))

    print(comp_perm(inv_perm(p), p))
    print(comp_perm(p, inv_perm(p)))
