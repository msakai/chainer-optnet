import numpy as np
import chainer

try:
    import cupy
    import cupy_decomp_lu
    cupy_available = True
except ImportError:
    cupy_available = False

def get_array_module(a):
    if cupy_available and isinstance(a, cupy.ndarray):
        return cupy
    else:
        return np


def pivots_to_perm(piv):
    xp = get_array_module(piv)
    if cupy_available and xp == cupy:
        piv = piv.get()
        p = id_perm(np, len(piv))
        for (i,j) in enumerate(piv):
            p[i], p[j] = p[j], p[i]
        return cupy.asarray(p)
    else:
        p = id_perm(xp, len(piv))
        for (i,j) in enumerate(piv):
            p[i], p[j] = p[j], p[i]
    return p

def perm_to_pivots(idx):
    xp = get_array_module(idx)
    o2c = id_perm(xp, len(idx)) # 元のindexの要素が現在どこにあるか
    c2o = id_perm(xp, len(idx)) # 現在のindexの要素は元のどこの要素か
    piv = xp.empty_like(idx, dtype=np.int32)
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
    xp = get_array_module(a)
    ret = xp.empty_like(a)
    ret[idx] = a
    return ret


def id_perm(xp, n):
    return xp.arange(n, dtype=np.int32)

def comp_perm(p1, p2):
    """
    Apply p2 then p1.
    permute(a, comp_perm(p1,p2)) == permute(permute(a, p2), p1)
    """
    return p1[p2]

def inv_perm(idx):
    xp = get_array_module(idx)
    ret = xp.empty_like(idx)
    #for (i,j) in enumerate(idx):
    #    ret[j] = i
    ret[idx] = id_perm(xp, len(idx))
    return ret
