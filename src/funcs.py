import numpy as np
from numba import njit


def refine_knotvector(knotvector, p):
    # Tested: Working!
    knots = np.unique(knotvector)
    mids = 0.5*(knots[1:]+knots[:-1])
    refined_knotvector = np.concatenate([knotvector[:p], np.unique(np.sort(np.concatenate([knots, mids]))), knotvector[-p:]])
    return refined_knotvector

def generate_parametric_coordinates(shape):
    ndim = len(shape)
    pts = np.hstack(tuple(map(lambda x: x.reshape(-1, 1), np.meshgrid(
        *[np.linspace(1e-5, 1, shape[dim], endpoint=False) for dim in range(ndim)]))))
    return pts

def grevilleAbscissae(fn_sh, degrees, knotvectors):
    ndim = len(fn_sh)
    CP = np.zeros((*fn_sh, ndim))
    
    for pt in np.ndindex(fn_sh):
        CP[pt] = np.array([np.sum(knotvectors[dim][pt[dim]:pt[dim]+degrees[dim]+1])/degrees[dim] for dim in range(ndim)])
    
    return CP

def tensor_product(args):
    if len(args)==2:
        return tensor_product_2D(args[0], args[1])
    if len(args)==3:
        return tensor_product_3D(args[0], args[1], args[2])
    
def tensor_product_2D(X, Y):
    return np.einsum("i, j -> ij", X, Y)

def tensor_product_3D(X, Y, Z):
    return np.einsum("i, j, k -> ijk", X, Y, Z)

def findSpan(n, p, u, U):
    if u==U[n+1]:
        return n
    low = p
    high = n+1
    mid = int((low+high)/2)

    knot_span_idx = 0
    while (u<U[mid] or u>=U[mid+1]):
        if u<U[mid]:
            high = mid
        else:
            low = mid
        mid = int((low+high)/2)
    knot_span_idx = mid
    return knot_span_idx

@njit
def basisFun(i, u, p, U):
    """computes basis functions required to evaluate a point on the b-spline

    Args:
        i (int): knot span index
        u (float): parametric coordinate
        p (int): degree of the b-spline
        U (ndarray): knot vector

    Returns:
        ndarray: returns basis functions required to compute a point on the b-spline
    """
    N = np.zeros((p+1))
    N[0] = 1
    left = np.zeros((p+1))
    right = np.zeros((p+1))
    for j in range(1, p+1):
        left[j] = u - U[i+1-j]
        right[j] = U[i+j] - u
        saved = 0
        for r in range(0, j):
            temp = N[r]/(right[r+1] + left[j-r])
            N[r] = saved + right[r+1] * temp
            saved = left[j-r]*temp
        N[j] = saved

    return N
            
@njit
def assemble_Tmatrix(knotVec, newKnotVec, knotVec_len, newKnotVec_len, p):
    # TODO: convert to c++ function
    T1 = np.zeros((newKnotVec_len-1, knotVec_len-1))

    for i in range(newKnotVec_len-1):
        for j in range(knotVec_len-1):
            if (newKnotVec[i]>=knotVec[j]) and (newKnotVec[i]<knotVec[j+1]):
                T1[i, j] = 1
    
    for q in range(1, p+1):
        T2 = np.zeros(((newKnotVec_len-q-1), (knotVec_len-q-1)))
        for i in range(newKnotVec_len-q-1):
            for j in range(knotVec_len-q-1):
                if (knotVec[j+q]-knotVec[j]==0) and (knotVec[j+q+1]-knotVec[j+1]!=0):
                    T2[i, j] = (knotVec[j+q+1]-newKnotVec[i+q])/(knotVec[j+q+1]-knotVec[j+1])*T1[i, j+1]
                if (knotVec[j+q]-knotVec[j]!=0) and (knotVec[j+q+1]-knotVec[j+1]==0):
                    T2[i, j] = (newKnotVec[i+q]-knotVec[j])/(knotVec[j+q]-knotVec[j])*T1[i, j]
                if (knotVec[j+q]-knotVec[j]!=0) and (knotVec[j+q+1]-knotVec[j+1]!=0):
                    T2[i, j] = (newKnotVec[i+q]-knotVec[j])/(knotVec[j+q]-knotVec[j])*T1[i, j] + (knotVec[j+q+1]-newKnotVec[i+q])/(knotVec[j+q+1]-knotVec[j+1])*T1[i, j+1]
            
        T1 = T2
    return T1



if __name__=='__main__':
    kv1 = np.array([0, 0, 0, 0.2, 0.4, 0.6, 0.8, 1, 1, 1])
    p = 2
    kv2 = refine_knotvector(kv1, p)
    ualpha = assemble_Tmatrix(kv1, kv2, kv1.size, kv2.size, p)
    print(ualpha.T.shape)