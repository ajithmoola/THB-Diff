import numpy as np
from numba import njit
import jax.numpy as jnp
from jax import jit

def refine_knotvector(knotvector, p):
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
        CP[pt] = np.array([np.sum(knotvectors[dim][pt[dim]+1:pt[dim]+degrees[dim]+1])/degrees[dim] for dim in range(ndim)])
    
    return CP

def compute_projection(args, ndim):
    if ndim==2:
        return np.einsum('ijkl, klmn -> ijmn', *args, optimize=True)
    elif ndim==3:
        return np.einsum('ijklmn, lmnopq -> ijkopq', *args, optimize=True)

def compute_coeff_tensor_product(args):
    if len(args)==2:
        return np.einsum('ij, kl -> ikjl', *args, optimize=True)
    elif len(args)==3:
        return np.einsum('ij, kl, mn -> ikmjln', *args, optimize=True)

def compute_tensor_product(args):
    if len(args)==2:
        return np.einsum('i, j -> ij', *args)
    if len(args)==3:
        return np.einsum('i, j, k -> ijk', *args, optimize=True)
    
def compute_bezier_projection(args, ndim):
    if ndim==2:
        return np.einsum('ij, ijkl -> kl', *args, optimize=True)
    elif ndim==3:
        return np.einsum('ijk, ijklmn -> lmn', *args, optimize=True)


def findSpan(n, p, u, U):
    if u==U[n+1]:
        return n
    low = p
    high = n+1
    mid = int((low+high)/2)
    while (u<U[mid] or u>=U[mid+1]):
        if u<U[mid]:
            high = mid
        else:
            low = mid
        mid = int((low+high)/2)
    return mid

@jit
def find_span_vectorized_jax(param, U):
    n = len(U) - 2
    
    indices = jnp.searchsorted(U, param, side='right') - 1
    
    indices = jnp.where(indices > n, n, indices)
    
    return indices

def basisFun(i, u, p, U):
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

def bezier_extraction(knot, p):
    m  = len(knot)-p-1
    a  = p+1
    b  = a+1
    ne = 0
    C = []
    C.append(np.eye(p+1,p+1))
    alphas = {}
    
    while b <= m:
        C.append(np.eye(p+1,p+1))
        i=b
        while b <= m and knot[b] == knot[b-1]:
            b=b+1
            
        multiplicity = b-i+1
        if multiplicity < p:
            numerator = (knot[b-1]-knot[a-1])
            for j in range(p,multiplicity,-1):
                alphas[j-multiplicity]=numerator/(knot[a+j-1]-knot[a-1])

            r=p-multiplicity
            for j in range(1,r+1):
                save = r-j+1
                s = multiplicity + j
                for k in range(p+1,s,-1):
                    alpha=alphas[k-s]
                    C[ne][:,k-1]= alpha*C[ne][:,k-1] + (1-alpha)*C[ne][:,k-2]
                if b <= m:
                    C[ne+1][save-1:save+j, save-1] = C[ne][p-j:p+1, p]
            ne=ne+1
            if b <= m:
                a=b
                b=b+1

        elif multiplicity == p:
            if b <= m:
                ne=ne+1
                a=b
                b=b+1
    return C


if __name__=='__main__':
    kv = np.array([0, 0, 0, 0, 1, 2, 3, 4, 4, 4, 4])
    deg = 3
    num_knots = len(np.unique(kv))
    C, nb = bezier_extraction(kv, deg)
    print(C, nb, num_knots)