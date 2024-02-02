import matplotlib.pyplot as plt
import numpy as np
from funcs import *

def plotGrid(plot, cells, knotvectors, degrees):
    fig, ax = plot
    max_lev = max(knotvectors.keys())
    ndim = len(degrees)
    knots = {lev: {dim: np.unique(knotvectors[lev][dim]) for dim in range(ndim)} for lev in range(max_lev+1)}
    for lev in cells.keys():
        for cellIdx in np.ndindex(cells[lev].shape):
            if cells[lev][cellIdx]==1:
                if ndim==2:
                    ax = plot_cell_2D(ax, cellIdx, knots[lev])
    plt.plot()

def plot_cell_2D(ax, cellIdx, knots):
    ll = [knots[0][cellIdx[0]], knots[1][cellIdx[1]]]
    ur = [knots[0][cellIdx[0]+1], knots[1][cellIdx[1]+1]]

    lr = [ur[0], ll[1]]
    ul = [ll[0], ur[1]]

    x_coo = [ll[0], lr[0], ur[0], ul[0], ll[0]]
    y_coo = [ll[1], lr[1], ur[1], ul[1], ll[1]]

    ax.plot(x_coo, y_coo, color='k', marker='')
    return ax

def plot2DAdaptiveGrid(ac_cells, ctrl_pts, knot_vectors, fn_coeffs, fn_shapes, degrees):
    fig, ax = plt.subplots()
    ndim = len(degrees)
    max_lev = len(knot_vectors.keys()) - 1
    knots = {lev: {dim: np.unique(knot_vectors[lev][dim]) for dim in range(ndim)} for lev in range(max_lev+1)}
    for lev in range(max_lev+1):
        for cell in ac_cells[lev].keys():
            supp = ac_cells[lev][cell]

            x1 = knots[lev][0][cell[0]]
            y1 = knots[lev][1][cell[1]]
            x2 = knots[lev][0][cell[0]+1]
            y2 = knots[lev][1][cell[1]+1]

            if y2==1:
                y2 -= 1e-9
            if x2==1:
                x2 -= 1e-9
            if y1==0:
                y1 += 1e-9
            if x1==0:
                x1 += 1e-9

            ll = [x1, y1]
            ur = [x2, y2]
            lr = [x2, y1]
            ul = [x1, y2]
            
            param = np.array([ll, lr, ur, ul])
            # print(param)
            out = np.zeros((4, 2))
            phi = []
            for i, g in enumerate(param):
                max_lev_cellIdx = [findSpan(fn_shapes[max_lev][dim]-1, degrees[dim], g[dim], knot_vectors[max_lev][dim]) for dim in range(ndim)]
                basis_fns = [basisFun(max_lev_cellIdx[dim], g[dim], degrees[dim], knot_vectors[max_lev][dim]) for dim in range(ndim)]
                fn_values = []
                for fn in supp:
                    fn_lev, fnIdx = fn
                    slice_tuple = tuple(slice(max_lev_cellIdx[dim]-degrees[dim], max_lev_cellIdx[dim]+1) for dim in range(ndim))
                    sub_coeff = fn_coeffs[fn_lev][fnIdx][slice_tuple]
                    # sub_coeff_tp = tensor_product(sub_coeff)
                    fn_tp = compute_tensor_product(basis_fns)
                    fn_value = np.sum(sub_coeff*fn_tp)
                    out[i] += fn_value*ctrl_pts[fn_lev][fnIdx]
                    fn_values.append(fn_value)
                phi.append(np.array(fn_values))
            out = np.vstack([out, out[0]])
            ax.plot(out[:,0], out[:,1], color='k')
    ax.set_box_aspect(1)
    ax.set_axis_off()
    plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0, 
            hspace = 0, wspace = 0)
    plt.margins(0)
    plt.show()