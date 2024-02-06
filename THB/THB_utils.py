import matplotlib.pyplot as plt
import numpy as np
from funcs import *
import pyvista as pv


class THB_plot:

    def __init__(self, dir, figname):
        self.dir = dir
        self.fig = plt.figure()
        self.figname = figname
        self.ax = {}
    
    def add_3Daxis(self, axis_name):
        ax = self.fig.add_subplot(projection='3d')
        self.ax[axis_name] = ax
        return ax
    
    def add_2Daxis(self, axis_name):
        ax = self.fig.add_subplot()
        self.ax[axis_name] = ax
    
    def save_fig(self, dpi=150):
        self.fig.savefig(self.dir+'/'+self.figname+'.pdf', dpi=dpi, bbox_inches='tight',
                         pad_inches=0, transparent=True, format='pdf')


def plot2DGrid(ax, cells, knotvectors, show_fig=True):

    max_lev = max(knotvectors.keys())
    knots = {lev: {dim: np.unique(knotvectors[lev][dim]) for dim in range(2)} for lev in range(max_lev+1)}

    for lev in cells.keys():
        for cellIdx in np.ndindex(cells[lev].shape):
            if cells[lev][cellIdx] == 1:
                ll = [knots[lev][0][cellIdx[0]], knots[lev][1][cellIdx[1]]]
                ur = [knots[lev][0][cellIdx[0]+1], knots[lev][1][cellIdx[1]+1]]

                lr = [ur[0], ll[1]]
                ul = [ll[0], ur[1]]

                x_coo = [ll[0], lr[0], ur[0], ul[0], ll[0]]
                y_coo = [ll[1], lr[1], ur[1], ul[1], ll[1]]

                ax.plot(x_coo, y_coo, color='k')

    ax.set_box_aspect(1)
    ax.set_axis_off()
    plt.margins(0)
    
    if show_fig:
        plt.show()
    
    return ax


def plot3DGrid(cells, knotvectors):

    max_lev = max(knotvectors.keys())
    knots = {lev: {dim: np.unique(knotvectors[lev][dim]) for dim in range(3)} for lev in range(max_lev+1)}

    corners = []

    for lev in cells.keys():
        for cellIdx in np.ndindex(cells[lev].shape):
            if cells[lev][cellIdx] == 1:
                x1 = knots[lev][0][cellIdx[0]]
                y1 = knots[lev][1][cellIdx[1]]
                z1 = knots[lev][2][cellIdx[2]]
                x2 = knots[lev][0][cellIdx[0]+1]
                y2 = knots[lev][1][cellIdx[1]+1]
                z2 = knots[lev][2][cellIdx[2]+1]

                points = np.array([[x1, y1, z1],
                                   [x2, y1, z1],
                                   [x2, y2, z1],
                                   [x1, y2, z1],
                                   [x1, y1, z2],
                                   [x2, y1, z2],
                                   [x2, y2, z2],
                                   [x1, y2, z2]])
                
                corners.append(points)
    
    num_cells = len(corners)
    corners = np.vstack(corners)
    unique_points, inverse_indices = np.unique(corners, axis=0, return_inverse=True)

    cell_types = np.full(num_cells, pv.CellType.HEXAHEDRON, dtype=np.uint8)

    cells = np.vstack([np.concatenate([np.array([8]), inverse_indices[8*i:8*i+8]], dtype=np.int_) for i in range(num_cells)], dtype=np.int_).ravel()

    grid = pv.UnstructuredGrid(cells, cell_types, unique_points)

    grid.save('unstructured_grid.vtu')

    # _ = grid.plot(show_edges=True)


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


def plot3DAdaptiveGrid(ac_cells, ctrl_pts, knot_vectors, fn_coeffs, fn_shapes, degrees):
    ndim = len(degrees)
    max_lev = len(knot_vectors.keys()) - 1
    knots = {lev: {dim: np.unique(knot_vectors[lev][dim]) for dim in range(ndim)} for lev in range(max_lev+1)}

    corners = []
    for lev in range(max_lev+1):
        for cell in ac_cells[lev].keys():
            supp = ac_cells[lev][cell]

            x1 = knots[lev][0][cell[0]]
            y1 = knots[lev][1][cell[1]]
            z1 = knots[lev][2][cell[2]]
            x2 = knots[lev][0][cell[0]+1]
            y2 = knots[lev][1][cell[1]+1]
            z2 = knots[lev][2][cell[2]+1]

            if y2 == 1:
                y2 -= 1e-9
            if x2 == 1:
                x2 -= 1e-9
            if z2 == 1:
                z2 -= 1e-9
            if y1 == 0:
                y1 += 1e-9
            if x1 == 0:
                x1 += 1e-9
            if z1 == 0:
                z1 += 1e-9

            llf = [x1, y1, z1]
            lrf = [x2, y1, z1]
            urf = [x2, y2, z1]
            ulf = [x1, y2, z1]
            llb = [x1, y1, z2]
            lrb = [x2, y1, z2]
            urb = [x2, y2, z2]
            ulb = [x1, y2, z2]

            param = np.array([llf, lrf, urf, ulf, llb, lrb, urb, ulb])
            out = np.zeros((8, 3))
            for i, g in enumerate(param):
                max_lev_cellIdx = [findSpan(fn_shapes[max_lev][dim]-1, degrees[dim], g[dim], knot_vectors[max_lev][dim]) for dim in range(ndim)]
                basis_fns = [basisFun(max_lev_cellIdx[dim], g[dim], degrees[dim], knot_vectors[max_lev][dim]) for dim in range(ndim)]
                fn_values = []
                for fn in supp:
                    fn_lev, fnIdx = fn
                    slice_tuple = tuple(slice(max_lev_cellIdx[dim]-degrees[dim], max_lev_cellIdx[dim]+1) for dim in range(ndim))
                    sub_coeff = fn_coeffs[fn_lev][fnIdx][slice_tuple]
                    fn_tp = compute_tensor_product(basis_fns)
                    fn_value = np.sum(sub_coeff*fn_tp)
                    out[i] += fn_value*ctrl_pts[fn_lev][fnIdx]

            corners.append(out)
    
    num_cells = len(corners)
    corners = np.vstack(corners)
    unique_points, inverse_indices = np.unique(corners, axis=0, return_inverse=True)

    cell_types = np.full(num_cells, pv.CellType.HEXAHEDRON, dtype=np.uint8)

    cells = [np.concatenate([np.array([8]), inverse_indices[8*i:8*i+8]], dtype=np.int_) for i in range(num_cells)]
    cells = np.vstack(cells, dtype=np.int_).ravel()
    grid = pv.UnstructuredGrid(cells, cell_types, unique_points)

    grid.save('unstructured_grid.vtu')

    _ = grid.plot(show_edges=True)