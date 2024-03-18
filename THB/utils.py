import matplotlib.pyplot as plt
import numpy as np
from THB.funcs import *
import pyvista as pv
from time import time


def timer(func):
    def wrap_func(*args, **kwargs):
        t1 = time()
        result = func(*args, **kwargs)
        t2 = time()
        print(f"Function {func.__name__!r} executed in {(t2-t1):.4f}s")
        return result

    return wrap_func


class THB_plot:

    def __init__(self, dir, figname):
        self.dir = dir
        self.fig = plt.figure()
        self.figname = figname
        self.ax = {}

    def add_3Daxis(self, axis_name):
        ax = self.fig.add_subplot(projection="3d")
        self.ax[axis_name] = ax

    def add_2Daxis(self, axis_name):
        ax = self.fig.add_subplot()
        self.ax[axis_name] = ax

    def save_all_axes_seperately(self, dpi=150):
        for ax in self.ax.keys():
            ax.remove()

            new_fig = plt.figure()
            new_fig.add_axes(ax)

            self.save_fig

    def save_fig(self, fig=None, dpi=150):
        if fig is not None:
            save_fig = fig
        else:
            save_fig = self.fig

        save_fig.savefig(
            self.dir + "/" + self.figname + ".pdf",
            dpi=dpi,
            bbox_inches="tight",
            pad_inches=0,
            transparent=True,
            format="pdf",
        )

    def plotAdaptiveGrid(self, axisname, THB):
        if THB.h_space.ndim == 2:
            ax = self.ax[axisname]
            ax = plot2DAdaptiveGrid(
                ax,
                THB.ac_cells,
                THB.GA,
                THB.h_space.knotvectors,
                THB.fn_coeffs,
                THB.h_space.sh_fns,
                THB.h_space.degrees,
            )
        elif THB.h_space.ndim == 3:
            plot3DAdaptiveGrid(
                self.dir + "/" + axisname,
                THB.ac_cells,
                THB.GA,
                THB.h_space.knotvectors,
                THB.fn_coeffs,
                THB.h_space.sh_fns,
                THB.h_space.degrees,
            )

    def plot_3D_wireframe_surface(
        self, axisname, xyz, shape, linestyle="solid", linewidth=1, color="green"
    ):
        ax = self.ax[axisname]
        xyz = xyz.reshape(*shape, 3)
        ax.plot_wireframe(
            xyz[:, :, 0],
            xyz[:, :, 1],
            xyz[:, :, 2],
            linestyle=linestyle,
            linewidth=linewidth,
            color=color,
        )
        ax.set_axis_off()
        ax.grid(False)
        plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
        plt.margins(0)


def plot2DGrid(ax, cells, knotvectors, show_fig=True):

    max_lev = max(knotvectors.keys())
    knots = {
        lev: {dim: np.unique(knotvectors[lev][dim]) for dim in range(2)}
        for lev in range(max_lev + 1)
    }

    for lev in cells.keys():
        for cellIdx in np.ndindex(cells[lev].shape):
            if cells[lev][cellIdx] == 1:
                ll = [knots[lev][0][cellIdx[0]], knots[lev][1][cellIdx[1]]]
                ur = [knots[lev][0][cellIdx[0] + 1], knots[lev][1][cellIdx[1] + 1]]

                lr = [ur[0], ll[1]]
                ul = [ll[0], ur[1]]

                x_coo = [ll[0], lr[0], ur[0], ul[0], ll[0]]
                y_coo = [ll[1], lr[1], ur[1], ul[1], ll[1]]

                ax.plot(x_coo, y_coo, color="k")

    ax.set_box_aspect(1)
    ax.set_axis_off()
    plt.margins(0)

    if show_fig:
        plt.show()

    return ax


def plot3DGrid(cells, knotvectors):

    max_lev = max(knotvectors.keys())
    knots = {
        lev: {dim: np.unique(knotvectors[lev][dim]) for dim in range(3)}
        for lev in range(max_lev + 1)
    }

    corners = []

    for lev in cells.keys():
        for cellIdx in np.ndindex(cells[lev].shape):
            if cells[lev][cellIdx] == 1:
                x1 = knots[lev][0][cellIdx[0]]
                y1 = knots[lev][1][cellIdx[1]]
                z1 = knots[lev][2][cellIdx[2]]
                x2 = knots[lev][0][cellIdx[0] + 1]
                y2 = knots[lev][1][cellIdx[1] + 1]
                z2 = knots[lev][2][cellIdx[2] + 1]

                points = np.array(
                    [
                        [x1, y1, z1],
                        [x2, y1, z1],
                        [x2, y2, z1],
                        [x1, y2, z1],
                        [x1, y1, z2],
                        [x2, y1, z2],
                        [x2, y2, z2],
                        [x1, y2, z2],
                    ]
                )

                corners.append(points)

    num_cells = len(corners)
    corners = np.vstack(corners)
    unique_points, inverse_indices = np.unique(corners, axis=0, return_inverse=True)

    cell_types = np.full(num_cells, pv.CellType.HEXAHEDRON, dtype=np.uint8)

    cells = np.vstack(
        [
            np.concatenate(
                [np.array([8]), inverse_indices[8 * i : 8 * i + 8]], dtype=np.int_
            )
            for i in range(num_cells)
        ],
        dtype=np.int_,
    ).ravel()

    grid = pv.UnstructuredGrid(cells, cell_types, unique_points)

    grid.save("unstructured_grid.vtu")


def plot2DAdaptiveGrid(
    ax, ac_cells, ctrl_pts, knot_vectors, fn_coeffs, fn_shapes, degrees
):
    ndim = len(degrees)
    max_lev = len(knot_vectors.keys()) - 1
    knots = {
        lev: {dim: np.unique(knot_vectors[lev][dim]) for dim in range(ndim)}
        for lev in range(max_lev + 1)
    }
    for lev in range(max_lev + 1):
        for cell in ac_cells[lev].keys():
            supp = ac_cells[lev][cell]

            x1 = knots[lev][0][cell[0]]
            y1 = knots[lev][1][cell[1]]
            x2 = knots[lev][0][cell[0] + 1]
            y2 = knots[lev][1][cell[1] + 1]

            if y2 == 1:
                y2 -= 1e-9
            if x2 == 1:
                x2 -= 1e-9
            if y1 == 0:
                y1 += 1e-9
            if x1 == 0:
                x1 += 1e-9

            ll = [x1, y1]
            ur = [x2, y2]
            lr = [x2, y1]
            ul = [x1, y2]

            param = np.array([ll, lr, ur, ul])
            out = np.zeros((4, 2))
            phi = []
            for i, g in enumerate(param):
                max_lev_cellIdx = [
                    findSpan(
                        fn_shapes[max_lev][dim] - 1,
                        degrees[dim],
                        g[dim],
                        knot_vectors[max_lev][dim],
                    )
                    for dim in range(ndim)
                ]
                basis_fns = [
                    basisFun(
                        max_lev_cellIdx[dim],
                        g[dim],
                        degrees[dim],
                        knot_vectors[max_lev][dim],
                    )
                    for dim in range(ndim)
                ]
                fn_values = []
                for fn in supp:
                    fn_lev, fnIdx = fn
                    slice_tuple = tuple(
                        slice(
                            max_lev_cellIdx[dim] - degrees[dim],
                            max_lev_cellIdx[dim] + 1,
                        )
                        for dim in range(ndim)
                    )
                    sub_coeff = fn_coeffs[fn_lev][fnIdx][slice_tuple]
                    fn_tp = compute_tensor_product(basis_fns)
                    fn_value = np.sum(sub_coeff * fn_tp)
                    out[i] += fn_value * ctrl_pts[fn_lev][fnIdx]
                    fn_values.append(fn_value)
                phi.append(np.array(fn_values))
            out = np.vstack([out, out[0]])
            ax.plot(out[:, 0], out[:, 1], color="k")
    ax.set_box_aspect(1)
    ax.set_axis_off()
    plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
    plt.margins(0)
    plt.show()


def plot3DAdaptiveGrid(
    filename, ac_cells, ctrl_pts, knot_vectors, fn_coeffs, fn_shapes, degrees
):
    ndim = len(degrees)
    max_lev = len(knot_vectors.keys()) - 1
    knots = {
        lev: {dim: np.unique(knot_vectors[lev][dim]) for dim in range(ndim)}
        for lev in range(max_lev + 1)
    }

    corners = []
    for lev in range(max_lev + 1):
        for cell in ac_cells[lev].keys():
            supp = ac_cells[lev][cell]

            x1 = knots[lev][0][cell[0]]
            y1 = knots[lev][1][cell[1]]
            z1 = knots[lev][2][cell[2]]
            x2 = knots[lev][0][cell[0] + 1]
            y2 = knots[lev][1][cell[1] + 1]
            z2 = knots[lev][2][cell[2] + 1]

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
                max_lev_cellIdx = [
                    findSpan(
                        fn_shapes[max_lev][dim] - 1,
                        degrees[dim],
                        g[dim],
                        knot_vectors[max_lev][dim],
                    )
                    for dim in range(ndim)
                ]
                basis_fns = [
                    basisFun(
                        max_lev_cellIdx[dim],
                        g[dim],
                        degrees[dim],
                        knot_vectors[max_lev][dim],
                    )
                    for dim in range(ndim)
                ]
                fn_values = []
                for fn in supp:
                    fn_lev, fnIdx = fn
                    slice_tuple = tuple(
                        slice(
                            max_lev_cellIdx[dim] - degrees[dim],
                            max_lev_cellIdx[dim] + 1,
                        )
                        for dim in range(ndim)
                    )
                    sub_coeff = fn_coeffs[fn_lev][fnIdx][slice_tuple]
                    fn_tp = compute_tensor_product(basis_fns)
                    fn_value = np.sum(sub_coeff * fn_tp)
                    out[i] += fn_value * ctrl_pts[fn_lev][fnIdx]

            corners.append(out)

    num_cells = len(corners)
    corners = np.vstack(corners)
    unique_points, inverse_indices = np.unique(corners, axis=0, return_inverse=True)

    cell_types = np.full(num_cells, pv.CellType.HEXAHEDRON, dtype=np.uint8)

    cells = [
        np.concatenate(
            [np.array([8]), inverse_indices[8 * i : 8 * i + 8]], dtype=np.int_
        )
        for i in range(num_cells)
    ]
    cells = np.vstack(cells, dtype=np.int_).ravel()
    grid = pv.UnstructuredGrid(cells, cell_types, unique_points)

    grid.save(filename=filename + ".vtu")
