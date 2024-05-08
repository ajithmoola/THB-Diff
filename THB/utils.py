import matplotlib.pyplot as plt
import numpy as np
from THB.funcs import *
import pyvista as pv
from time import time

from OCC.Core.Geom import Geom_BSplineSurface
from OCC.Core.TColgp import TColgp_Array2OfPnt
from OCC.Core.TColStd import TColStd_Array1OfReal, TColStd_Array1OfInteger
from OCC.Core.gp import gp_Pnt
from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_MakeFace
from OCC.Core.STEPControl import STEPControl_Writer, STEPControl_AsIs


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

            self.save_fig(new_fig)

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

    def plot_3D_surface(
        self, axisname, xyz, shape, color="grey", rstride=None, cstride=None
    ):
        ax = self.ax[axisname]
        xyz = xyz.reshape(*shape, 3)
        ax.plot_surface(
            xyz[:, :, 0],
            xyz[:, :, 1],
            xyz[:, :, 2],
            color=color,
            rstride=rstride,
            cstride=cstride,
            edgecolor=None,
            linewidth=0,
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


def BSplineSurf_to_STEP(CP, knotvectors, degrees, fname):
    """Exports maximum level b-spline to a step file

    Args:
        CP (ndarray): control points
        knotvectors (list): knotvectors in a tuple
        degrees (tuple): degree of b-splines in the tensor product
        fname (str): file name
    """
    degree_u = degrees[0]
    degree_v = degrees[1]

    knots_u = np.unique(knotvectors[0])
    knots_v = np.unique(knotvectors[1])

    multiplicities_u = np.ones_like(knots_u)
    multiplicities_v = np.ones_like(knots_v)
    multiplicities_u[0] = degree_u + 1
    multiplicities_u[-1] = degree_u + 1
    multiplicities_v[0] = degree_u + 1
    multiplicities_v[-1] = degree_v + 1

    knots_u_occ = TColStd_Array1OfReal(1, len(knots_u))
    knots_v_occ = TColStd_Array1OfReal(1, len(knots_v))

    multiplicities_u_occ = TColStd_Array1OfInteger(1, len(multiplicities_u))
    multiplicities_v_occ = TColStd_Array1OfInteger(1, len(multiplicities_v))

    for i, val in enumerate(knots_u, start=1):
        knots_u_occ.SetValue(i, val)
    for i, val in enumerate(knots_v, start=1):
        knots_v_occ.SetValue(i, val)
    for i, val in enumerate(multiplicities_u):
        multiplicities_u_occ.SetValue(i + 1, int(val))
    for i, val in enumerate(multiplicities_v):
        multiplicities_v_occ.SetValue(i + 1, int(val))

    control_points_occ = TColgp_Array2OfPnt(1, CP.shape[0], 1, CP.shape[1])
    for i in range(CP.shape[0]):
        for j in range(CP.shape[1]):
            x, y, z = map(float, CP[i, j])
            control_points_occ.SetValue(i + 1, j + 1, gp_Pnt(x, y, z))

    bspline_surface = Geom_BSplineSurface(
        control_points_occ,
        knots_u_occ,
        knots_v_occ,
        multiplicities_u_occ,
        multiplicities_v_occ,
        degree_u,
        degree_v,
        False,
        False,
    )

    face = BRepBuilderAPI_MakeFace(bspline_surface, 1e-6).Face()

    writer = STEPControl_Writer()
    writer.Transfer(face, STEPControl_AsIs)

    status = writer.Write(fname + ".step")

    if status:
        print("Successfully exported B-spline surface to STEP file.")
    else:
        print("Failed to export B-spline surface to STEP file.")


def plot_active_3D_cells(ac_cells, knotvectors, wd, filename):
    max_lev = max(ac_cells.keys())
    ndim = len(knotvectors[0])
    knots = {
        lev: {dim: np.unique(knotvectors[lev][dim]) for dim in range(ndim)}
        for lev in range(max_lev + 1)
    }

    corners = []
    for lev in range(max_lev + 1):
        for cell in ac_cells[lev].keys():

            x1 = knots[lev][0][cell[0]]
            y1 = knots[lev][1][cell[1]]
            z1 = knots[lev][2][cell[2]]
            x2 = knots[lev][0][cell[0] + 1]
            y2 = knots[lev][1][cell[1] + 1]
            z2 = knots[lev][2][cell[2] + 1]

            llf = [x1, y1, z1]
            lrf = [x2, y1, z1]
            urf = [x2, y2, z1]
            ulf = [x1, y2, z1]
            llb = [x1, y1, z2]
            lrb = [x2, y1, z2]
            urb = [x2, y2, z2]
            ulb = [x1, y2, z2]

            boxes = np.array([llf, lrf, urf, ulf, llb, lrb, urb, ulb])

            corners.append(boxes)

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

    grid.save(filename=wd + "/" + filename + ".vtu")
