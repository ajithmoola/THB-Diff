import matplotlib.pyplot as plt
import numpy as np

def plotGrid(plot, cells, knotvectors, degrees):
    fig, ax = plot
    max_lev = max(knotvectors.keys())
    ndim = len(degrees)
    knots = {lev: {dim: np.unique(knotvectors[lev][dim]) for dim in range(ndim)} for lev in range(max_lev)}
    for lev in cells.keys():
        for cellIdx in np.ndindex(cells[lev].shape):
            if cells[lev][cellIdx]==1:
                if ndim==2:
                    ax = plot_cell_2D(ax, cellIdx, knots[lev])
    plt.plot()

def plot_cell_2D(ax, cellIdx, knots):
    ll = [knots[cellIdx[0]], knots[cellIdx[1]]]
    ur = [knots[cellIdx[0]+1], knots[cellIdx[1]+1]]

    lr = [ll[1], ur[0]]
    ul = [ll[0], ur[1]]

    x_coo = [ll[0], lr[0], ur[0], ul[0]]
    y_coo = [ll[1], lr[1], ur[1], ul[1]]

    ax.plot(x_coo, y_coo, marker='')
    return ax