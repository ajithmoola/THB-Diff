import pyvista as pv
import numpy as np
from pyvista import CellType

# Contains information on the points composing each cell.
# Each cell begins with the number of points in the cell and then the points
# composing the cell
cells = np.array([8, 0, 1, 2, 3, 5, 4, 6, 7, 8, 8, 9, 10, 11, 12, 13, 14, 15])

# cell type array. Contains the cell type of each cell
cell_type = np.array([CellType.HEXAHEDRON, CellType.HEXAHEDRON])

# in this example, each cell uses separate points
cell1 = np.array(
    [
        [0, 0, 0],
        [1, 0, 0],
        [1, 1, 0],
        [0, 1, 0],
        [0, 0, 1],
        [1, 0, 1],
        [1, 1, 1],
        [0, 1, 1],
    ]
)

cell2 = np.array(
    [
        [0, 0, 2],
        [1, 0, 2],
        [1, 1, 2],
        [0, 1, 2],
        [0, 0, 3],
        [1, 0, 3],
        [1, 1, 3],
        [0, 1, 3],
    ]
)

# points of the cell array
points = np.vstack((cell1, cell2)).astype(float)

# create the unstructured grid directly from the numpy arrays
grid = pv.UnstructuredGrid(cells, cell_type, points)

# For cells of fixed sizes (like the mentioned Hexahedra), it is also possible to use the
# simplified dictionary interface. This automatically calculates the cell array.
# Note that for mixing with additional cell types, just the appropriate key needs to be
# added to the dictionary.
cells_hex = np.arange(16).reshape([2, 8])
# = np.array([[0, 1, 2, 3, 4, 5, 6, 7], [8, 9, 10, 11, 12, 13, 14, 15]])
grid = pv.UnstructuredGrid({CellType.HEXAHEDRON: cells_hex}, points)

# plot the grid (and suppress the camera position output)
_ = grid.plot(show_edges=True)