import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation
import h5py
import vtk
import pyvista as pv

plt.style.use('./figstyle.mplstyle')


# --------------------------------------------------------
# Cube representation figure: Approx (left) vs Exact (right)
# --------------------------------------------------------
dt = 2e-5
k6 = 0.01
file = np.loadtxt("JAMESON4/output_nx(64)_k6(1.0E-02)_dt(1.0E-07).txt")
x_file = np.loadtxt("JAMESON4/x_set_nx(64)_k6(1.0E-02)_dt(1.0E-07).txt")
X_set = x_file[:,0]
nx = len(X_set)

Q_list = file[:,0]
GradRho_list = file[:,1]
max_grad = max(GradRho_list)
GradRho_list /= max_grad

###############################################
Q = Q_list.reshape((nx, nx, nx))
G = GradRho_list.reshape((nx, nx, nx))

# Create ImageData grid
grid = pv.ImageData()

# IMPORTANT: dimensions = number of *points*, so if your values are on points:
grid.dimensions = np.array(Q.shape) + 1
grid.cell_data["Q"] = Q.flatten(order="F")
grid.cell_data["G"] = G.flatten(order="F")

# Spatial reference
dx = X_set[1] - X_set[0]
grid.origin = (X_set[0], X_set[0], X_set[0])
grid.spacing = (dx, dx, dx)

# # Assign scalar field (flattened Fortran order!)
# grid.point_data["Q"] = Q.flatten(order="F")

grid_pt = grid.cell_data_to_point_data()

# Choose isovalue
isovalue = 0.5 * Q.max()

# Extract isosurface
iso = grid_pt.contour(isosurfaces=[isovalue], scalars="Q")

# Plot
# plotter = pv.Plotter()
# plotter.add_mesh(iso, cmap="viridis")
# plotter.add_axes()
# plotter.show()

scalar_bar_args = {
    "title": r"$ \nabla \mid \rho \mid$",
    "title_font_size": 25,
    "position_x": 0.75,
    "position_y": 0.10,
    "width": 0.09,
    "height": 0.8,
    "fmt": "%.2e",
    'vertical': True,
    'font_family': 'courier',
    'bold': True,
    # 'interactive': True
}

# sargs = dict(interactive=True)
print("Max grad de rho:",max(GradRho_list),"| Min grad de rho:", min(GradRho_list))

p = pv.Plotter(off_screen=True)
p.add_mesh(
    iso,
    scalars="G",   # <-- This selects the coloring field
    cmap="viridis",
    scalar_bar_args=scalar_bar_args,
    clim=[1.0,1.0]
)
# p.add_axes()
p.show_grid(xtitle="X", ytitle="Y", ztitle="Z")


p.screenshot(f"PLOTS/JS4_3Dvisu_k6({k6:1.1E}).png", window_size=[1400,700])
# p.show()
