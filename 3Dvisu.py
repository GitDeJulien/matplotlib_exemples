import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation
import h5py
import pyvista as pv

# --------------------------------------------------------
# Cube representation figure: Approx (left) vs Exact (right)
# --------------------------------------------------------

file = np.loadtxt("JAMESON5/output_nx(80)_k6(1.0E-02)_dt(2.0E-05).txt")
nx = 18
x_file = np.loadtxt("JAMESON5/x_set_nx(80)_k6(1.0E-02)_dt(2.0E-05).txt")
X_set = x_file[:,0]
nx = len(X_set)
density_approx = np.zeros((nx,nx,nx))
for k in range(nx):
    for j in range(nx):
        for i in range(nx):
            ind = i + nx*j + nx**2*k
            density_approx[i,j,k] = file[ind]

density_exact = np.zeros((nx, nx, nx))
tend = 1.0
U0 = 1.0; V0 = -1./2.; W0 = 1.0
for k,z in enumerate(X_set):
    for j,y in enumerate(X_set):
        for i,x in enumerate(X_set):
            density_exact[i,j,k] = 1.0 + 0.20*np.sin(np.pi*(x + y + z - tend*(U0 + V0 + W0)))

def plot_cube(ax, X, Y, Z, D):
    XX, YY = np.meshgrid(X, Y, indexing='ij')
    XX2, ZZ = np.meshgrid(X, Z, indexing='ij')
    YY2, ZZ2 = np.meshgrid(Y, Z, indexing='ij')

    dmin, dmax = D.min(), D.max()
    norm = lambda A: (A - dmin) / (dmax - dmin + 1e-12)

    # XY faces (z=0, z=max)
    ax.plot_surface(XX, YY, np.full_like(XX, Z[0]),
                    facecolors=plt.cm.viridis(norm(D[:, :, 0])),
                    rstride=1, cstride=1, shade=False)
    ax.plot_surface(XX, YY, np.full_like(XX, Z[-1]),
                    facecolors=plt.cm.viridis(norm(D[:, :, -1])),
                    rstride=1, cstride=1, shade=False)

    # XZ faces (y=0, y=max)
    ax.plot_surface(XX2, np.full_like(XX2, Y[0]), ZZ,
                    facecolors=plt.cm.viridis(norm(D[:, 0, :])),
                    rstride=1, cstride=1, shade=False)
    ax.plot_surface(XX2, np.full_like(XX2, Y[-1]), ZZ,
                    facecolors=plt.cm.viridis(norm(D[:, -1, :])),
                    rstride=1, cstride=1, shade=False)

    # YZ faces (x=0, x=max)
    ax.plot_surface(np.full_like(YY2, X[0]), YY2, ZZ2,
                    facecolors=plt.cm.viridis(norm(D[0, :, :])),
                    rstride=1, cstride=1, shade=False)
    ax.plot_surface(np.full_like(YY2, X[-1]), YY2, ZZ2,
                    facecolors=plt.cm.viridis(norm(D[-1, :, :])),
                    rstride=1, cstride=1, shade=False)

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")

fig = plt.figure(figsize=(14, 6))

# LEFT: approx solution
ax1 = fig.add_subplot(121, projection='3d')
plot_cube(ax1, X_set, X_set, X_set, density_approx)
ax1.set_title("Approximate Solution")

# RIGHT: exact solution
ax2 = fig.add_subplot(122, projection='3d')
plot_cube(ax2, X_set, X_set, X_set, density_exact)
ax2.set_title("Exact Solution")

fig.suptitle(f"t={tend}")
plt.show()