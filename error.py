import numpy as np
import matplotlib.pyplot as plt
import os

plt.style.use('../figstyle.mplstyle')

# -------------------------------------------------------------------
# BOOLS
# ---------------
#----------------------------------------------------
PLOT_LINES = True
PLOT_PANEL = True
SAVE_FIG = True
SHOW = False
# -------------------------------------------------------------------
# Problem setup (exact solution)
# -------------------------------------------------------------------
L    = 2.0
TEND = 1.0
U0   = 1.0
V0   = -0.5
W0   = 1.0
dt = 2e-5
k6 = 0.01

# ============================================================
# Utility: compute exact solution
# ============================================================
def rho_exact_fun(X, Y, Z):
    return 1.0 + 0.2 * np.sin(np.pi * (X + Y + Z - TEND * (U0 + V0 + W0)))


# ============================================================
# Utility: plot on a line
# ============================================================
def plot_a_line(scheme, nx, line, abscisse, sol_app, sol_ex, dir="x"):
    fig ,ax = plt.subplots()
    if dir=="x":
        y,z = line
        ax.plot(abscisse, sol_app[:,y,z], label="approximate")
        ax.plot(abscisse, sol_ex[:,y,z], label="exact", color="red")
    elif dir=="y":
        x,z = line
        ax.plot(abscisse, sol_app[x,:,z], label="approximate")
        ax.plot(abscisse, sol_ex[x,:,z], label="exact", color="red")
    elif dir=="z":
        x,y = line
        ax.plot(abscisse, sol_app[x,y,:], label="approximate")
        ax.plot(abscisse, sol_ex[x,y,:], label="exact", color="red")
    else:
        print("Unknown dir:", dir)

    ax.set_xlabel(dir)
    ax.legend()

    if SAVE_FIG: fig.savefig(f"PLOTS/{scheme}_nx({nx})_dt({dt:1.1E})_k6({k6:1.1E})_line.pdf", dpi=300, bbox_inches='tight')
    plt.close(fig)

# ============================================================
# Utility: plot on a slice
# ============================================================
def plot_slice(scheme, nx, slice_num, abs, ord, sol_app, sol_ex, dir="x"):
    fig ,ax = plt.subplots(1,2)
    if dir=="x":
        cs = ax[0].contourf(abs,ord,sol_app[slice_num,:,:], 21)
        ax[1].contourf(abs,ord,sol_ex[slice_num,:,:], 21)
    elif dir=="y":
        cs = ax[0].contourf(abs,ord,sol_app[:,slice_num,:], 21)
        ax[1].contourf(abs,ord,sol_ex[:,slice_num,:], 21)
    elif dir=="z":
        cs = ax[0].contourf(abs,ord,sol_app[:,:,slice_num], 21)
        ax[1].contourf(abs,ord,sol_ex[:,:,slice_num], 21)


    cbar = fig.colorbar(cs, ax=ax, location="right")
    cbar.set_label("Density")

    ax[0].set_xlabel("X")
    ax[0].set_ylabel("Y")
    ax[1].set_xlabel("X")
    ax[1].set_ylabel("Y")
    # ax.legend()

    if SAVE_FIG: fig.savefig(f"PLOTS/{scheme}_nx({nx})_dt({dt:1.1E})_k6({k6:1.1E})_panel.pdf", dpi=300, bbox_inches='tight')
    plt.close(fig)


# -------------------------------------------------------------------
# Schemes and resolutions
# -------------------------------------------------------------------
schemes = ["JAMESON5"]#,"JAMESON3"]#, "JAMESON3"]#, "HLLC", "WENO3"]       # add "WENO3", "WENO5", ...
Nx_list = [10,20,40,80]#,80,160]
k6_list = [0.01]#, 0.001, 0.0001]
dt_list = [2e-5, 5e-6, 5e-6, 2.5e-6]

markers = ["o", "s", "^", "d", "X"]
colors = ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple"]

norms_name = ["L1", "L2", "Linf"]

# -------------------------------------------------------------------
# Plot setup
# -------------------------------------------------------------------
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
axes[0].set_xscale("log"); axes[0].set_yscale("log")
axes[1].set_xscale("log"); axes[1].set_yscale("log")
axes[2].set_xscale("log"); axes[2].set_yscale("log")
axes[0].set_title(r"$L^1$ error")
axes[0].set_xlabel(r"$\Delta x$")
axes[1].set_title(r"$L^2$ error")
axes[1].set_xlabel(r"$\Delta x$")
axes[2].set_title(r"$L^\infty$ error")
axes[2].set_xlabel(r"$\Delta x$")


# -------------------------------------------------------------------
# Main loop over schemes
# -------------------------------------------------------------------
for s_id, scheme in enumerate(schemes):

    h_list   = []
    L1_list  = []
    L2_list  = []
    Linf_list = []

    for Nx in Nx_list:

        # -----------------------------
        # Read numerical solution
        # -----------------------------s
        file_name   = f"{scheme}/output_nx({Nx})_k6({k6_list[s_id]:1.1E})_dt({dt:1.1E}).txt"
        x_file_name = f"{scheme}/x_set_nx({Nx})_k6({k6_list[s_id]:1.1E})_dt({dt:1.1E}).txt"
        # file_name   = f"{scheme}/output_nx({Nx})_k6({k6:1.1E})_dt({dt_list[s_id]:1.1E}).txt"
        # x_file_name = f"{scheme}/x_set_nx({Nx})_k6({k6:1.1E})_dt({dt_list[s_id]:1.1E}).txt"

        rho_flat = np.loadtxt(file_name)
        x_file = np.loadtxt(x_file_name)
        X_set = x_file[:,0]
        Y_set = x_file[:,1]
        Z_set = x_file[:,2]
        n    = X_set.size

        dx = x_file[3,3]
        dy = x_file[3,4]
        dz = x_file[3,5]
        h_list.append(dx)

        # Reshape numerical solution to 3-D array
        rho = rho_flat.reshape((n, n, n))#, order='F')

        # -----------------------------
        # Build 3-D mesh
        # -----------------------------
        X, Y, Z = np.meshgrid(X_set, X_set, X_set, indexing='ij')
        XX, YY = np.meshgrid(X_set, Y_set, indexing='ij')
        XX2, ZZ = np.meshgrid(X_set, Z_set, indexing='ij')
        YY2, ZZ2 = np.meshgrid(Y_set, Z_set, indexing='ij')

        # Exact solution
        rho_ex = rho_exact_fun(X, Y, Z)

        if PLOT_LINES: plot_a_line(scheme, Nx, (n//2,n//2), X_set, rho, rho_ex, dir="x")
        if PLOT_PANEL: plot_slice(scheme, Nx, n//2, YY2, ZZ2, rho, rho_ex, dir="x")

        # -----------------------------
        # Error fields
        # -----------------------------
        err  = np.abs(rho - rho_ex)
        err2 = err**2

        # -----------------------------
        # Norms
        # -----------------------------
        dV = dx**3
        Vol = L**3
        L1 = np.sum(err) * dV
        L2 = np.sqrt(np.sum(err2) * dV)
        Linf = np.max(err)

        L1_list.append(L1)
        L2_list.append(L2)
        Linf_list.append(Linf)

        print(f"{scheme}: Nx={Nx}, dx={dx:.3e}, L1={L1:.3e}, L2={L2:.3e}, Linf={Linf:.3e}")

    # Convert to numpy arrays
    h = np.array(h_list)
    h_ref = np.logspace(np.log10(min(h)), np.log10(max(h)), 100)

    L1 = np.array(L1_list)
    L2 = np.array(L2_list)
    LINF = np.array(Linf_list)
    NORMS = [L1,L2,LINF]

    print(f"Scheme {scheme}:", end=" ")

    for col, norm in enumerate(NORMS):
        ax = axes[col]
        # --- Compute convergence slopes ----
        s, b = np.polyfit(np.log10(h), np.log10(norm), 1)

        ax.scatter(h, norm,
                edgecolor=colors[s_id],
                marker=markers[s_id],
                #label=f"{scheme} (s={s:.2f})")
                #label=f"k6={k6_list[s_id]:1.0e} (s={s:.2f})")
                label=f"dt={dt_list[s_id]:1.1e} (s={s:.2f})")

        ax.plot(h_ref, 10**(s*np.log10(h_ref) + b), '--', color=colors[s_id])
            

        print(f"order({norms_name[col]}): {s:.3f}", end="  ")

    print("\n")


for col, norm_name in enumerate(norms_name):
    ax = axes[col]
    ax.set_axisbelow(True)
    ax.grid(True, which="both", ls="--")
    ax.plot(h_ref, 10**(2.0*np.log10(h_ref) + b), 'k:', label=f"Linear (s={2.:.1f})")
    ax.legend()

plt.tight_layout()

pdf_name = f"PLOTS/JS5_error_dt({dt:1.1E})_k6({k6:1.1E}).pdf"
# pdf_name = f"PLOTS/JS5_error_k6({k6:1.1E}).pdf"
if SAVE_FIG: fig.savefig(pdf_name, dpi=300, bbox_inches='tight')

print(f"[OK] PDF saved → {pdf_name}")
if SHOW: plt.show()
plt.close(fig)
