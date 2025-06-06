import Converter.Internal as Internal
import Converter.PyTree as C
import FastS.PyTree as FastS
import Geom.PyTree as D
import Post.PyTree as P
import Transform.PyTree as T

import matplotlib.pyplot as plt
import numpy as np

# font=font_manager.FontProperties(weight='bold')
plt.style.use('./figstyle.mplstyle')

#==============================================
## Parameters
#==============================================

#Physical constants
Pref   = 101320
Rhoref = 1.1765
gamma  = 1.4
Rgp    = 287.053
mach   = 0.1
Tref   = Pref/(Rhoref*Rgp)
c0  = np.sqrt(gamma*Rgp*Tref)
U0 = mach*c0
L0 = 1.
dt = 0.0002
coeff = np.pi


#Datas
it = 40100
stat_filenames = [f'STAT/stat_ordre2_ausmp1.0_spg0.{it}.cgns', 
            f'STAT/stat_ordre2_ausmp1.0_spg0_Allcons.{it}.cgns', 
            f'STAT/stat_ordre2_ausmp1.0_spg1_Allcons.{it}.cgns', 
            f'STAT/stat_ordre2_seuseur1.0_spg0_Allcons.{it}.cgns', 
            f'STAT/stat_ordre2_seuseur1.0_spg1_Allcons.{it}.cgns']

#Label
labels = ["A.O2.NC.S0",
        "A.O2.CO.S0", 
        "A.O2.CO.S1", 
        "S.O2.CO.S0",
        "S.O2.CO.S1"]

#ON-OFF
PLOTPRMS_CIRC  = True
PLOTPRMS_LINE  = True
CONVERT2POST   = False
FIGSHOW        = False

#Plot
r = 30.
prms_max = 2.0
prms_ticks = [0.5, 1.0, 1.5, 2.0]

#==============================================
#==============================================
datas_stat = []
if PLOTPRMS_CIRC or PLOTPRMS_LINE:
    for i, filename in enumerate(stat_filenames):
        stat = C.convertFile2PyTree(filename)
        FastS._postStats(stat)
        datas_stat.append(stat)
        if CONVERT2POST :
            C.convertPyTree2File(stat, f"STAT/post_{labels[i]}.{it}.cgns")


def plotCirc(datas, center=(0,0,0), radius=1.0, 
            thetamin=0., 
            thetamax=360.,
            rticks = None,
            rmax   = None):
    
    print(f"Extracting circle of radius {radius}", end="...")
    a = D.circle(center, radius, thetamin, thetamax)
    print("done.")

    print(f"Writting PLOT/prmsCirc{int(radius)}_comparison.pdf", end="...", flush=True)
    fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
    ax.grid(True)

    for i,data in enumerate(datas):
        a1 = P.extractMesh(data, a)
        a10 = T.cart2Cyl(a1, (0.,0.,0.),(0,0,1))

        for z in Internal.getZones(a10):
            theta = Internal.getNodeFromName(z, 'CoordinateY')[1]
            Prms  = Internal.getNodeFromName(z,'Prms')[1]

        ax.plot(theta, Prms, label=labels[i])

    fig.legend()
    ax.set_ylabel("$\mathbf{P_{rms}}$", rotation=0)
    # ax.set_rmax(2.0)
    # ax.set_rticks([0.5, 1.0, 1.5, 2.0])  # Less radial ticks
    ax.set_rmax(rmax)
    ax.set_rticks(rticks)
    ax.yaxis.set_label_coords(0.9,0.5)
    ax.set_title(f"r={radius:.1f}D$", loc='left')

    fig.savefig(f"PLOT/prmsCirc{int(radius)}_comparison.pdf", format="pdf")
    print("done.")
    print("\n")


if PLOTPRMS_CIRC : plotCirc(datas=datas_stat, center=(0,0,0), 
                            radius=r, 
                            rticks=prms_ticks, 
                            rmax=prms_max)
    
if FIGSHOW : plt.show()

# radius = 30*L0
# a = D.circle((0,0,0), radius, 0., 360.)
# a1 = P.extractMesh(tmy1, a)
# a2 = P.extractMesh(tmy2, a)
# a3 = P.extractMesh(tmy3, a)
# # a4 = P.extractMesh(tmy4, a)

# a10 = T.cart2Cyl(a1, (0.,0.,0.),(0,0,1))
# a20 = T.cart2Cyl(a2, (0.,0.,0.),(0,0,1))
# a30 = T.cart2Cyl(a3, (0.,0.,0.),(0,0,1))
# # a40 = T.cart2Cyl(a4, (0.,0.,0.),(0,0,1))

# for z in Internal.getZones(a10):
#     theta1 = Internal.getNodeFromName(z, 'CoordinateY')[1]
#     Prms1  = Internal.getNodeFromName(z,'Prms')[1]

# for z in Internal.getZones(a20):
#     theta2 = Internal.getNodeFromName(z, 'CoordinateY')[1]
#     Prms2  = Internal.getNodeFromName(z,'Prms')[1]

# for z in Internal.getZones(a30):
#     theta3 = Internal.getNodeFromName(z, 'CoordinateY')[1]
#     Prms3  = Internal.getNodeFromName(z,'Prms')[1]

# # for z in Internal.getZones(a40):
# #     theta4 = Internal.getNodeFromName(z, 'CoordinateY')[1]
# #     Prms4  = Internal.getNodeFromName(z,'Prms')[1]



# fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
# # ax.plot(theta1, Prms1, label='CYL1')
# # ax.plot(theta2, Prms2, label='CYL2')
# # ax.plot(theta3, Prms3, label='CYL3')
# # ax.plot(theta4, Prms4, label='CYL4')
# labels = ["A.O2.NC.S0", "A.O2.CO.S0", "A.O2.CO.S1", "S.O2.CO.S1"]
# ax.plot(theta1, Prms1, label=labels[0])
# ax.plot(theta2, Prms2, label=labels[1])
# ax.plot(theta3, Prms3, label=labels[2])
# # ax.plot(theta4, Prms4, label=labels[3])

# ax_c, ax_p, rmax_inset = create_polar_zoom_inset(ax, xlims=[2,5], ylims=[0,1], inset_bounds=[0.25, 0.12, 0.6, 0.3])
# ax_p.plot(theta1, Prms1, label=labels[0])
# ax_p.plot(theta2, Prms2, label=labels[1])
# ax_p.plot(theta3, Prms3, label=labels[2])
# ax_p.set_rmax(rmax_inset)

# ax.set_title("$\mathbf{r=30\\times D}$", loc='left')

# fig.legend()
# ax.grid(True)
# ax.set_ylabel("$\mathbf{P_{rms}}$", rotation=0)
# ax.set_rmax(2.0)
# ax.set_rticks([0.5, 1.0, 1.5, 2.0])  # Less radial ticks
# ax.yaxis.set_label_coords(0.9,0.5)

# plt.savefig(f"PLOT/prmsCirc{int(radius)}_compare_order2.pdf", format="pdf")

# ## Extract data
# ### Rayon = 10
# r10 = 10
# with open(f"STAT/statCircle_{name_tag1}.15100-20100.ref", 'rb') as f:
#     cons = pkl.load(f)

# for z in Internal.getZones(cons): 
#       theta0 = Internal.getNodeFromName(z, 'CoordinateY')[1]
#       Prms0 = Internal.getNodeFromName(z,'Prms')[1]

# with open(f"STAT/statCircle_{name_tag2}.15100-20100.ref", 'rb') as f:
#     cons = pkl.load(f)

# for z in Internal.getZones(cons): 
#       theta1 = Internal.getNodeFromName(z, 'CoordinateY')[1]
#       Prms1 = Internal.getNodeFromName(z,'Prms')[1]


# with open(f"STAT/statAero_{name_tag1}.15100-20100.ref", 'rb') as f:
#     cons = pkl.load(f)

# for z in Internal.getZones(cons): 
#       if z[0] == "Cart.32X0": 
#             ind = 11
#             CoordX1 = Internal.getNodeFromName(z, 'CoordinateX')[1][:-1,ind]
#             Prmsline1 = Internal.getNodeFromName(z,'Prms')[1][:,ind]


# with open(f"STAT/statAero_{name_tag2}.15100-20100.ref", 'rb') as f:
#     cons = pkl.load(f)

# for z in Internal.getZones(cons): 
#       if z[0] == "Cart.32X0": 
#             ind = 11
#             CoordX2 = Internal.getNodeFromName(z, 'CoordinateX')[1][:-1,ind]
#             Prmsline2 = Internal.getNodeFromName(z,'Prms')[1][:,ind]

# # ### Rayon = 20
# # r20 = 20
# # with open("STAT/statCircle20_ausmp1_spg.15100-20100.ref", 'rb') as f:
# #     Enri1 = pkl.load(f)

# # for z in Internal.getZones(Enri1): 
# #       theta20 = Internal.getNodeFromName(z, 'CoordinateY')[1]
# #       Prms20 = Internal.getNodeFromName(z,'Prms')[1]


# ## Figure1
# fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
# ax.plot(theta0, Prms0, 'r-', label='InterpDataType=1')
# ax.plot(theta1, Prms1, 'b-', label='InterpDataType=0')

# # ax.plot(theta20, Prms20, 'b-', label='R = 20')#, markerfacecolor='none')

# ax.set_rmax(2.5)
# ax.set_rticks([0.5, 1.0, 1.5, 2.0, 2.5])  # Less radial ticks
# # #ax.set_rlabel_position(-22.5)  # Move radial labels away from plotted line
# # ax.legend(loc='upper left', bbox_to_anchor=(0.7, 0.3))
# ax.legend()
# ax.grid(True)

# ax.set_ylabel("$\mathbf{P_{rms}}$", rotation=0)
# ax.yaxis.set_label_coords(0.9,0.5)
# ax.tick_params(axis='y', labelsize=12)
# #ax.set_title("AUSMPRED (R=0.5)", va='bottom')
# # plt.savefig("Prms_enri.png")
# plt.savefig("PLOT/prms_InterpDataType_compare.pdf", format="pdf")
# #plt.show()

# print('=======================================================================')
# print("PRMS (InterpDataType, ordre, spg, r, max, min, mean):", 1, 2, 0, 10, np.max(Prms0), np.min(Prms0), np.mean(Prms0))
# print("PRMS (InterpDataType, ordre, spg, r, max, min, mean):", 0, 2, 0, 10, np.max(Prms1), np.min(Prms1), np.mean(Prms1))
# print('=======================================================================')
# # print("PRMS (r, max, min, mean):", r20, np.max(Prms20), np.min(Prms20), np.mean(Prms20))
# # print('=======================================================================')

# fig, ax = plt.subplots()
# ax.plot(CoordX1, Prmsline1, 'r-', label='InterpDataType=1')
# ax.plot(CoordX2, Prmsline2, 'b-', label='InterpDataType=0')

# ax.legend()
# ax.grid()
# ax.set_xlabel("X")
# ax.set_ylabel("$\mathbf{P_{rms}}$")

# plt.savefig("PLOT/prms_lineX_InterpDataType_compare.pdf", format="pdf")


# print("\n")
# print("-- Ploting terminate correclty --")
# print("\n")