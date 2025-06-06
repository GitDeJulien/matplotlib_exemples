
import Converter.PyTree as C
import Converter.Internal as Internal
import Post.PyTree as P

import numpy as np
import matplotlib.pyplot as plt

## fold all methods   : Ctrl + K and Ctrl + 0
## unfold all methods : Ctrl + K and Ctrl + J

plt.style.use('./figstyle.mplstyle')

#==============================================
## Parameters
#==============================================

# Physical constants
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


# Other params
loads_filenames = ['LOADS/loads_ordre2_ausmp1.0_spg0.txt', 
            'LOADS/loads_ordre2_ausmp1.0_spg0_Allcons.txt', 
            'LOADS/loads_ordre2_ausmp1.0_spg1_Allcons.txt', 
            'LOADS/loads_ordre2_seuseur1.0_spg0_Allcons.txt', 
            'LOADS/loads_ordre2_seuseur1.0_spg1_Allcons.txt']

it = 40100
wall_filenames = [f'STAT/wall_ordre2_ausmp1.0_spg0.{it}.cgns', 
            f'STAT/wall_ordre2_ausmp1.0_spg0_Allcons.{it}.cgns', 
            f'STAT/wall_ordre2_ausmp1.0_spg1_Allcons.{it}.cgns', 
            f'STAT/wall_ordre2_seuseur1.0_spg0_Allcons.{it}.cgns', 
            f'STAT/wall_ordre2_seuseur1.0_spg1_Allcons.{it}.cgns']

labels = ["A.O2.NC.S0",
        "A.O2.CO.S0", 
        "A.O2.CO.S1", 
        "S.O2.CO.S0", 
        "S.O2.CO.S1"]

#ON-OFF
PRINTINFO = False
PLOTLOADS = True
PLOTCP    = True
FIGSHOW   = False

#==============================================
#==============================================

datas_loads = []
datas_wall  = []
if PRINTINFO or PLOTLOADS:
    for i, filename in enumerate(loads_filenames):
        print(f"Reading {filename}", end="...")
        datas_loads.append(np.loadtxt(filename))
        print("done.")
if PLOTCP:
    for i, filename in enumerate(wall_filenames):
        t_wall = C.convertFile2PyTree(filename)
        datas_wall.append(P.isoSurfMC(t_wall, '{centers:CoordinateZ}', 
                                      value=0.05))
        
print("\n")
        

def printAeroLoadsInfo(datas, tmin_adim=1., each=1):

    """*Affiche des éléments d'info à partir des coefficients 
    adimensionnés de portance et de trainee (Strouhal, ...)*

    Args:
        data (_NDArray_) : *[iteration, trainee, portance]*
        tmin_adim (_float_) : *temps minimum à partir duquel le 
        calcul est convergé, default 1.0*
        each (_int_) : *enregistrement des coefficient tout les 
        'each' itérations*
    
    """
    print("[printAeroLoadsInfo]")
    for i, data in enumerate(datas):

        it_min = np.where(data[:,0]*dt*U0/L0 > tmin_adim)[0][0]
        fourier = np.fft.fft(data[it_min:,2]*coeff-
                             np.mean(data[it_min:,2]*coeff))
        n = data[it_min:,2].size
        freq = np.fft.fftfreq(n, d=each*dt) #addapter le pas de temps

        print('================================================')
        print('File label:', labels[i])
        print("it_min =", it_min)
        f_lift = freq[np.where(fourier.real==fourier.real.max())]
        St_real = f_lift*L0/U0
        print('Frequence Lift (real part) =',f_lift)
        print('Strouhal number (real part) = ',St_real)
        f_lift = freq[np.where(fourier.imag==fourier.imag.max())]
        St_img = f_lift*L0/U0
        print('Frequence Lift (img part) =',f_lift)
        print('Strouhal number (img part) = ',St_img)
        print('Mean Drag coefficient =', 
              np.mean(data[it_min:,1]*coeff))
        print('Mean Lift coefficient =', 
              np.mean(data[it_min:,2]*coeff))
        print('================================================\n')

    return St_real, St_img

def plotAeroLoads(datas, outname = "./CD-CL.png"):
    print("[plotAeroLoads]")
    fig, axs = plt.subplots(2,1, layout="constrained", figsize=(6, 8))
    axs[0].grid()
    axs[1].grid()
    # ax.set_xlim([100,200])
    # ax.set_ylim([1,1.45])
    print(f"Writting {outname}", end="...")
    for i, data in enumerate(datas):

        # print(f"Max iteration for {labels[i]} =", int(data[-1,0]))
        #Portance
        axs[0].plot(data[:,0]*dt*U0/L0, data[:,2]*coeff, 
                    label=labels[i], 
                    alpha=0.8)
        #Trainee
        axs[1].plot(data[:,0]*dt*U0/L0,data[:,1]*coeff, 
                    alpha=0.8)                  

    axs[0].set_xlabel('$\mathbf{t*=tU_{\\infty}/D}$')
    axs[0].set_ylabel('$\mathbf{C_L}$', rotation=0,labelpad=12,
                      va='center')

    axs[1].set_xlabel('$\mathbf{t*=tU_{\\infty}/D}$')
    axs[1].set_ylabel('$\mathbf{C_D}$', rotation=0,labelpad=12,
                      va='center')

    axs[0].legend()

    fig.savefig(outname ,bbox_inches='tight')
    print("done.")
    print("\n")

def plotCp(datas, outname="./Cp.png"):

    print("[plotCp]")
    print(f"Writting {outname}", end="...")
    fig, ax = plt.subplots(figsize=(5,5))
    ax.grid()

    for i,t_cp in enumerate(datas):

        for z in Internal.getZones(t_cp):
            gridcoord = Internal.getNodeFromName(z, 'GridCoordinates')
            xx = Internal.getNodeFromName(gridcoord, 'CoordinateX')[1]
            yy = Internal.getNodeFromName(gridcoord, 'CoordinateY')[1]
            theta = np.arctan2(yy, xx)*180/np.pi
            idx_sort = np.argsort(theta)
            # print(theta[idx_sort])
            # flowsol = Internal.getNodeFromName(z, 'FlowSolution#Centers')
            flowsol = Internal.getNodeFromName(z, 'FlowSolution')


            Cp = Internal.getNodeFromName(flowsol, 'Cp')[1]
        
        ax.plot(theta[idx_sort]+180, Cp[idx_sort], label=labels[i])

    ax.set_xlabel('$\mathbf{\\theta}$')
    ax.set_xlim([0,180])
    ax.set_ylabel('$\mathbf{C_p}$',rotation=0,labelpad=12,va='center')
    ax.legend()
    fig.savefig(outname ,bbox_inches='tight')
    print("done.")
    print("\n")

#==========================================
# Print informations (Strouhal)
#==========================================

if PRINTINFO: printAeroLoadsInfo(datas_loads, tmin_adim=80., each=50)

#==========================================
# Plot coefficient de pression à la paroie
#==========================================

if PLOTCP   : plotCp(datas_wall, "LOADS/CP_comparison.pdf")

#==========================================
# Plot coefficients de portance et trainee
#==========================================

if PLOTLOADS: plotAeroLoads(datas_loads, 
                            outname="LOADS/CD-CL_comparison.pdf")
    

if FIGSHOW : plt.show()
#==========================================
# Plot PSD
#==========================================
# probes1 = C.convertFile2PyTree('PROBES/probe_haut_'+ tag_name1 + '.cgns')
# probes2 = C.convertFile2PyTree('PROBES/probe_haut_'+ tag_name2 + '.cgns')
# probes3 = C.convertFile2PyTree('PROBES/probe_haut_'+ tag_name3 + '.cgns')

# # print(probes)
# # X = 0, Y = 50, Z = 0
# dp1 = []
# dp1 = np.array(dp1)
# dp2 = dp1
# dp3 = dp1
# for i in range(0,50):
#     sol1 = Internal.getNodeFromName(probes1, f'FlowSolution#{i}')
#     dp1 = np.concatenate((dp1,Internal.getNodeFromName(sol1, 'dp')[1]))
#     sol2 = Internal.getNodeFromName(probes2, f'FlowSolution#{i}')
#     dp2 = np.concatenate((dp2,Internal.getNodeFromName(sol2, 'dp')[1]))
#     sol3 = Internal.getNodeFromName(probes3, f'FlowSolution#{i}')
#     dp3 = np.concatenate((dp3,Internal.getNodeFromName(sol3, 'dp')[1]))

# time = np.arange(4.02, 8.02, 0.0002*4)
# # print(f"time[{len(time)}]:",time)
# # print(f"dp[{len(dp)}]:",dp)


# fig, ax = plt.subplots(layout="constrained", figsize=(5, 5))#figsize=(5,4))

# ax.grid()
# # ax.set_xlim([100,200])
# # ax.set_ylim([1,1.45])

# freq1, psd1 = welch(dp1, fs=1250, window='hann', nperseg=625, scaling='density')
# freq2, psd2 = welch(dp2, fs=1250, window='hann', nperseg=625, scaling='density')
# freq3, psd3 = welch(dp3, fs=1250, window='hann', nperseg=625, scaling='density')


# # PSD
# psd1 = 10*np.log(psd1/0.00002)
# psd2 = 10*np.log(psd2/0.00002)
# psd3 = 10*np.log(psd3/0.00002)

# ax.plot(freq1/U0, psd1, label=labels[0], alpha=0.8)
# ax.plot(freq2/U0, psd2, label=labels[1], alpha=0.8)
# ax.plot(freq3/U0, psd3, label=labels[2], alpha=0.8)

# ax.set_ylim([-10,100])
# ax.set_xlim([0.03,7])

# ax.set_xscale('log')

# ax.set_xlabel('$\mathbf{St}$')
# ax.set_ylabel('$\mathbf{PSD~[dB/Hz]}$',labelpad=12,va='center')

# ax.legend()
# fig.savefig('LOADS/PSD_X0Y50.pdf',bbox_inches='tight')


# ==========================================
# Chargement du fichier wall
# ==========================================


# filename1 = 'STAT/wall_ordre2_ausmp1.0_spg0.20100.cgns'
# filename2 = 'STAT/wall_ordre2_ausmp1.0_spg0_Allcons.20100.cgns'
# filename3 = 'STAT/wall_ordre2_ausmp1.0_spg1_Allcons.20100.cgns'
# filename4 = 'STAT/wall_ordre2_seuseur1.0_spg1_Allcons.20100.cgns'

# labels = ["A.O2.NC.S0", "A.O2.CO.S0", "A.O2.CO.S1", "S.O2.CO.S1"]
# t_cp_list = []

# t_wall1 = C.convertFile2PyTree(filename1)
# t_cp_list.append(P.isoSurfMC(t_wall1, '{centers:CoordinateZ}', value=0.05))

# t_wall2 = C.convertFile2PyTree(filename2)
# t_cp_list.append(P.isoSurfMC(t_wall2, '{centers:CoordinateZ}', value=0.05))

# t_wall3 = C.convertFile2PyTree(filename3)
# t_cp_list.append(P.isoSurfMC(t_wall3, '{centers:CoordinateZ}', value=0.05))

# t_wall4 = C.convertFile2PyTree(filename4)
# t_cp_list.append(P.isoSurfMC(t_wall4, '{centers:CoordinateZ}', value=0.05))


# fig, ax = plt.subplots(figsize=(5,5))
# ax.grid()

# # t_cp = C.convertFile2PyTree(filename)
# for t_cp, nm in zip(t_cp_list, labels):

#     for z in Internal.getZones(t_cp):
#         gridcoord = Internal.getNodeFromName(z, 'GridCoordinates')
#         xx = Internal.getNodeFromName(gridcoord, 'CoordinateX')[1]
#         yy = Internal.getNodeFromName(gridcoord, 'CoordinateY')[1]
#         theta = np.arctan2(yy, xx)*180/np.pi
#         idx_sort = np.argsort(theta)
#         # print(theta[idx_sort])
#         # flowsol = Internal.getNodeFromName(z, 'FlowSolution#Centers')
#         flowsol = Internal.getNodeFromName(z, 'FlowSolution')


#         Cp = Internal.getNodeFromName(flowsol, 'Cp')[1]
    
#     ax.plot(theta[idx_sort]+180, Cp[idx_sort], label=nm)

# ax.set_xlabel('$\mathbf{\\theta}$')
# ax.set_xlim([0,180])
# ax.set_ylabel('$\mathbf{C_p}$',rotation=0,labelpad=12,va='center')
# ax.legend()

# # cp_mean = np.load('cp_mean.npy')
# # cp_ref = np.loadtxt('cp_refInoue.dat')

# # cp_mean = []
# # theta_new = []
# # for i in range (0, len(idx_sort), 1):
# #     # mean= np.mean(Cp[idx_sort[i:i+8]])
# #     mean = Cp[idx_sort[i]]
# #     cp_mean.append(mean)
# #     theta_new.append(theta[idx_sort[i]])

# # cp_mean = np.array(cp_mean)
# # theta_new = np.array(theta_new)


# # ax.plot(theta_new, Cp[theta_idx])

# # ax.plot(theta_new, cp_mean)
# # ax.plot(theta[idx_sort]+180,(1+0.2)*cp_mean[idx_sort])
# # ax.plot(cp_ref[:,0],cp_ref[:,1],'o',markevery=3)


# fig.savefig('LOADS/Cp_compare.pdf',bbox_inches='tight')


# plt.show()
