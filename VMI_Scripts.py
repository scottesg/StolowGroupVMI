# Imports

import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d # This IS being used, even though Spyder doesn't think so
from scipy.interpolate import interp1d
from VMI_Functions import (vmigetstats, vmibgsubtr, vmicropdifstk,
                          vmirotate, vmiiabel_stk, vmiplotaverages,
                          vmiplottraces, vmical)

#%%

plt.close('all')

#%%

path = "20241010/20241010_cs2test3"
inpath = r"C:\Users\Scott\Python\VMI\data/" + path
outpath = r"C:\Users\Scott\Python\VMI\reduceddata/" + path

# [-1] for 'all scans', or [scan, [start, stop], [start, stop]] ex [0, [3, 50], [60, 100]] Stop not inclusive
scans = [-1]
#scans = [15, [20,22]]

if not os.path.isdir(inpath):
    print("Folder {} not found".format(inpath))
    raise SystemExit;

if not os.path.exists(outpath):
    os.makedirs(outpath)

vmistats = vmigetstats(inpath, scans, 0);

vmiplottraces(vmistats)
vmiplotaverages(vmistats)

#%%

peb = '001'

# peb = '000' - no bg subtraction
#       '001' - subtract 'nobeam'
#       '100' - subtract 'pumponly'
#       '101' - subtract 'pumponly' and 'nobeam'
#       '010' - subtract 'probeonly'
#       '011' - subtract 'probeonly' and 'nobeam'
#       '111' - subtract 'pumponly', 'probeonly' and 'nobeam'

#vmistats['vmicentre'] = [234, 355]

vmistats = vmibgsubtr(vmistats, peb)
vmistats = vmicropdifstk(vmistats)

#%%

vmistats = vmirotate(vmistats, 62.5, lockangle=True)
#vmistats = vmirotate(vmistats, 70)

#%%

vmistats = vmiiabel_stk(vmistats, 3000, 1, 0.2, 0)

#%% Test Transformation

Ir = vmistats['Ir']
delays = vmistats['delays']
r = vmistats['r']

imraw = vmistats['imstks'][7]
imin = vmistats['imstks']['iminstk']
iminpol = vmistats['imstks']['iminpol']
imoutpol = vmistats['imstks']['imoutpol']
imout = vmistats['imstks']['idifstk']

fig, axs = plt.subplots(2,2)

i = 20

axs[0,0].imshow(imin[i])
axs[0,1].imshow(iminpol[i])
axs[1,0].imshow(imoutpol[i])
axs[1,1].imshow(imout[i])

plt.figure()
plt.imshow(imraw[i])

plt.figure()
plt.plot(r, Ir[i])

#%% plot TR-PES (r[pix],t[ps])

plot3d = False

Ir = vmistats['Ir']
delays = vmistats['delays']
r = vmistats['r']

upperlimit_percent_2d = 100

fig = plt.figure()
if len(delays)>1:
    if plot3d:
        ax = plt.axes(projection="3d")
        X, Y = np.meshgrid(r, delays);
        ax.plot_surface(X, Y, Ir, cmap='nipy_spectral')
        ax.set_zlabel("Intensity")
    else:
        ax = plt.axes()
        X, Y = np.meshgrid(r, delays);
        ax.pcolormesh(X, Y, Ir, cmap='nipy_spectral', vmax=np.percentile(Ir,upperlimit_percent_2d))
    ax.set_ylabel("delays, [ps]")
else:
    plt.plot(r, Ir)
ax.set_xlabel("ring radius, [pix]")
ax.set_title("TRPES")

fout = outpath + "/" + vmistats['subfolder'] + ".trpes.r[pix].csv"
out = np.vstack((np.hstack((2, r)), np.vstack((delays, Ir.T)).T))
np.savetxt(fout, out, fmt='%.2f', delimiter=",")

#%% TR-PES (r[pix],t[ps]): plot time evolutions of few energy-slices 

rbinsa = [[1, 30], [31, 100], [150, 200]] #pixels
r = vmistats['r']
delays = vmistats['delays']
Ir = vmistats['Ir']

irbinsa = [np.searchsorted(r, n) for n in rbinsa]

Ir_rbinsa = np.zeros((Ir.shape[0],len(rbinsa)))
labels = []
for i in range(0, len(rbinsa)):
    Ir_rbinsa[:,i] = np.sum(Ir[:,irbinsa[i][0]:irbinsa[i][1]],1)/(irbinsa[i][1]-irbinsa[i][0]+1)  
    labels.append("r[pix]= {}...{}".format(rbinsa[i][0], rbinsa[i][1]))

fig, ax = plt.subplots()
ax.set(xlabel = "delays, [ps]", title = "Evolution of trpes slices with p-p delay")
    
for i in range(0, len(rbinsa)):
        ax.plot(delays, Ir_rbinsa[:,i], label=labels[i])
            
leg = ax.legend()
leg.set_draggable(1)

#%% TR-PES (r[pix],t[ps]): plot PES at few pp-delays 

dbinsa = [[-0.1, 0.1], [-0.5, -0.2], [-2.0, -0.8]] #ps
r = vmistats['r']
delays = vmistats['delays']
Ir = vmistats['Ir']

idbinsa = [len(delays)-np.searchsorted(delays[::-1], n)[::-1] for n in dbinsa]

Ir_dbinsa = np.zeros((len(dbinsa),Ir.shape[1]))
labels = []
for i in range(0, len(dbinsa)):
    Ir_dbinsa[i,:] = np.sum(Ir[idbinsa[i][0]:idbinsa[i][1],:],0)/(idbinsa[i][1]-idbinsa[i][0]+1)  
    labels.append("[{}...{}]ps".format(dbinsa[i][0], dbinsa[i][1]))

fig, ax = plt.subplots()
ax.set(xlabel = "r, [pix]", title = "Evolution of trpes slices with p-p delay")
    
for i in range(0, len(dbinsa)):
        ax.plot(r, Ir_dbinsa[i,:], label=labels[i])
            
leg = ax.legend()
leg.set_draggable(1)

#%% plot TR-PES (ke[eV],t[ps])

#calfile = '20190612_01_xenon_cal'
calfile = '03_xenon.vmical_ml.txt'
plot3d = False
upperlimit_percent_2d = 100

Ir = vmistats['Ir']
r = vmistats['r']
delays = vmistats['delays']
delaysN = len(delays)
#K = vmical(calfile)

#nm160 = 160.9
#xeip = 12.1299
cs2ip = 10.073
h = 4.135667516e-15 # eV*s
c = 299792458 # m/s
E200 = h*c/(200*1e-9)
E267 = h*c/(267*1e-9)
posE = 2*E200-cs2ip
radius = 170

K = posE / radius**2 #0.002264

KEs = np.arange(0, 3.0, 0.005)
bins = (KEs/K)**(0.5)

Ir_ke = np.zeros((Ir.shape[0], len(bins)))
    
for i in range(0, Ir.shape[0]):
    f = interp1d(r, Ir[i], fill_value='extrapolate')
    Ir_ke[i] = f(bins)

fig = plt.figure(1)
#fig.canvas.set_window_title("TRPES: {}".format(vmistats['subfolder']))

if delaysN>1:
    if plot3d:
        ax = plt.axes(projection="3d")
        X, Y = np.meshgrid(KEs, delays);
        ax.plot_surface(X, Y, Ir_ke, cmap='nipy_spectral')
        ax.set_zlabel("Intensity")
    else:
        ax = plt.axes()
        X, Y = np.meshgrid(KEs, delays);
        ax.pcolormesh(X, Y, Ir_ke, cmap='nipy_spectral', vmax=np.percentile(Ir_ke,upperlimit_percent_2d))
    ax.set_ylabel("delays, [ps]")
else:
    plt.plot(KEs, Ir_ke)
ax.set_xlabel("eKE, [eV]")
ax.set_title("TRPES")

vmistats['Ir_ke'] = Ir_ke
vmistats['ke'] = KEs
vmistats['K'] = K

fout = outpath + "/" + vmistats['subfolder'] + ".trpes.ke.csv"
out = np.vstack((np.hstack((0, KEs)), np.vstack((delays, Ir_ke.T)).T))
np.savetxt(fout, out, fmt='%.2f', delimiter=",")

#%% TR-PES (ke[eV],t[ps]): plot time evolutions of few energy-slices 

kebinsa = [[0.02, 0.2], [0.3, 0.5], [0.5, 0.8], [0.02, 1.0]]

KEs = vmistats['ke']
delays = vmistats['delays']
Ir_ke = vmistats['Ir_ke']
K = vmistats['K']

def ke2r(ke, K): return np.round((ke/K)**(0.5))
def r2ke(r, K): return K*r**2

irbinsa = [np.searchsorted(r, ke2r(np.array(n), K)) for n in kebinsa]

Ir_rbinsa = np.zeros((Ir.shape[0],len(kebinsa)))
labels = []
for i in range(0, len(kebinsa)):
    Ir_rbinsa[:,i] = np.sum(Ir[:,irbinsa[i][0]:irbinsa[i][1]],1)  
    labels.append("ke[eV]= {}...{}".format(kebinsa[i][0], kebinsa[i][1]))

fig, ax = plt.subplots()
ax.set(xlabel = "delays, [ps]", title = "Evolution of trpes slices with p-p delay")
    
Ir_rbinsa = Ir_rbinsa / (np.max(Ir_rbinsa) - np.min(Ir_rbinsa))

for i in range(0, len(kebinsa)):
        ax.plot(delays, Ir_rbinsa[:,i], label=labels[i])
            
leg = ax.legend()
leg.set_draggable(1)

#%% TR-PES (ke[eV],t[ps]): plot PES at few pp-delays 

dbinsa = [[-0.1, 0.1], [-0.5, -0.2], [-2.0, -0.8]] #ps
KEs = vmistats['ke']
delays = vmistats['delays']
Ir_ke = vmistats['Ir_ke']

idbinsa = [len(delays)-np.searchsorted(delays[::-1], n)[::-1] for n in dbinsa]

Ir_Ke_dbinsa = np.zeros((len(dbinsa),Ir_ke.shape[1]))
labels = []
for i in range(0, len(dbinsa)):
    Ir_Ke_dbinsa[i,:] = np.sum(Ir_ke[idbinsa[i][0]:idbinsa[i][1],:],0)/(idbinsa[i][1]-idbinsa[i][0]+1)  
    labels.append("[{}...{}]ps".format(dbinsa[i][0], dbinsa[i][1]))

fig, ax = plt.subplots()
ax.set(xlabel = "ke, [eV]", title = "Evolution of trpes slices with p-p delay")
    
for i in range(0, len(dbinsa)):
        ax.plot(KEs, Ir_Ke_dbinsa[i,:], label=labels[i])
            
leg = ax.legend()
leg.set_draggable(1)
