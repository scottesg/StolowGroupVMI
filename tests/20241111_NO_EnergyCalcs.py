#%% Setup

import numpy as np
import matplotlib.pyplot as plt

h = 4.135667516e-15 # eV*s
#hbar = h/(2*np.pi)
c = 299792458 # m/s

E200 = h*c/(200*1e-9)
E267 = h*c/(267*1e-9)

def zpe(we, wexe):
    return (1/2)*we - (1/4)*wexe

# Ev+1 - Ev
def deltaEvib(v, we, wexe):
    return we - 2*(v+1)*wexe

def wn2ev(wn):
    return 100*c*h*wn

def vibProg(beamEnergy, IP, we, wexe, emax, vmax):
    if beamEnergy <= IP: return None
    
    prog = []
    if beamEnergy-IP < emax:
        prog = [[beamEnergy-IP, 0]]
    
    v = 1
    Dv = wn2ev(deltaEvib(v-1, we, wexe))
    stateE = IP + Dv
    while (stateE < beamEnergy) and (v <= vmax):
        if beamEnergy-stateE < emax:
            prog.append([beamEnergy-stateE, v])
        
        v += 1
        Dv += wn2ev(deltaEvib(v-1, we, wexe))
        stateE = IP + Dv
        
    return prog
        
#%% NO Constants

NOIP = 9.2642 # eV

emax = 1.00 # eV
vmax = 5

# NO vibrational constants, cm-1

# Ground state
NOX_we = 1904.20 # hbar*we
NOX_wexe = 14.075 # hbar*wexe

# A state
NOA_Te = 43965.7
NOA_we = 2374.31 # hbar*we
NOA_wexe = 10.106 # hbar*wexe

# Ion ground state
NOp_we = 2376.42 # hbar*we
NOp_wexe = 16.262 # hbar*wexe

#%% Ground state progressions

NOX_2x200 = np.array(vibProg(2*E200, NOIP, NOp_we, NOp_wexe, emax, vmax))
NOX_2x267 = np.array(vibProg(2*E267, NOIP, NOp_we, NOp_wexe, emax, vmax))
NOX_ppr = np.array(vibProg(E200+E267, NOIP, NOp_we, NOp_wexe, emax, vmax))

#%% NO(A) IP

# Zero point energy
NOX_zpe = zpe(NOX_we, NOX_wexe)
NOA_zpe = zpe(NOA_we, NOA_wexe)

# NO(A) IP, eV
NO_XA_00 = wn2ev((NOA_Te + NOA_zpe) - NOX_zpe)
NOAIP = NOIP-NO_XA_00

#%% NO(A) progressions

NOA_200 = np.array(vibProg(E200, NOAIP, NOp_we, NOp_wexe, emax, vmax))
NOA_267 = np.array(vibProg(E267, NOAIP, NOp_we, NOp_wexe, emax, vmax))

#%% Plotting

progs = [NOX_2x200, NOX_2x267, NOX_ppr, NOA_200, NOA_267]
labels = ['X, 2x200nm', 'X, 2x267nm', 'X, 200+267nm', 'A, 200nm', 'A, 267nm']

fig, [ax1, ax2] = plt.subplots(1,2)
markers= ['x', 's', '+', 'd', '^']
lss = ['-', ':', '--', '-.', (0, (3, 1, 1, 1))]
msize = 8

tks = [0.766]
for i in range(0, len(progs)):
    prog = progs[i]
    if len(prog)==0: continue
    
    tks.append(prog[:,0])
    ax1.plot(prog[:,1], prog[:,0], marker=markers[i], color='C{}'.format(i+1), lw=0, markersize=msize, label=labels[i])
    ax2.hlines(prog[:,0], 0, 1, color='C{}'.format(i+1), ls=lss[i], label=labels[i])

ax1.hlines(0.766, 0, 5, lw=3, ls='--', color='black', label='CS2 Pump-Probe Signal')
ax2.hlines(0.766, 0, 1, lw=3, ls='--', color='black', label='CS2 Pump-Probe Signal')

ax1.grid()

fig.suptitle("Available NO Photoionization Processes", fontsize=20)
ax1.set_ylabel("Energy (eV)", fontsize=15)
ax1.set_xlabel("Vibrational State", fontsize=15)

ax1.tick_params(labelsize=12)

ax2.tick_params(labelsize=12)
ax2.set_ylabel("Energy (eV)", fontsize=15)
ax2.set_yticks(np.hstack(tks))

ax2.axes.get_xaxis().set_visible(False)

leg = ax1.legend(fontsize=12)
leg.set_draggable(1)

ax1.set_ylim(0, 1.1)
ax2.set_ylim(0, 1.1)

leg = ax2.legend(fontsize=12)
leg.set_draggable(1)