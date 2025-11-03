#%% Setup

import numpy as np
import matplotlib.pyplot as plt

def analyseADS(adsp, adsh, snum):
    
    nmax = adsh[0,2]
    nstk = adsh[0,0]

    nshots = []
    nhits = []
    npoints = []
    nmm = []
    for i in range(1, nmax+1):
        pti = adsp[adsp[:,6]==i]
        
        nshots.append(snum[i-1]*nstk)
        nhits.append(snum[i-1]*nstk*i)
        npoints.append(len(pti))
        
        ptmm = pti[pti[:,1] != pti[:,2]]
        nmmi = len(ptmm)
        nmm.append(nmmi)

    nhits = np.array(nhits)
    npoints = np.array(npoints)
    nshots = np.array(nshots)
    nmm = np.array(nmm)

    nu = nhits - npoints # Unmatched
    ng = npoints - nmm # Correctly matched
    pmm = nmm/npoints # Mismatchs/Matches
    #pss = ng/nhits # Correct matches/Hits
    pse = ng/nshots # Correct matches/Shots
    pnu = nu/nshots # Unmatched/Shots
    pnm = npoints/nshots # Matches/Shot
    fm = pnm*pmm # Mismatch/Shot
    
    return [nmax, pmm, pse, pnu, pnm, fm]

def analyseEXP(csc, pmm, nctrmax, g):
    hps = csc[:,2:5]
        
    shoteffall = np.zeros(nctrmax)
    shoteffcorrect = np.zeros(nctrmax)
    shotcount = np.zeros(nctrmax)
    nomatch = np.zeros(nctrmax)
    unequal = np.zeros(nctrmax)
    totalunmatched = np.zeros(nctrmax)
    for nctr in range(1, nctrmax+1):
        
        hpsnC = hps[hps[:,0]==nctr]
        hpsnS = hps[hps[:,1]==nctr]
        hpsnE = hps[hps[:,0]==nctr]
        hpsnE = hpsnE[hpsnE[:,1]==nctr]
        groups = [hpsnC, hpsnS, hpsnE]
        
        hpsn = groups[g]
        nshots = len(hpsn)
        
        shotcount[nctr-1] = nshots   
        allmatches = np.sum(hpsn[:,2])
        
        shoteffall[nctr-1] = allmatches/nshots
        shoteffcorrect[nctr-1] = (allmatches * (1 - pmm[nctr-1])) / nshots
        
        if g==0: #only for centroid grouping
            d = -1*(hpsn[:,1] - nctr)
            d[d<0] = 0
            nm = np.min(hpsn[:,:2], 1)
            
            unequal[nctr-1] = np.sum(d)/nshots
            nomatch[nctr-1] = (np.sum(nm) - allmatches)/nshots
        
        totalunmatched[nctr-1] = (nshots*nctr - allmatches)/nshots
    
    return totalunmatched, shoteffall, shoteffcorrect

#%% Load and Process

names = ["xenon_30ns", "xenon_9ns", "CH3I_7ns", "NO_0L_7ns"]
labels = ["Xe [TTS = 31 ns]", "Xe [TTS = 8.8 ns]",
          "CH\u2083I [TTS = 6.9 ns]", "NO [TTS = 6.2 ns]"]
nctrmax = [15, 7, 15, 11]
cols = ['red', 'blue', 'green', 'purple']
markers = ['o', '^', 's', 'D']
path = "../data/MHE/"

snum = list(np.ones(20, dtype=int)*200)
nstk = 20
nsam = 4000
g = 0

adspts = []
adshps = []
csc = []

for i in range(0, len(names)):
    adspts.append(np.load(path + names[i] + "_ADS.pt.npy"))
    adshps.append(np.load(path + names[i] + "_ADS.hps.npy"))
    csc.append(np.load(path + names[i] + ".csc.npy"))

nmax = []
pmm = []
pse = []
pnu = []
pnm = []
fm = []
for i in range(0, len(names)):
    adsp = adspts[i]
    adsh = adshps[i]    
    adsa = analyseADS(adsp, adsh, snum)
    nmax.append(adsa[0])
    pmm.append(adsa[1])
    pse.append(adsa[2])
    pnu.append(adsa[3])
    pnm.append(adsa[4])
    fm.append(adsa[5])

totalunmatched = []
shoteffall = []
shoteffcorrect = []
for i in range(0, len(names)):
    exp = analyseEXP(csc[i], pmm[i], nctrmax[i], g)
    totalunmatched.append(exp[0])
    shoteffall.append(exp[1])
    shoteffcorrect.append(exp[2])

#%% ADS plot

FSt = 20
FSlb = 18
FSlg = 15
FStk = 15
col1 = 'black'
col2 = 'white'
pos1 = 0.02
pos2 = 0.96
fslett = 20
lett = ['a', 'b', 'c', 'd']
mksize = 8

fig = plt.figure()
fig.set_tight_layout(True)
ax1 = plt.subplot2grid((2, 2), (0, 0))
ax2 = plt.subplot2grid((2, 2), (0, 1))
ax3 = plt.subplot2grid((2, 2), (1, 0))
ax4 = plt.subplot2grid((2, 2), (1, 1))
ax = [ax1, ax2, ax3, ax4]

for i in range(0, len(ax)):
    
    xax = np.arange(1, nmax[i]+1)
    
    #ax[i].plot(xax, pnu[i], marker='s', color='blue', label = 'Unmatched', lw=2, markersize=mksize)
    #ax[i].plot(xax, pnm[i], label="Total Matches", marker='^', color='pink', lw=2, markersize=mksize)
    ax[i].plot(xax, pse[i], marker='o', color='green', label="Correctly Assigned Pairs", lw=2, markersize=mksize)
    ax[i].plot(xax, fm[i], marker='x', color='red', label = 'Incorrectly Assigned Pairs', lw=2, markersize=mksize)
    #ax[i].plot(xax, pse[i] - fm[i], marker='d', color='purple', label = 'Correct - False', lw=2, markersize=mksize)
    ax[i].plot([0, nmax[i]+1], [0, nmax[i]+1], ls='--', color='black', lw=4)
    
    ax[i].set_title("ADS: " + labels[i], fontsize=FSt)
    ax[i].set_xlabel("Hits Per Shot", fontsize=FSlb)
    ax[i].set_ylabel("Number of Hits", fontsize=FSlb)
    ax[i].set_ylim([0, nmax[i]+0.5])
    ax[i].set_xlim([0, nmax[i]+0.5])
    ax[i].set_xticks(xax)
    ax[i].set_yticks(xax[::2])
    
    #ax[i].minorticks_on()
    ax[i].tick_params(labelsize=FStk, which='both', width=2, length=4)
    ax[i].legend(loc=9, fontsize=FSlg)
    ax[i].grid()
    #ax[i].set_aspect(1./ax[i].get_data_ratio())

    txt = ax[i].text(pos1, pos2, '({})'.format(lett[i]), ha='left', va='top',
                     color=col1, transform=ax[i].transAxes, fontsize=fslett)
    txt.set_bbox(dict(facecolor=col2, alpha=0.8, edgecolor=col2))

figman = plt.get_current_fig_manager()
figman.window.showMaximized()

#%% EXP Plots

FSt = 20
FSlb = 18
FSlg = 15
FStk = 15
col1 = 'black'
col2 = 'white'
pos1 = 0.02
pos2 = 0.96
fslett = 20
lett = ['a', 'b', 'c', 'd']
mksize = 8

fig = plt.figure()
fig.set_tight_layout(True)
ax1 = plt.subplot2grid((2, 2), (0, 0))
ax2 = plt.subplot2grid((2, 2), (0, 1))
ax3 = plt.subplot2grid((2, 2), (1, 0))
ax4 = plt.subplot2grid((2, 2), (1, 1))
ax = [ax1, ax2, ax3, ax4]

for i in range(0, len(ax)):
    
    xax = np.arange(1, nctrmax[i]+1)
    sefalse = shoteffall[i] - shoteffcorrect[i]
    
    #ax[i].plot(xax, totalunmatched[i], marker='s', color='blue', label = 'Unmatched', lw=2, markersize=mksize)
    ax[i].plot(xax, shoteffall[i], marker='^', color='blue', label="Total Matches", lw=2, markersize=mksize)
    ax[i].plot(xax, shoteffcorrect[i], marker='o', color='green', label="Correctly Assigned Pairs", lw=2, markersize=mksize)
    ax[i].plot(xax, sefalse, marker='x', color='red', label = 'Incorrectly Assigned Pairs', lw=2, markersize=mksize)
    #ax[i].plot(xax, shoteffcorrect[i] - sefalse[i], marker='d', color='purple', label = 'Correct - False', lw=2, markersize=mksize)
    ax[i].plot([0, nctrmax[i]+1], [0, nctrmax[i]+1], ls='--', color='black', lw=4)
    
    ax[i].set_title(labels[i], fontsize=FSt)
    ax[i].set_xlabel("Hits Per Shot", fontsize=FSlb)
    ax[i].set_ylabel("Number of Hits", fontsize=FSlb)
    ax[i].set_ylim([0, nctrmax[i]+0.5])
    ax[i].set_xlim([0, nctrmax[i]+0.5])
    ax[i].set_xticks(xax)
    ax[i].set_yticks(xax[::2])
    
    #ax[i].minorticks_on()
    ax[i].tick_params(labelsize=FStk, which='both', width=2, length=4)
    ax[i].legend(loc=9, fontsize=FSlg)
    ax[i].grid()
    #ax[i].set_aspect(1./ax[i].get_data_ratio())

    txt = ax[i].text(pos1, pos2, '({})'.format(lett[i]), ha='left', va='top',
                     color=col1, transform=ax[i].transAxes, fontsize=fslett)
    txt.set_bbox(dict(facecolor=col2, alpha=0.8, edgecolor=col2))

figman = plt.get_current_fig_manager()
figman.window.showMaximized()

#%% Combined Plot

FSlb = 18
FSlg = 15
FStk = 15
mksize = 8

fig, ax = plt.subplots(figsize=(9,9))
fig.set_tight_layout(True)

nm = max(nctrmax)
xax = np.arange(1, nm+1)

markers = ['o', 's', '^', 'x']
for i in range(0, len(names)):
    ax.plot(xax[:nctrmax[i]], shoteffcorrect[i], label=labels[i], marker=markers[i], color=cols[i], lw=3, markersize=mksize)
ax.plot([0, nm+1], [0, nm+1], ls='--', lw=4, color='black')

ax.set_xlabel("Hits Per Shot", fontsize=FSlb)
ax.set_ylabel("Correctly Assigned Hits", fontsize=FSlb)
ax.set_ylim([0, 8.5])
ax.set_xlim([0, nctrmax[i]+0.5])
ax.set_xticks(xax)
ax.set_yticks(xax[:8])

ax.tick_params(labelsize=FStk, which='both', width=2, length=4)
ax.legend(loc=2, fontsize=FSlg)
ax.grid()
ax.set_aspect(1./ax.get_data_ratio())

