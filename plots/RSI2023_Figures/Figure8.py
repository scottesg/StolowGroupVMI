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
        print("{}: {}".format(nctr, nshots))
        
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

def ADSplots(pmm, pnu, pnm, pse, fm, name, nmax):
    
    x = np.arange(1, nmax+1)

    fig, ax = plt.subplots()
    ax.plot(x, pmm*100, marker='o', lw=4, markersize=10)
    ax.set_title("Mismatch Percentage: {}".format(name), fontsize=30)
    ax.set_xlabel("Hit Count", fontsize=30)
    ax.set_xlim([0, nmax+0.5])
    ax.set_xticks(x)
    ax.tick_params(labelsize=25, width=2, length=5)

    fig, ax = plt.subplots()
    ax.plot(x, pnu, marker='s', color='blue', label = 'Unmatched', lw=4, markersize=10)
    ax.plot(x, pnm, label="Total Matches", marker='^', color='red', lw=4, markersize=10)
    ax.plot(x, pse, label="Correct Matches", marker='o', color='black', lw=4, markersize=10)
    ax.plot(x, fm, marker='x', color='green', label = 'False Matches', lw=4, markersize=10)
    ax.plot(x, pse - fm, marker='d', color='purple', label = 'Correct - False', lw=4, markersize=10)
    ax.plot([0, nmax+1], [0, nmax+1], ls='--', lw=4)
    ax.set_xlabel("Position Hits Per Shot", fontsize=30)
    ax.set_ylabel("Position Hits", fontsize=30)
    ax.set_ylim([0, nmax+0.5])
    ax.set_xlim([0, nmax+0.5])
    ax.set_xticks(x)
    ax.set_yticks(x)
    ax.set_title(name, fontsize=30)
    ax.tick_params(labelsize=25, width=2, length=5)
    leg = ax.legend(fontsize=25)
    leg.set_draggable(1)
    plt.grid(axis='y')
    ax.set_aspect(1./ax.get_data_ratio())
    
def EXPplots(totalunmatched, shoteffall, shoteffcorrect, nctrmax):
    
    x = np.arange(1, nctrmax+1)

    fig, ax1 = plt.subplots()
    ax1.plot(x, totalunmatched, marker='s', color='blue', label = 'Unmatched', lw=4, markersize=10)
    sefalse = shoteffall - shoteffcorrect
    ax1.plot(x, shoteffall, label="Total Matches", marker='^', color='red', lw=4, markersize=10)
    ax1.plot(x, shoteffcorrect, label="Correct Matches", marker='o', color='black', lw=4, markersize=10)
    ax1.plot(x, sefalse, marker='x', color='green', label = 'False Matches', lw=4, markersize=10)
    ax1.plot([0, nctrmax+1], [0, nctrmax+1], ls='--', lw=4, color='black')
    ax1.set_xlabel("Hits Per Shot", fontsize=30)
    ax1.set_ylabel("Correlated Hits", fontsize=30)
    ax1.set_ylim([0, nctrmax+0.5])
    ax1.set_xlim([0, nctrmax+0.5])
    ax1.set_xticks(x)
    ax1.set_yticks(x)
    ax1.tick_params(labelsize=25, width=2, length=5)
    leg = ax1.legend(fontsize=25)
    leg.set_draggable(1)
    plt.grid(axis='y')
    ax1.set_aspect(1./ax1.get_data_ratio())

#%% Params

name = "NO_90H"
path = r"C:\Users\Scott\Python\VMI\data\20221003_VMI3DPaperData/MHE/OtherRuns/"

snum = list(np.ones(20, dtype=int)*200)
nstk = 20
nsam = 4000

adspts = np.load(path + name + "_ADS.pt.npy")
adshps = np.load(path + name + "_ADS.hps.npy")
csc = np.load(path + name + ".csc.npy")

#%% Analyse ADS

nmax, pmm, pse, pnu, pnm, fm = analyseADS(adspts, adshps, snum)

#%% ADS Plots

ADSplots(pmm, pnu, pnm, pse, fm, name, nmax)

#%% Experimental Data Multihit Stats

g = 0
nctrmax = 20
totalunmatched, shoteffall, shoteffcorrect = analyseEXP(csc, pmm, nctrmax, g)

#%% EXP Plots

EXPplots(totalunmatched, shoteffall, shoteffcorrect, nctrmax)

#%% Combined efficiency plot

names = ["xenon_30ns", "xenon_9ns", "CH3I_7ns", "NO_0L_7ns"]
labels = ["Xe [\u0394t = 31 ns]", "Xe [\u0394t = 8.8 ns]",
          "CH\u2083I [\u0394t = 6.9 ns]", "NO [\u0394t = 6.2 ns]"]
nctrmax = [15, 7, 15, 11]
cols = ['red', 'blue', 'green', 'purple']
markers = ['o', '^', 's', 'D']
path = r"C:\Users\Scott\Python\VMI\data\20221003_VMI3DPaperData/MHE/"

snum = list(np.ones(20, dtype=int)*200)
nstk = 20
nsam = 4000

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

#%% ADS plots

for i in range(0, len(names)):
    ADSplots(pmm[i], pnu[i], pnm[i], pse[i], fm[i], names[i], nmax[i])

#%% Analyse EXP

g = 0

totalunmatched = []
shoteffall = []
shoteffcorrect = []
for i in range(0, len(names)):
    print(names[i])
    exp = analyseEXP(csc[i], pmm[i], nctrmax[i], g)
    totalunmatched.append(exp[0])
    shoteffall.append(exp[1])
    shoteffcorrect.append(exp[2])
    
#%% Combined plot

nm = max(nctrmax)
x = np.arange(1, nm+1)

fig, ax = plt.subplots()
for i in range(0, len(names)):
    sefalse = shoteffall[i] - shoteffcorrect[i]
    contrast = 2*shoteffcorrect[i] - shoteffall[i]
    ax.plot(x[:nctrmax[i]], shoteffcorrect[i], label=labels[i], marker=markers[i], color=cols[i], lw=4, markersize=10)
    #ax.plot(x[:nctrmax[i]], shoteffall[i], label=labels[i], marker='x', color=cols[i], lw=4, markersize=10)
ax.plot([0, nm+1], [0, nm+1], ls='--', lw=4, color='black')
ax.set_xlabel("Hits Per Shot", fontsize=30)
ax.set_ylabel("Total Assigned Hits", fontsize=30)
ax.set_xticks(x[::2])
ax.set_yticks(x)
ax.tick_params(labelsize=25, width=2, length=5)
ax.set_ylim([0, nm+0.5 - 5])
ax.set_xlim([0, nm+0.5])
leg = ax.legend(fontsize=20, framealpha=1)
leg.set_draggable(1)
plt.grid(axis='y')
ax.set_aspect(1./ax.get_data_ratio())

#%% Combined plot

nm = max(nctrmax)
x = np.arange(1, nm+1)

fig, ax = plt.subplots()
for i in range(0, len(names)):
    contrast = 2*shoteffcorrect[i] - shoteffall[i]
    #ax.plot(x[:nctrmax[i]], shoteffcorrect[i], label=labels[i], marker='o', color=cols[i], lw=4, markersize=10)
    ax.plot(x[:nctrmax[i]], contrast, label=names[i], marker='x', color=cols[i], lw=4, markersize=10)
ax.plot([0, nm+1], [0, nm+1], ls='--', lw=4, color='black')
ax.set_xlabel("Hits Per Shot", fontsize=30)
ax.set_ylabel("Correctly Assigned Hits", fontsize=30)
ax.set_xticks(x[::2])
ax.set_yticks(x)
ax.tick_params(labelsize=25, width=2, length=5)
ax.set_ylim([0, nm+0.5])
ax.set_xlim([0, nm+0.5])
leg = ax.legend(fontsize=25)
leg.set_draggable(1)
plt.grid(axis='y')
ax.set_aspect(1./ax.get_data_ratio())
