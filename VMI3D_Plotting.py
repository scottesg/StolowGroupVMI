import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from VMI3D_Functions import pts2img, getngrs, fastpeaks
from VMI3D_IO import readctr, readwfm, genT, readrois

# VMI image
def vpImage(pts, dim, smooth):
    img = pts2img(pts, dim, smooth)
    
    fig, ax = plt.subplots()
    ax.imshow(img)
    ax.set_title("Full VMI Image")
    ax.axes.get_xaxis().set_visible(False)
    ax.axes.get_yaxis().set_visible(False)
    return fig
    
# Correlation plot
def vpCorrelation(pts, nbins):
    
    ci = pts[:,3]
    pi = pts[:,4]
    
    fig, ax = plt.subplots()
    hist = ax.hist2d(ci, pi, nbins, cmap='gray_r')
    plt.colorbar(hist[3], ax=ax)
    ax.set_xlim((np.percentile(ci, 1), np.percentile(ci, 99)))
    ax.set_ylim((np.percentile(pi, 1), np.percentile(pi, 99)))
    ax.set_title("Correlation Plot")
    ax.set_ylabel("Pickoff Amplitude")
    ax.set_xlabel("Camera Intensity")
    return fig

# Time distribution
def vpTimehist(pts, rf=30):
    
    t = pts[:,2]
    tmax = np.median(t)
    
    tr = t[t>tmax-rf]
    tr = tr[tr<tmax+rf]
    
    d = np.diff(np.unique(tr)).min()
    nbt = (max(tr) - min(tr))/d
    
    if nbt < 5000:
        ul = max(tr) + d/2
        ll = min(tr) - d/2
        bins = np.arange(ll, ul + d, d)
    else:
        ul = max(tr)
        ll = min(tr)
        bins = np.linspace(ll, ul, 1000)
    
    hist, bins = np.histogram(tr, bins)
    bins = bins[:-1] + (bins[1]-bins[0])
    
    fig, ax = plt.subplots()
    ax.plot(bins, hist)
    ax.set_xlim((ll, ul))
    ax.set_title("Histogram of Arrival Times")
    ax.set_xlabel("Time [ns]")
    return fig 
    
# PHDs
def vpPHD(pts, phdtype, nbins):
    
    if phdtype == 'c':
        ind = 3
        title = 'Camera PHD'
    elif phdtype == 'p':
        ind = 4
        title = 'Pickoff PHD'
    else: return
        
    fig, ax = plt.subplots()
    ax.hist(pts[:,ind], nbins)
    ax.set_title(title)
    return fig

# All VMI plots
def vmi3dplots(pts, dim, smooth=2, cbins=200, phdbins=200, save=None, name=''):
    
    figImage = vpImage(pts, dim, smooth)
    figCor = vpCorrelation(pts, cbins)
    figTH = vpTimehist(pts, rf=30)
    figCamPHD = vpPHD(pts, 'c', phdbins)
    figPickoffPHD = vpPHD(pts, 'p', phdbins)
    
    if not save == None:
        figImage.savefig(save + "/vmi3dplots_{}_VMI.png".format(name))
        figCor.savefig(save + "/vmi3dplots_{}_Correlation.png".format(name))
        figTH.savefig(save + "/vmi3dplots_{}_TimeHistogram.png".format(name))
        figCamPHD.savefig(save + "/vmi3dplots_{}_CameraPHD.png".format(name))
        figPickoffPHD.savefig(save + "/vmi3dplots_{}_PickoffPHD.png".format(name))
        
# Data set preview plots
def datapreview(cpath, wpath, smth, blpts, dt, dim,
                name, stksize, trsign=-1, fasttraces=True, fastheight=20, userois=False):
    
    print("Reading data...")
    if userois:
        ctrs = readrois(cpath, stksize)
    else:
        ctrs = readctr(cpath)
    wfms = readwfm(wpath)
    t = genT(len(wfms[0]), dt)
    print("Done reading data.")
    
    nshots = len(wfms)
    nstks = int(nshots/stksize)
    ngrs = getngrs(ctrs, stksize, nstks)
    
    print("Analysing traces...")
    allhits = []
    for i in range(0, nshots):
        if i % 1000 == 0: print("Trace #{} / {}".format(i, nshots))
        if fasttraces:
            allhits.append(fastpeaks(trsign*wfms[i], fastheight,
                                     blpts, smth, plot=False))
        else:
            print("Can't do slow peak-picking preview yet!")
            return None
        
    print("Done analysing traces.")
    
    print("Plotting...")
    averagewfm = np.mean(wfms, axis=0)
    wfmmax = np.max(trsign*wfms, axis=1)
    
    nhits = np.zeros(nshots)
    pos = []
    pki = []
    for i in range(0, nshots):
        shot = allhits[i]
        nhits[i] = len(shot)
        pos.extend(shot[:,0])
        pki.extend(shot[:,1])
    pos = np.array(pos)
    pki = np.array(pki)
    
    nhitstime = [sum(nhits[i*stksize:(i+1)*stksize]) for i in range(0, nstks)]
    nctrstime = np.sum(ngrs, axis=1)
    
    # Average of all scope traces
    fig, ax = plt.subplots()
    ax.plot(t, gaussian_filter(trsign*averagewfm,10))
    ax.set_xlabel("Time [ns]")
    ax.set_ylabel("Average Amplitude")
    ax.set_title("Average of {} traces".format(len(wfms)))
    plt.savefig(name + "/PV_AverageScopeTraces.png")
    
    # VMI Image Using All Centroids
    fig, ax = plt.subplots()
    img = pts2img(ctrs, dim, 2)
    ax.imshow(img)
    ax.axes.get_xaxis().set_visible(False)
    ax.axes.get_yaxis().set_visible(False)
    ax.set_title("VMI Image Using All Centroids")
    plt.savefig(name + "/PV_VMIAllCentroids.png")
    
    # Maximum Amplitude of Each Trace (Saturation Check)
    fig, ax = plt.subplots()
    ax.hist(wfmmax, 500)
    ax.set_xlabel("Amplitude")
    ax.set_ylabel("Count")
    ax.set_title("Maximum Amplitude Distribution (Saturation Check)")
    plt.savefig(name + "/PV_SaturationCheck.png")
    
    # Yield Over Time in Centroids/Second
    fig, ax = plt.subplots()
    nsec = stksize/1000
    x = nsec*np.arange(0, nstks)
    yc = np.divide(nctrstime, nsec)
    ys = np.divide(nhitstime, nsec)
    ax.plot(x, yc, color='red', label='Centroids')
    ax.plot(x, ys, color='blue', label='Pickoff Hits')
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Hits/Second")
    ax.set_title("Yield Over Time in Hits/Second")
    leg = ax.legend(fontsize=20)
    leg.set_draggable(1)
    plt.savefig(name + "/PV_YieldOverTime.png")
    
    # Histogram of Centroid Number per Shot
    fig, ax = plt.subplots()
    bins = np.arange(0, 21)
    ax.hist(np.reshape(ngrs, np.size(ngrs)), bins=bins, fill=False, label='Centroids', edgecolor='red')
    ax.hist(nhits, bins=bins, fill=False, label='Pickoff Hits', edgecolor='blue')
    ax.set_xlabel("Number of Hits")
    ax.set_xticks(bins[:-1]+0.5)
    ax.set_xticklabels(bins[:-1])
    ax.set_ylabel("Count")
    ax.set_title("Histogram of Hits per Shot")
    leg = ax.legend(fontsize=20)
    leg.set_draggable(1)
    plt.savefig(name + "/PV_HitsPerShot.png")
    
    # PHDs
    fig, ax = plt.subplots(2, 1)
    ax[0].hist(ctrs[:,2], 40, range=(min(ctrs[:,2]), np.percentile(ctrs[:,2], 99)))
    ax[0].set_xlabel("Centroid Intensity")
    ax[0].set_ylabel("Count")
    ax[0].set_title("Centroid PHD")
    ax[1].hist(pki, 40, range=(min(pki), np.percentile(pki, 99)))
    ax[1].set_xlabel("Pickoff Amplitude")
    ax[1].set_ylabel("Count")
    ax[1].set_title("Scope PHD")
    fig.suptitle("Pulse Height Distributions")
    plt.savefig(name + "/PV_PHD.png")
       
