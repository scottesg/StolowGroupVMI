#%% Imports

import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

from time import sleep
from datetime import datetime as dt
from timeit import default_timer as timer
from shutil import move, copyfile
from VMI3D_Functions import pts2img
from VMI3D_IO import readwfm, readctr, rawdaqtovolts
from VMI3D_DASFunctions import writevmi3dparams, camdaqhgtxcorrn, fstdiff

#%% Generate data folder from current date/time

currentdatefolder = '20250528_202054' #dt.now().strftime("%Y%m%d_%H%M%S")
datafolder = "C:\\Users\\Scott\\Python\\VMI\\documents\\DSC\\" #"e:\\data\\vmi\\"
datadir = datafolder + currentdatefolder + '\\'

if os.path.exists(datadir):
    print("Warning: Directory already exists!")
else:
    os.mkdir(datadir)
    
#%% Define parameters

# Digitizer parameters
npts = 2048*2 # Half pre-trigger, half post-trigger
daqdc = 0.0002761 # DC level

# Camera parameters
nframes = 64000 # Shots/run
nstacks = 1
imgsize = 512
iexposure = 50
roix0 = 544
roiy0 = 356
storeimages = 0
storertcs = 1
dclvl = 2.8
rtclvl = 3.2
nruns = 10

test = True
plots = True
xcorr = True

corpos = np.zeros(nruns)

#%% Acquire data

# Create figures
# TODO: Make figures update each loop

def makefig(n, x, y):
    backend = matplotlib.get_backend()
    
    fig = plt.figure(n)
    fman = fig.canvas.manager.window
    
    if backend=='tkagg':
        fman.wm_geometry("+%d+%d" % (x, y))
    
    elif backend=='qtagg':
        fman.move(x, y)
    
    ax = fig.gca()
    plt.pause(0.1)
    return fig, ax

if plots:
    fig1, ax1 = makefig(1, 0, 0)
    fig2, ax2 = makefig(2, 625, 0)
    
    if xcorr:
        fig3, ax3 = makefig(3, 1250, 0)
        fig4, ax4 = makefig(4, 0, 550)
        
    plt.pause(1)

for currentrun in range(1, nruns+1):
    currentrun = (currentrun-1)%3 + 1
    s = timer()
    print("Acquiring run #%d..."%currentrun)

    # Create run folder
    datadirc = datadir + "run%d"%currentrun
    if os.path.exists(datadirc):
        print("Warning: Directory already exists!")
    else:
        os.mkdir(datadirc)
        
    # Write camera parameters for C program to read
    # Filename hard coded to c:/data/vmi3drunparamsrtc.txt
    # writevmi3dparams(datadirc, nstacks, nframes, iexposure, imgsize,
    #                  roix0, roiy0, storeimages, storertcs, dclvl, rtclvl)
    
    # Start acquisition script
    if not test:
        alazardirc = r'C:\AlazarTech\ATS-SDK\23.1.0\Samples_Python\ATS9373\NPT_StreamToMemory'
        os.chdir(alazardirc)
        import ATS9373_NPT_StreamToMemory_edited as stream
        stream.start()
    else:
        wait=True
        t0 = timer()
        c = 0
        while wait:
            t = timer() - t0
            if t>(64000/1000):
                wait=False
            c = c+1
            sleep(0.1)
        print(c)
            
        # copyfile(r"E:\data\vmi\20240207_135554\run1\daqdata.uint16",
        #          "C:\\data\\daqdata.uint16")
        # copyfile(r"E:\data\vmi\20240207_135554\run1\blbposb1.single",
        #          datadirc + "\\blbposb1.single")
    print("Done. Time elapsed: {}".format(timer() - s))
    
    #move("C:\\data\\daqdata.uint16", datadirc + "\\daqdata.uint16")
    #print('\a') # Generates beep sound
    
    # Change to run directory
    os.chdir(datadirc)
    
    if plots:
        
        ax1.cla()
        ax2.cla()
        if xcorr:
            ax3.cla()
            ax4.cla()
        
        t = np.hstack(readwfm('daqdata.uint16', npts, ch2=True))
        v1 = rawdaqtovolts(t, 1)
        v1p = np.mean(v1, 0)
        ax1.plot(v1p)
        ax2.imshow(-1*v1, vmin=0, vmax=0.01)
        ax2.set_aspect(1./ax2.get_data_ratio())
        
        # Camera/Digitizer correlation test
        if xcorr:
            (xcorro1, xcorro2, valc, vald, p, d) = camdaqhgtxcorrn(npts, nframes, daqdc)
            xcorro2d = fstdiff(xcorro2)
            d2 = np.argmax(xcorro2d)
            corpos[currentrun-1] = d2+1
            
            ax3.plot(xcorro2)
            ax3.set_title('Max at %d'%(d2+1))
            
            ctrs = readctr('blbposb1.single', nframes)
            
            img = pts2img(ctrs*2, imgsize, 2)
            ax4.imshow(img)
            ax4.set_title("Full VMI Image")
            ax4.axes.get_xaxis().set_visible(False)
            ax4.axes.get_yaxis().set_visible(False)
            
            hps = len(ctrs)/nframes
            print(hps)
            
        plt.pause(0.1)#nframes/1000)

os.chdir(datadirc)