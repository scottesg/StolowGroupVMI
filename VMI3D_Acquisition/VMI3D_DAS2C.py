#%% Imports

import os
import sys
import time
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

# Path of Alazar Python SDK
sys.path.append("C:\\AlazarTech\\ATS-SDK\\23.1.0\\Samples_Python\\Library\\")

# Path of EVT Python SDK
sys.path.append("C:\Program Files\EVT\eSDK\Examples\EVT_Py\\")

import dg535
from datetime import datetime as dt
from timeit import default_timer as timer
from VMI3D_Functions import pts2img
from VMI3D_IO import readwfm, readctr, rawdaqtovolts
from VMI3D_DASFunctions import writevmi3dparams, camdaqhgtxcorrn, fstdiff, DAS2C_start, camconfig

#%% Generate data folder from current date/time

currentdatefolder = dt.now().strftime("%Y%m%d_%H%M%S")
datafolder = "V:\\data\\"
datadir = datafolder + currentdatefolder + '\\'

if os.path.exists(datadir):
    print("Warning: Directory already exists!")
else:
    os.mkdir(datadir)
    
#%% Define parameters

twochannel = True
usecamera = True

# Address for DG535
dgid = 'GPIB0::2::INSTR'

# Location of currentevtacq.exe
acqpath = "V:\\data\\currentevtacq.exe"

# Digitizer parameters
npts = 2048*5 # Half pre-trigger, half post-trigger. Pretrigger caps at 4088.
daqdc = 0.0002761 # DC level
threshold = 2000

# Camera parameters
nframes = 32000 # Shots/run
nstacks = 1
imgsize = 512
iexposure = 5 # Some values of this may also cause errors?
roix0 = 552 # Only some values will work correctly,
#             incorrect value will either fail or give a line artefact in the image
roiy0 = 356
storeimages = 0
storertcs = 1
dclvl = 2.8
rtclvl = 3.2
nruns = 1

params = [
    ["UInt32", "Width", imgsize],
    ["UInt32", "Height", imgsize],
    ["Enum", "PixelFormat", "Mono8"],
    ["Enum", "ADC", "Bit8"],
    ["UInt32", "OffsetX", roix0],
    ["UInt32", "OffsetY", roiy0],
    ["Enum", "AcquisitionMode", "MultiFrame"],
    ["Enum", "TriggerSelector", "AcquisitionStart"],
    ["Enum", "TriggerMode", "On"],
    ["Enum", "TriggerSource", "Hardware"],
    ["UInt32", "Exposure", iexposure],
    ["Bool", "AutoExposure", 0],
    ["UInt32", "AutoExpSet", 128],
    ["UInt32", "AutoExpIGain", 4],
    ["UInt32", "FrameRate", 1000],
    ["UInt32", "FrameRatemHz", 1000000],
    ["UInt32", "LineTime", 625],
    ["UInt32", "AcquisitionFrameCount", 1],
    ["UInt32", "Gain", 256],
    ["UInt32", "PGAGain", 40],
    ["Bool", "HCG", 1],
    ["UInt32", "GevSCPSPacketSize", 9000],
    ["Enum", "GPO_0_Mode", "Exposure"],
    ["Bool", "GPO_0_Polarity", 0],
    ["Enum", "GPO_1_Mode", "Exposure"],
    ["Bool", "GPO_1_Polarity", 0],
    ["Enum", "GPO_3_Mode", "Readout"],
    ["Bool", "GPO_3_Polarity", 0],
    ["Enum", "GPI_2_Mode", "Exposure"],
#    ["Bool", "GPI_2_Polarity", 0],
    ["UInt32", "GPI_2_Debounce_Count", 50],
    ["Enum", "GPI_4_Mode", "Exposure"],
#    ["Bool", "GPI_4_Polarity", 0],
    ["UInt32", "GPI_4_Debounce_Count", 50],
    ["Enum", "GPI_5_Mode", "Test_Gen_Uart_Rxd"],
#    ["Bool", "GPI_5_Polarity", 0],
    ["UInt32", "GPI_5_Debounce_Count", 50],
    ["Enum", "GPI_Start_Exp_Mode", "GPI_4"],
    ["Enum", "GPI_Start_Exp_Event", "Rising_Edge"],
    ["Enum", "GPI_End_Exp_Mode", "Internal"],
    ["Enum", "GPI_End_Exp_Event", "Rising_Edge"]]

camconfig(params)

plots = False
xcorr = True

corpos = np.zeros(nruns)

#%% Acquire data

# Open connection to DG535
dg = dg535.DG535(address=dgid)

# Create figures
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
    
    if usecamera and xcorr:
        fig3, ax3 = makefig(3, 1250, 0)
        fig4, ax4 = makefig(4, 0, 550)
        
    plt.pause(0.1)

for currentrun in range(1, nruns+1):

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
    writevmi3dparams(datadirc, nstacks, nframes, iexposure, imgsize,
                     roix0, roiy0, storeimages, storertcs, dclvl, rtclvl)
    
    # Start acquisition script
    DAS2C_start(threshold, dg, npts, nframes, datadirc+"\\daqdata.uint16",
                acqpath, twochannel, usecamera)

    print("Done. Time elapsed: {}".format(timer() - s))
    print('\a') # Generates beep sound
    
    if plots:
              
        ax1.cla()
        ax2.cla()
        if usecamera and xcorr:
            ax3.cla()
            ax4.cla()
        
        t = np.hstack(readwfm(datadirc+'\\daqdata.uint16', npts, ch2=True))
        v1 = rawdaqtovolts(t, 1)
        v1p = np.mean(v1, 0)
        ax1.plot(v1p)
        ax2.imshow(-1*v1, vmin=0, vmax=0.01)
        ax2.set_aspect(1./ax2.get_data_ratio())
        
        # Camera/Digitizer correlation test
        if usecamera and xcorr:
            (xcorro1, xcorro2, valc, vald, p, d) = camdaqhgtxcorrn(npts, nframes, daqdc,
                                                                   datadirc+'\\blbposb1.single',
                                                                   datadirc+'\\daqdata.uint16')
            xcorro2d = fstdiff(xcorro2)
            d2 = np.argmax(xcorro2d)
            corpos[currentrun-1] = d2+1
            
            ax3.plot(xcorro2)
            ax3.set_title('Max at %d'%(d2+1))
            
            ctrs = readctr(datadirc+'\\blbposb1.single', nframes)
            
            img = pts2img(ctrs*2, imgsize, 2)
            ax4.imshow(img)
            ax4.set_title("Full VMI Image")
            ax4.axes.get_xaxis().set_visible(False)
            ax4.axes.get_yaxis().set_visible(False)
            
            hps = len(ctrs)/nframes
            print(hps)
            
        plt.pause(0.1)