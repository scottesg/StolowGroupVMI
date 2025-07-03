#%% Imports

import os
os.chdir("..")

import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate
from scipy.signal import find_peaks, windows
from numpy.fft import fft, ifft
from VMI3D_IO import readwfm

from VMI3D_PeakFinding import gauss, fitting_routine, find_arg

# Constants:
datapath = r"C:\Users\Scott\Python\VMI\data/"
dt = 0.5 # ns, sampling interval
refwidth = 2.0 # ns, reference width
gausstresh = 0.025 # sets height threshold for reference+background peak
ArgS = 0 # pts, shift applied to reference peak when adding to traces
t00 = 1011 # ns, position of reference peak
argscan = 30 # data points around maximum to scan for reference peak shift
tracelen = 4096
stksize = 64000

#%% Step 0: Create custom filter including background, reference, and smoothing
 
source = r"20240125\144323"

nsize = 4 # number of runs to look at 
n0 = 0 # number of run to start with

runwfm = readwfm(datapath+source+"/wfm/daqdata%d.uint16"%(n0+1), tracelen, ch2=True)

# set a time axis
t = np.arange(0, tracelen)*dt

t0 = t00 - ArgS*dt # shifted position of reference peak

R00 = [] # stores reference peaks
R0 = [] # stores 0-hit MCP traces
args = [] # debug list to check if scan range is good

find_ref = True # False assumes constant reference peak position
interp = False # interpolate (x10) reference trace for locating the position

if interp:
    F0 = interpolate.interp1d(t, runwfm[1][0]) # interpolation function (first reference)
    extrapolate = 10 # new dt is 50 ps
    tint = np.linspace(t.min(), t.max(), extrapolate*len(t))

for l in range(nsize):
    print("Processing stack {}...".format(l+1+n0))
    
    runwfm = readwfm(datapath+source+"/wfm/daqdata%d.uint16"%(l+1+n0), tracelen, ch2=True)
    tresh1 = np.average(runwfm[1][:,:400]) # reference channel baseline  
    tresh0 = np.average(runwfm[0][:,:400]) # data channel baseline
     
    if l==0:
        x0 = -1*(runwfm[1][0,:] - tresh1)
    else:
        ref = np.average(R00, axis=0) # line new reference peaks up with the average from the previous run(s)
        F0 = interpolate.interp1d(t, ref) # interpolation function (average reference)
    
    for i in range(0, stksize):
        if i % 1000 == 0: print("Shot #{} / {}".format(i, stksize))
        
        # subtract baseline and flip
        x = -1*(runwfm[1][i] - tresh1)
        
        # shift each reference peak to line up with the first one (arbitrary)
        if find_ref==True:
            if interp==True:
                F = interpolate.interp1d(t, runwfm[1][i]) # interpolation function (current reference)
                arg = find_arg(F0(tint), F(tint), np.argmin(F(tint)), argscan*10) # find shift
                F = interpolate.interp1d(tint, np.roll(F(tint), arg)) # shift and interpolate again
                x2 = F(t) # return to pre-interpolation size
            else:
                if l==0: arg = find_arg(x0, x, np.argmax(x0), argscan)
                else: arg = find_arg(ref, x, np.argmax(ref), argscan)
                x2 = np.roll(runwfm[1][i], arg)
        args.append(arg)
        
        # subtract baseline and flip
        x2 = -1*(x2 - tresh1)
        R00.append(x2)
        
        # find 0-hit mcp traces and shift by the same amount as reference
        xmcp = -1*(runwfm[0][i] - tresh0)   
        peaks, properties = find_peaks(xmcp, height=600, width=1.5)   
        if len(peaks)==0:
            R0.append(np.roll(xmcp, arg))
            
print("Number of reference pulses: %d"%(len(R00)))
print("Number of 0-hit traces: %d"%(len(R0)))

R00A = np.average(R00, axis=0)
R0A = np.average(R0, axis=0)
            
plt.figure()
for i in range(300):
    plt.plot(t, R00[i])
plt.plot(t, R00A, "k--")
plt.xlabel("ToF (ns)")
plt.ylabel("Signal (mV)")
plt.title("Average Reference Pulse")

plt.figure()
for i in range(300):
    plt.plot(t, R0[i])
plt.plot(t, R0A, "k--")
plt.xlabel("ToF (ns)")
plt.ylabel("Signal (mV)")
plt.title("Average MCP Background")

gaussian = gauss(np.arange(int(2*12)+1)*dt, 12*dt, 1, refwidth)

# pad to double length of traces
gaussianpad = np.hstack((gaussian*windows.tukey(len(gaussian),sym=False),
                         np.zeros(tracelen),
                         np.zeros(tracelen-len(gaussian))))

# combine reference pulse and mcp background, then deconvolve gaussian
signal = np.hstack((np.roll(R00A, -ArgS), R0A))
S0 = ifft(fft(signal)/fft(gaussianpad)) # combined filter function

#%% Step 1: Create average 1-hit response

R1 = [] # stores 1-hit traces

# window to clean the edges of the signal before storing it in R1
w1 = np.hstack((np.zeros(tracelen - 400),
                windows.tukey(1250, alpha=0.05),
                np.zeros(tracelen - (1250-400))))

# t for double length trace
t = np.hstack((t, t+tracelen*dt))

plt.figure()
plt.subplot(131)
ax1 = plt.gca()
plt.subplot(132)
ax2=plt.gca()
plt.subplot(133)
ax3=plt.gca()
ax1.set_title("Channels 1 & 2")
ax2.set_title("Deconvolved Traces")
ax3.set_title("Averaged Single Hit")

for l in range(nsize):
    print("Processing stack {}...".format(l+1+n0))
    
    runwfm = readwfm(datapath+source+"/wfm/daqdata%d.uint16"%(l+1+n0), tracelen, ch2=True)
    tresh1 = np.average(runwfm[1][:,:400]) # reference channel baseline  
    tresh0 = np.average(runwfm[0][:,:400]) # data channel baseline
              
    for i in range(stksize):
        if i % 1000 == 0: print("Shot #{} / {}".format(i, stksize))
        
        # subtract baseline and flip
        x0 = -1*(runwfm[0][i] - tresh0)
        
        # find 1-hit peaks
        peaks, properties = find_peaks(x0, height=1000, width=2)
        if len(peaks)==1:
            
            # combine both channels
            x0 = np.hstack((np.roll(-1*(runwfm[1][i] - tresh1), -ArgS), x0))  
            
            # apply filter
            x = ifft(fft(x0)/fft(S0)).real
            
            if i<200:
                ax1.plot(t, x0, c="C0", lw=3)            
                ax2.plot(t, x)
            
            # shift maximum (inside window) to 500 (arbitrary)
            arg = np.argmax(x*w1)
            xs = np.roll(x*w1, (500-arg))
            
            # normalize and store
            R1.append(xs/xs.max())

S1 = np.average(R1, axis=0) # Average 1-hit response

ax2.plot(t, w1)
ax3.plot(t, S1, "k--")

arg1 = np.argmax(S1)

# pad time and S1
tpad = np.hstack((-t[::-1]-1, t, t[-1]+1+t))
S1pad = np.hstack((np.zeros(len(t)), S1, np.zeros(len(t))))

#%% Step 2: Peak finding on all traces

# if using a filter from lower hit count data, load it here:
# filterpath = ""
# tpad, S1pad = np.load(filterpath)

F = interpolate.interp1d(tpad, S1pad.real) # interpolation function on S1, for fitting

nmax = 9 # maximum number of peaks to fit (up to 11)
nmaxloss = 0 # counts number of traces with more than nmax peaks
SNR = 15 # sets lower amplitude limit on peaks
ntrace = stksize # number of traces to process
idx0 = 0 # starting trace within stack
nsize = 4 # number of runs to process
n0 = 0 # number of run to start with

test = True
if test:
    nbad = 0
    nsize = 1
    ntrace = 20
else:
    # Create the output directory
    if os.path.exists(datapath+source+"/hits"):
        print("Warning: Hits directory already exists!")
    else:
        os.mkdir(datapath+source+"/hits")
    
for l in range(nsize):
    print("Processing stack {}...".format(l+1+n0))
    
    runwfm = readwfm(datapath+source+"/wfm/daqdata%d.uint16"%(l+1+n0), tracelen, ch2=True)
    tresh1 = np.average(runwfm[1][:,:400]) # reference channel baseline  
    tresh0 = np.average(runwfm[0][:,:400]) # data channel baseline
    
    # create array to store hits
    THits = np.full((ntrace, (nmax+2)*2+2), np.nan)
    
    for i in range(ntrace):
        if i % 100 == 0: print("Shot #{} / {}".format(i, stksize))

        # subtract baseline, flip, and combine both channels
        x = np.hstack((np.roll(-1*(runwfm[1][i+idx0] - tresh1), -ArgS),
                       -1*(runwfm[0][i+idx0] - tresh0)))
        
        result = fitting_routine(nmax, x, t, t0, refwidth, F, S0, SNR, dt, gausstresh)
        if result is None:
            print("Bad Shot! [Reference Pulse Not Found]")
            continue
        
        hits, P = result
        THits[i,:len(hits)] = hits
        
        if test and THits[i,0]>0:
        
            #plot each fit to find the minimum pulse-pair resolution 
            #roughly equal to 0.6ns see for instance index 67, 7
            #label the index on the figure title
            plt.figure()
            
            if THits[i,0]==99:
                plt.title("Stack %d; Shot %d; Bad Fit or Too Many Hits"%(l, i))
                nbad += 1
            else:
                plt.title("Stack %d; Shot %d; %d Events"%(l, i, THits[i,0]))

            x = ifft(fft(x) / fft(S0)).real
            targ = int(np.round(t0/dt))
            x = np.roll(x, targ)
            res = np.sum((x - np.sum(P, axis=0))**2)/len(t)
            plt.plot(t, x)        
            plt.plot(t, np.sum(P, axis=0), "k--", label = "Residual %.1e"%(res))
            plt.legend()
            for i in range(len(P)):
                plt.plot(t, P[i])
                plt.xlim(3050, 3200) # might need to zoom out to see all events
                plt.ylim(0, 0.25)
    
    if test:
        print("Number of Bad Fits: %d"%(nbad))
    else:
        # save results
        np.savetxt(datapath+source+"/hits/THits_%d.txt"%(l+1+n0), THits)

