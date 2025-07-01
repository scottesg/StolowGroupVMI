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
dt = 0.25 # ns, sampling interval
refwidth = 1.7 # ns, reference width fwhm
gausstresh = 0.3 # sets height threshold for reference+background peak
t0 = 529 # position of reference peak
tracelen = 4096
stksize = 64000

#%% Step 0: Create custom filter including background, reference, and smoothing
 
source = "20231213/162638"

nsize = 5 # number of runs to look at 
n0 = 0 # number of run to start with

runwfm = readwfm(datapath+source+"/wfm/daqdata%d.uint16"%(n0+1), tracelen, ch2=False)

# set a time axis
t = np.arange(0, tracelen)*dt

find_ref = True # False assumes constant reference peak position
interp = False # interpolate (x10) reference trace for locating the position

if interp:
    F0 = interpolate.interp1d(t, runwfm[0]) # interpolation function (first reference)
    extrapolate = 10 # new dt is 50 ps
    tint = np.linspace(t.min(), t.max(), extrapolate*len(t))

R00 = [] # stores reference peaks
args = [] # debug list to check if scan range is good


for l in range(nsize):
    print("Processing stack {}...".format(l+1+n0))
    
    runwfm = readwfm(datapath+source+"/wfm/daqdata%d.uint16"%(l+1+n0), tracelen, ch2=False)
    tresh = np.average(runwfm[:,:400]) # baseline  
    
    if l==0:
        x0 = -1*(runwfm[0] - tresh)
    else:
        ref = np.average(R00, axis=0) # line new reference peaks up with the average from the previous run(s)
        F0 = interpolate.interp1d(t, ref) # interpolation function (average reference)
    
    for i in range(0, stksize):
        if i % 1000 == 0: print("Shot #{} / {}".format(i, stksize))

        # subtract baseline and flip
        x = -1*(runwfm[i] - tresh)

        peaks, properties = find_peaks(x, height=20000, width=1)   
    
        if len(peaks)==1:
            peaks, properties = find_peaks(x, height=1500, width=1)  
            
            if len(peaks) < 2:
                
                # shift each reference peak to line up with the first one (arbitrary)
                if find_ref==True:
                    if interp==True:
                        F = interpolate.interp1d(t, runwfm[i]) # interpolation function (current reference)
                        arg = find_arg(F0(tint), F(tint), np.argmin(F(tint)), 150) # find shift
                        F = interpolate.interp1d(tint, np.roll(F(tint), arg)) # shift and interpolate again
                        x2 = F(t) # return to pre-interpolation size
                    else:
                        if l==0: arg = find_arg(x0, x, np.argmax(x0), 20)
                        else: arg = find_arg(ref, x, np.argmax(ref), 20)
                        x2 = np.roll(runwfm[i], arg)
                        
                args.append(arg)
            
                # subtract baseline and flip
                x2 = -1*(x2 - tresh)
                R00.append(x2)

print("Number of clean reference pulses: %d"%(len(R00)))

R00A = np.average(R00, axis=0)
            
plt.figure()
for i in range(300):
    plt.plot(t, R00[i])
plt.plot(t, R00A, "k--")
plt.xlabel("ToF (ns)")
plt.ylabel("Signal (mV)")
plt.title("Average Reference Pulse")      

gaussian = gauss(np.arange(int(2*12)+1)*dt, 12*dt, 1, refwidth) # Gaussian smoothing

# pad to double length of traces
gaussianpad = np.hstack((gaussian*windows.tukey(len(gaussian),sym=False),
                         np.zeros(tracelen-len(gaussian))))

# deconvolve gaussian
S0 = ifft(fft(R00A)/fft(gaussianpad)) # combined filter function

#%% Step 1: Create average 1-hit response

R1 = [] # stores 1-hit traces

# windows
w0 = np.hstack((np.zeros(2290),
                windows.tukey(1500, alpha=0.2),
                np.zeros(tracelen-1500-2290)))
w1 = np.hstack((np.zeros(20),
                windows.tukey(800, alpha=0.05),
                np.zeros(tracelen-800-20)))

plt.figure()
plt.subplot(131)
ax1 = plt.gca()
plt.subplot(132)
ax2=plt.gca()
plt.subplot(133)
ax3=plt.gca()
ax1.set_title("Raw Single Hits")
ax2.set_title("Deconvolved Traces")
ax3.set_title("Averaged Single Hit")

fig, ax = plt.subplots()

find_ref = False

c = 0
for l in range(nsize):
    print("Processing stack {}...".format(l+1+n0))
    
    runwfm = readwfm(datapath+source+"/wfm/daqdata%d.uint16"%(l+1+n0), tracelen, ch2=False)
    tresh = np.average(runwfm[:,:200]) # baseline  
              
    for i in range(stksize):
        if i % 1000 == 0: print("Shot #{} / {}".format(i, stksize))
        
        if find_ref:
            arg = find_arg(runwfm[0], runwfm[i], np.argmin(runwfm[0]), 20)
            runwfm[i] = np.roll(runwfm[i], arg)
        
        # subtract baseline and flip
        x0 = -1*(runwfm[i] - tresh)
        
        # find 1-hit peaks
        peaks, properties = find_peaks(x0, height=18000, width=2)
        if len(peaks)==1:
            if t[peaks[0]] < 533:
                peaks, properties = find_peaks(x0, height=1200)
                if len(peaks)==2:  
                    h = properties['peak_heights'][1]
                    #arg = np.argmax(x0*w0)
                    if t[peaks[1]]>550:
                        # apply filter
                        
                        # ax.plot(x0)
                        # if i>10: break
                        x = ifft(fft(x0)/fft(S0)).real
            
                        # shift maximum (inside window) to 500 (arbitrary)
                        arg = np.argmax(x*w1)
                        xs = np.roll(x*w1, (500-arg))
            
                        # normalize and store
                        R1.append(xs/xs.max())
                        c = c+1

print(c)
S1 = np.average(R1, axis=0) # Average 1-hit response

ax1.plot(t, w0*10000)
ax2.plot(t, w1)
ax3.plot(t, S1, "k--")
ax3.set_xlim(100, 175) 

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
nsize = 5 # number of runs to process

find_ref = False
debug = True
if debug:
    nbad = 0
    nsize = 1
    ntrace = 40
    
for l in range(nsize):
    print("Processing stack {}...".format(l+1+n0))
    
    runwfm = readwfm(datapath+source+"/wfm/daqdata%d.uint16"%(l+1+n0), tracelen, ch2=False)
    tresh = np.average(runwfm[:,:200]) # baseline  

    # create array to store hits
    THits = np.full((ntrace, (nmax+2)*2+2), np.nan)
    
    for i in range(ntrace):
        if i % 100 == 0: print("Shot #{} / {}".format(i, stksize))
        
        if find_ref:
            arg = find_arg(runwfm[0], runwfm[i], np.argmin(runwfm[0]), 20)
            runwfm[i] = np.roll(runwfm[i], arg)

        # subtract baseline, flip, and combine both channels
        x = -1*(runwfm[i+idx0] - tresh)
        
        hits, P = fitting_routine(nmax, x, t, t0, refwidth, F, S0, SNR, dt, gausstresh)
        THits[i,:len(hits)] = hits
        
        if debug and THits[i,0]>0:
        
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
                plt.xlim(500, 650) # might need to zoom out to see all events
                plt.ylim(0, 1.2)
    
    if debug:
        print("Number of Bad Fits: %d"%(nbad))
    else:
        # save results
        np.savetxt(datapath+source+"/hits/THits_%d.txt"%(l+1+n0), THits)
