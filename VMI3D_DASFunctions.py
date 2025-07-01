import numpy as np
from VMI3D_IO import readwfm, rawdaqtovolts

def writevmi3dparams(datadir, nstacks, nframes, iexposure, imgsize,
                     roix0, roiy0, storeimgs, storertcs, dclvl, rtclvl):

    # Args: Run driectory, number of stacks, frames/stack, exposure(ms), image size (typically 512),
    # x offset, y offset, store full images? (y/n), store rtcs? (y/n), dc level in data, threshold for sensing ROIs
    
    file = open('C:/data/vmi3drunparamsrtc.txt','w')   
    
    file.write('data_folder %s\n'%datadir)
    file.write('nstacks %d\n'%nstacks)
    file.write('nframes %d\n'%nframes)
    file.write('exposure_us %d\n'%iexposure)
    file.write('imagesize %d\n'%imgsize)
    file.write('roix0 %d\n'%roix0)
    file.write('roiy0 %d\n'%roiy0)
    file.write('storeimgs %d\n'%storeimgs)
    file.write('storertcs %d\n'%storertcs)
    file.write('dclvl %f\n'%dclvl)
    file.write('rtclvl %f\n'%rtclvl)
    
    file.close()

# Generate intensity cross corr. For current camera and daq datasets
# assumes camera data are rtc in blposb1 and alzardata files
# for each image and frame in the current folder this generates an
# intensity metric for the daq frame (integration of the waveform) and for
# the image sum of all roi  z values in the rtc.
# if the acquisition system is working properly, these two metrics should be highly correlated
def camdaqhgtxcorrn(npts, nimg, daqdc):
    
    d = np.hstack(readwfm('daqdata.uint16', npts, ch2=True))
    d = -1*(rawdaqtovolts(d)+daqdc) # this is a minor adjustment; not really needed
    
    p = np.fromfile('blbposb1.single', dtype=np.single)
    p = np.reshape(p, (nimg, 60)).T
    
    ps = np.sum(p, 0) # metric for camera roi intensities
    valc = p[0]
    vald = np.sum(d, 1) # metric for daq frame intensities
    
    xcorro1 = np.correlate(valc, vald, "full")
    xcorro2 = np.correlate(ps, vald, "full")
    
    return (xcorro1, xcorro2, valc, vald, p, d)

def fstdiff(y):

    n = len(y)
    yo = np.zeros(n)
    for i in range(n-1):
      yo[i] = y[i+1]-y[i]
    
    yo[n-1] = yo[n-2]
    
    return yo