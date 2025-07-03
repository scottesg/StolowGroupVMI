#%% Imports

import numpy as np
import matplotlib.pyplot as plt
import os
import glob
import numpy.fft as fft
from scipy.ndimage import gaussian_filter, fourier_shift
from VMI3D_IO import readroiframe, rois2xyi
from VMI3D_Functions import pts2img

#%% Default parameters

path = os.getcwd()
nframes = 4000
roidim = 31
maxrois = 20
ism = 3
fastctrs = False

# Testing
# pathi = r"C:\Users\Scott\Python\VMI\documents\DSC\20250528_202054"
# pathi = r"C:\Users\Scott\Python\VMI\documents\DSC\20250528_202054/run2/"
# path = pathi
# nimg = nframes
# i=5
# j=0

#%% Functions

# Centroid all runs in a directory
def ctrallruns(path, nframes, roidim, maxrois, ism):
    nruns = len(glob.glob(path+'/run*'))
    print("Centroiding %d runs:"%nruns)
    
    for i in range(1, nruns+1):
        print("Run #%d..."%i, end='')
        pathi = path + '/run%d/'%i
        ctrs = centroidROIs(pathi, nframes, roidim, maxrois, ism) 
        np.save(pathi + "ctrs.npy", ctrs)
        print("Done!") 

# get stack of images from raw roi data in nami, optionally subtract projection 
def centroidROIs(path, nframes, roidim, maxrois, ism):

    ctrs = []
    
    # Load projection
    projs = np.fromfile(path + 'projsb1.int16', dtype=np.int16)
    projs = np.reshape(projs, (512, nframes))
    projs = projs/512

    fp = open(path + 'roisposb1.single', 'rb')

    for i in range(nframes):
        xpos, ypos, rois, nrois = readroiframe(fp, i+1, roidim, maxrois)
        
        if fastctrs:
            ctrs.extend(rois2xyi(xpos, ypos, rois, nrois, i))
        else:
            proji = projs[:,i]
            
            ctrf = centroidFrameROIs(xpos, ypos, rois, nrois, proji)
            ctrs.extend(ctrf)
    
    fp.close()
    
    # return [xp yp xvals yvals zvals  nrois  badptrs imgo t1 t2]
    return np.array(ctrs)

def centroidFrameROIs(xpos, ypos, rois, nrois, proji=None):
    
    ctrs = []
    
    for j in range(nrois):
        xj = xpos[j]
        yj = ypos[j]
        roij = rois[j]
        
        if proji:
            projij = proji[xj:xj+roidim]
            roij = roij - projij
        roij = gaussian_filter(roij, ism)
        
        xvc, yvc, zvc = centroidsingleROI(roij, roidim)
        
        ctr = [xj + xvc, yj + yvc, zvc]
        ctrs.append(ctr)
        
    return ctrs

# find precise position of global maximum in image y
def centroidsingleROI(roi, dim):

    pos = int(dim/2)
    xs = roi[pos,:]
    ys = roi[:,pos]
    
    errx, xp = maxROIslice(xs, 0.001)
    erry, yp = maxROIslice(ys, 0.001)
    zp = np.max(roi)
    
    return xp, yp, zp

# find max peaks in y 
def maxROIslice(slc, tol):
    
    slclen = len(slc)
    slcmax = max(slc)
    indmax = np.argmax(slc)
    shfttol = 0.003
    
    # default values
    poso = slclen/2
    hgto = slcmax
    shft = 0
    edge = False # Maximum on edge of slice

    shftdel = 0.5
    lastval = slcmax
    
    # special case for shft near .5
    if (indmax == 0) or (indmax > slclen-2):
        edge = True
        
    else:
        dely1 = (slcmax - slc[indmax-1])/slcmax
        dely2 = (slcmax - slc[indmax+1])/slcmax
        
        if (dely1 < shfttol) or (dely2 < shfttol):
            shftdel = 0.5
            slc = fft.ifft(fourier_shift(fft.fft(slc), 0.5)).real
            lastval = max(slc)
            indmax = np.argmax(slc)

        dl = 0.3
        shft = 0.1
        
        while abs(dl) > tol:
            slctmp = fft.ifft(fourier_shift(fft.fft(slc), shft)).real
            curval = max(slctmp)
            if curval <= lastval:
                dl = -dl/1.77 # go the other way
            lastval = curval
            shft = shft + dl
            
        if shft > 0.499:
            shft = shft - 1
    
        poso = indmax - shft - tol
        hgto = lastval

    if shftdel != 0:
        poso = poso - shftdel
        
    return edge, poso

#%% Run in current directory

if __name__=="__main__":
    
    #projs, ctrs = centroidROIs(path, nframes, roidim, maxrois, ism)
    ctrallruns(path, nframes, roidim, maxrois, ism)
    
    ctrs = np.load(path + r"/run2/ctrs.npy")
    img = pts2img(ctrs, 512, 5)
    plt.imshow(img)
