import numpy as np
from scipy.io import loadmat
import glob
import itertools
from shutil import copyfile
from scipy.ndimage import gaussian_filter

# Read in flare or evt camera image stack
# dim = dimension of image, assuming square
def readimg(file, dim):
    a = np.fromfile(file,dtype=np.uint8)
    nimg = int(len(a)/dim**2)
    a = np.reshape(a,(nimg,dim,dim))
    return np.transpose(a, (0,2,1))

def readcsv(file):
    return np.genfromtxt(file, delimiter=',');

# Read in tek scope or daq card trace stack (or stacks using *)
# If .mat file, combines all stacks together
# Returns stack of traces
# dqdim = length of daq traces
def readwfm(file, dqdim, groupstks=False, ch2=False):
    
    if '*' in file: # read and combine multiple files
        nf = len(glob.glob(file))
        file = file.replace('*','{}')
        traces = []
        if ch2:
            refs = []
        for i in range(0, nf):
            fname = file.format(i+1)
            fdata = readwfm(fname, dqdim=dqdim, ch2=ch2)
            if ch2:
                tracesi, refsi = fdata
                traces.append(tracesi)
                refs.append(refsi)
            else: traces.append(fdata)
        if not groupstks:
            traces = np.vstack([traces[i] for i in range(0, len(traces))])
            if ch2:
                refs = np.vstack([refs[i] for i in range(0, len(refs))])
        if ch2:
            traces = [traces, refs]
        
    elif file.endswith(".wfm.npy"): # unpacked npy file
        traces = np.load(file)
        
    elif file.endswith(".uint16"): # daq binary file
        data = np.fromfile(file, dtype=np.uint16)
        n = int(len(data)/dqdim)
        if ch2:
            n = int(n/2)
            signal = data[::2]
            refs = data[1::2]
            signal = np.reshape(signal, (n, dqdim))
            refs = np.reshape(refs, (n, dqdim))
            traces = [signal, refs]
        else: traces = np.reshape(data, (n, dqdim))
        
    elif file.endswith(".mat"): # matlab file
        data = loadmat(file)
        if 'daqdataa' in data: traces = data['daqdataa']
        elif 'daqdata' in data: traces = data['daqdata']
        traces = np.transpose(traces, (0,2,1))
        if groupstks:
            traces = [traces[i] for i in range(0, len(traces))]
        else:
            traces = np.vstack([traces[i] for i in range(0, len(traces))])
        
    return traces

# Generate time array
def genT(n, dt):
    return np.linspace(0, (n-1)*dt, n)

# Function for reading and converting centroid data
# Internal format is a (n,5) array of n centroids
# with rows of (x, y, h, frame, nctrs)
#
# Read in centroids from binary, numpy, or Matlab file
# or base name for series of files
# Returns centroids in internal format
def readctr(file, stksize, groupstks=False):
    
    form = 0
    
    if '*' in file: # read and combine multiple files
        nf = len(glob.glob(file))
        file = file.replace('*','{}')
        data = []
        for i in range(0, nf):
            fname = file.format(i+1)
            fdata = readctr(fname, stksize, groupstks=groupstks)
            data.extend(fdata)
        if not groupstks:
            for i in range(0, len(data)):
                data[i][:,3] += stksize*i
            data = list(itertools.chain(*data))
            data = np.array(data)
            
    elif file.endswith(".npy"): # numpy file in internal format
        data = np.load(file)
        if groupstks:
            data = groupctrstks(data, stksize)
    
    elif file.endswith(".single"): # binary file from real-time centroiding
        a = np.fromfile(file, dtype=np.single)
        nimg = int(len(a)/60)
        a = np.reshape(a, (nimg, 60)).T
        ngr = a[0,:]
        data = np.zeros((int(sum(ngr)),5))

        c = 0
        for i in range(0, a.shape[1]):
            for j in range(0, int(a[0,i])):
                data[c,:3] = a[3+3*j:6+3*j,i]
                data[c,3] = i
                data[c,4] = ngr[i]
                c = c+1
    
    elif file.endswith(".mat"): # matlab centroids
        ip = loadmat(file)
        
        if 'xvalsa' in ip: 
            x = ip['xvalsa']
            y = ip['yvalsa']
            z = ip['zvalsa']
            ngr = ip['nroisa']
            form = 1
        elif ('roictrsx' in ip) and ('nrois' in ip):
            x = ip['roictrsx']
            y = ip['roictrsy']
            z = ip['roictrsz']
            ngr = ip['nrois']
            form = 4
        elif 'nroisa' in ip: 
            x = ip['roictrsx']
            y = ip['roictrsy']
            z = ip['roictrsz']
            ngr = ip['nroisa']
            form = 2
        elif 'xvals' in ip:
            x = np.reshape(ip['xvals'], (1, *ip['xvals'].shape))
            y = np.reshape(ip['yvals'], (1, *ip['yvals'].shape))
            z = np.reshape(ip['zvals'], (1, *ip['zvals'].shape))
            ngr = ip['nrois']
            form = 3
            
        if groupstks:
            gdata = []
        else:
            data = np.zeros((int(np.sum(ngr)),5))
        
        c = 0
        for k in range(0, len(ngr)):
            if groupstks: 
                c=0
                data = np.zeros((int(np.sum(ngr[k])),5))
            for i in range(ngr.shape[1]):
                for j in range(0, ngr[k,i]):
                    data[c,0] = x[k,j,i]
                    data[c,1] = y[k,j,i]
                    data[c,2] = z[k,j,i]
                    if groupstks:
                        data[c,3] = i
                    else:
                        data[c,3] = k*ngr.shape[1] + i
                    data[c,4] = ngr[k,i]
                    c = c+1
            if groupstks: gdata.append(data)
        if groupstks: data = gdata
    
    # exception for format 3 ml centroids
    if form==3 and groupstks==False: return [data]
    
    return data

# Group centroids in to a list of arrays for each stack
def groupctrstks(ctrs, stksize):
    
    groupedctrs = []
    nstk = 0
    inds = ctrs[:,3]
    ind1 = np.searchsorted(inds, (nstk)*stksize)
    ind2 = np.searchsorted(inds, (nstk+1)*stksize)
    while (ind1 < ind2):
        ctrstk = ctrs[ind1:ind2]
        ctrstk[:,3] = ctrstk[:,3] - nstk*stksize
        groupedctrs.append(ctrstk)
        nstk = nstk+1
        ind1 = ind2
        ind2 = np.searchsorted(inds, (nstk+1)*stksize)
    return groupedctrs

# Read roi file (or files using *) into (x,y,i) format
# Rough values only using maxima of the rois
def readrois(file, nframe):
    
    if '*' in file:
        nf = len(glob.glob(file))
        file = file.replace('*','{}')
        data = []
        for i in range(0, nf):
            print(i+1)
            fname = file.format(i+1)
            fdata = readrois(fname, nframe)
            fdata[:,3] += 4000*i
            data.extend(fdata)
    
    elif file.endswith(".single"):
        data = []
        for i in range(1, nframe+1):
            dataframe = readroiframe(file, i)
            data.extend(rois2xyi(*dataframe, i))
            
    return np.array(data)

# Read in rois for a given frame number from binary file
# roidim = dimension of each roi, assuming square
# maxrois = number of rois per frame that space has been alloted to in the file
# Returns xpos: array of length maxrois containing x values of each roi in frame
#         ypos: array of length maxrois containing y values of each roi in frame
#         rois: array of shape maxrois*roidim*roidim containing the rois
#         nrois: number of rois in the frame
def readroiframe(file, frame, roidim=31, maxrois=20):
    
    rois = np.zeros((maxrois, roidim, roidim), dtype=np.single)
    xpos = np.zeros(maxrois, dtype=np.uint16)
    ypos = np.zeros(maxrois, dtype=np.uint16)
    
    roiblk = (roidim * roidim + 12) # 973 for roidim = 31
    frmblk = roiblk * maxrois # 19232 for maxrois = 20
    frmoffset = frmblk * (frame - 1)
    
    nrois = np.fromfile(file, dtype=np.single, offset=frmoffset, count=1)[0]
    if nrois > 0:
        for i in range(0, int(nrois)):
            roisize = roidim * roidim
            step = frmoffset + roiblk * i
            ypos[i] = np.fromfile(file, dtype=np.single, offset=step+4, count=1)
            xpos[i] = np.fromfile(file, dtype=np.single, offset=step+8, count=1)
            roi = np.fromfile(file, dtype=np.uint8, offset=step+12, count=roisize)
            roi = np.reshape(roi, (roidim, roidim))
            rois[i, :, :] = roi
    return xpos, ypos, rois, nrois

# Converts the output of readroiframe to an array of xyi data (for rough preview)
# with rows of (x, y, intensity)
# Intensity is estimated roughly by taking the maximum value in the roi
def rois2xyi(xpos, ypos, rois, nrois, framenumber):
    n = int(nrois)
    if n==0: return np.zeros((0,3))
    d = int(len(rois[0])/2)
    pts = np.zeros((n, 5))
    for i in range(0, n):
        smroi = gaussian_filter(rois[i], 3)
        pts[i,0] = xpos[i] + d
        pts[i,1] = ypos[i] + d
        pts[i,2] = np.max(smroi)
        pts[i,3] = framenumber
        pts[i,4] = n
    return pts

# Expands an interval list (scan number) to give a list of included integers
# Elements of 'intervals' can be either [a, b] for range from a to b
# or a single integer for one element
def ilexpand(intervals):
    s = np.array(())
    for i in intervals:
        if np.size(i)==1: # include single scan
            s = np.append(s, i)
        if np.size(i)==2: # include range of scans
            s = np.append(s, np.arange(i[0],i[1]))
        else:
            continue
    return s.astype(int)
        
# copy wfm files from each run to common directory
def copywfms(path, nruns):
    for i in range(1, nruns+1):
        print(i)
        file = path + "run{}/daqdata.uint16".format(i)
        copyfile(file, path + "wfm/daqdata{}.uint16".format(i))
        
# copy roi files from each run to common directory
def copyrois(path, nruns):
    for i in range(1, nruns+1):
        print(i)
        file = path + "run{}/roispos1.single".format(i)
        copyfile(file, path + "roi/rois{}.single".format(i))
        
# copy ctr files from each run to common directory
def copyctrs(path, nruns):
    for i in range(1, nruns+1):
        print(i)
        file = path + "run{}/roictrs.mat".format(i)
        copyfile(file, path + "ctr/ctrs{}.mat".format(i))