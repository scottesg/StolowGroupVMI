import numpy as np
import glob
import matplotlib.pyplot as plt
from numpy.fft import fft, ifft, rfft, rfftfreq
import scipy.ndimage as ndimage
from scipy.signal import find_peaks, butter, sosfiltfilt, freqz
from scipy.optimize import least_squares
from scipy.signal import correlate
from VMI3D_IO import readwfm, readctr, genT
from VMI3D_Fitting import res_mgauss, mgauss, fit_gauss
from scipy.ndimage import gaussian_filter, maximum_filter

J_PER_EV = 1.602176565e-19
E_MASS = 9.1093837e-31 # kg
MMUS = 1000 # mm/us to m/s
EV_PER_AU = 27.211
VFACTOR = np.sqrt((J_PER_EV)/(E_MASS*MMUS**2)) # [mm/us] = vfactor * sqrt([2*eV])
PFACTOR = np.sqrt(EV_PER_AU)*VFACTOR # [mm/us] = pfactor * [au]

# Returns all peaks from the input trace using deconvolution with the supplied response
def getpeaks(trace, response, h, snr, filt, nmax, t, ef,
             widthlower=0.5, widthupper=2, intbd=None, d=20,
             plot=False, gfit=False, gsm=0, awf=False, fixbaseline=True,
             skipdecon=False):
    
    if skipdecon:
        wd = trace
    else:
        wd = wiener_deconvolution(trace, response, snr, relativeposition=True)
        if gsm>0:
            wd = gaussian_filter(wd, gsm)
            if ef is not None:
                lef = len(ef)
                wd[:lef] *= ef
                wd[-lef:] *= ef[::-1]
        else:
            wd = fftsmooth(t, wd, filt, edgefixer=ef)[1]
    
    pk, properties = find_peaks(wd, height=h, width=widthlower)
    I = properties['peak_heights']
    fwhm = properties['widths']
    
    skip = False
    if len(pk) == 0:
        if plot: print("No hits")
        gfit = False
        skip = True
    
    if len(pk) > nmax:
        if plot: print("Too many hits: {}".format(len(pk)))
        gfit = False
        skip = True

    def pt2t(pt):
        return t[0] + pt*t[-1]/(len(trace)-1)
    
    if gfit:
        
        # Group peaks
        npk = len(pk)
        groups = []
        for i in range(0, npk):
            pki = pk[i]
            grouped = False
            for j in range(0, len(groups)):
                gl = pk[groups[j][0]] - d
                gu = pk[groups[j][-1]] + d
                if (pki > gl) and (pki < gu):
                    groups[j].append(i)
                    grouped = True
            if not grouped:
                groups.append([i])
        
        out = []
        if plot:
            plx = []
            plxopt = []
        for i in range(0, len(groups)):
            g = groups[i]
            gn = len(g)
            
            guess = [0]
            
            if fixbaseline:
                bl = [-1e-10]
                bu = [1e-10]
            else:
                bl = [-np.inf]
                bu = [np.inf]
                
            for i in range(0, gn):
                ind = g[i]
                guess.extend([I[ind], pk[ind], 1])
                bl.extend([0, pk[ind]-2, widthlower])
                bu.extend([np.inf, pk[ind]+2, widthupper])
            guess = np.array(guess)
            bounds = (np.array(bl), np.array(bu))
            
            halfd = int(d/2)
            x = np.arange(0, len(wd))
            gl = max(0, int(pk[g[0]]) - halfd)
            gu = min(len(x)-1, int(pk[g[-1]]) + halfd)
            xf = x[gl:gu]
            wdf = wd[gl:gu]
            result = least_squares(res_mgauss, x0=guess,
                                   bounds=bounds, args=(xf, wdf), method='trf')
            xopt = result.x
            
            if plot:
                plx.append(xf)
                plxopt.append(xopt)
            
            c = xopt[0]
            for i in range(0, gn):
                out.append([pt2t(xopt[2+3*i]), xopt[1+3*i] + c, xopt[3+3*i], c])
        
        out = np.array(out)
            
    elif not skip:
        c = np.zeros(len(fwhm))
        out = np.vstack((pt2t(pk), I, fwhm, c)).T
    
    if plot:
             
        if gfit:
            fig, ax = plt.subplots()
            ax.plot(x, wd, label="Deconvolution")
            for i in range(0, len(groups)):
                fit = mgauss(plx[i], plxopt[i])
                ax.plot(plx[i], fit, label="Fit Group {}".format(i))
            ax.set_xlabel("Time (ns)")
            ax.set_ylabel("Amplitude")
            ax.set_title("Deconvolution with Fit Function")
            leg = ax.legend(fontsize=20)
            leg.set_draggable(1)
            
        fig, ax = plt.subplots(3, 1) 
        ax0 = ax[0]
        ax1 = ax[1]
        ax2 = ax[2]
        
        ax0.plot(t, trace)
        ax0.set_xlabel("Time (ns)")
        ax0.set_ylabel("Amplitude")
        
        if not skip:
            ax1.plot(t, trace)
            ax1.set_xlabel("Time (ns)")
            ax1.set_ylabel("Amplitude")
            ax1.set_title("Trace With Marked Peaks [{}]".format(len(out)))
            pos = []
            for i in range(0, len(out)):
                pltpos = out[i,0]
                pos.append(pltpos)
                ax1.scatter(pltpos, max(trace), s=10, c='red')
                ax1.vlines(pltpos, min(trace), max(trace), lw=2, color='red')
            tmin = min(pos) - 20
            tmax = max(pos) + 20
            ax1.set_xlim(tmin, tmax)
        
        ax2.plot(t, wd)
        ax2.hlines(h, t[0], t[-1], color='black')
        ax2.set_title("Deconvolution with Threshold")
        ax2.set_xlabel("Time (ns)")
        ax2.set_ylabel("Deconvolution Amplitude")
        ax2.set_xlim(tmin, tmax)
        
        if not skip:
            for i in range(0, len(out)):
                pltpos = out[i,0]
                ax2.scatter(pltpos, max(wd), s=12, c='red')
                ax2.vlines(pltpos, min(wd), max(wd), lw=2, color='red')
    
    if skip: return None
    return out, t, trace, wd

# roidim currently semi-hardocded
def extractrois(img, smth, size, threshold, maxrois, roidim):
    
    dim = len(img)
    
    # Smooth and subtract projections
    imgs = gaussian_filter(img.astype(float), smth)
    pjx = np.mean(imgs,0)
    imgs = imgs - pjx
    pjy = np.mean(imgs,1)
    imgs = imgs - pjy[:,None]

    imgsmx = maximum_filter(imgs, size)
    maskmx = (imgs == imgsmx)
    maskth = (imgsmx > threshold)
    maskmx[~maskth] = False
    
    labeled, num = ndimage.label(maskmx)
    slices = ndimage.find_objects(labeled)
    
    rois = np.zeros((maxrois, roidim, roidim), dtype=np.single)
    xpos = np.zeros(maxrois, dtype=np.uint16)
    ypos = np.zeros(maxrois, dtype=np.uint16)
    
    nrois = 0
    for dx,dy in slices:
        x_center = int((dx.start + dx.stop - 1)/2)
        y_center = int((dy.start + dy.stop - 1)/2)
        
        if (x_center<15) or (x_center>dim-16) or (y_center<15) or (y_center>dim-16):
            continue
        
        xpos[nrois] = x_center
        ypos[nrois] = y_center
        rois[nrois] = img[x_center-15:x_center+16, y_center-15:y_center+16]
        nrois += 1
        
        if nrois >= (maxrois-1): break
    
    return xpos, ypos, rois, nrois, imgs

def timecorrect(path, bk, rpr, rpos, sgn, snr, dfitpt, sm, tracesize, ch2=False, debug=False):

    nf = len(glob.glob(path))    
    corrections = []
    amps = []
    widths = []
    
    if debug: dbi = 0
    for i in range(0, nf):
        print("Stack #{} / {}".format(i+1, nf))
        wfm = readwfm(path.replace('*','{}').format(i+1), dqdim=tracesize, ch2=ch2)
        if ch2: wfm = wfm[1]
        cori = np.zeros(len(wfm))
        ampi = np.zeros(len(wfm))
        widi = np.zeros(len(wfm))
        for j in range(0, len(wfm)):
            if j % 10000 == 0: print("Trace #{} / {}".format(j, len(wfm)))
            
            if ch2: wfmj = sgn*(wfm[j] - np.mean(wfm[j][:500]))
            else: wfmj = sgn*(wfm[j] - bk)
            
            wd = wiener_deconvolution(wfmj, rpr, snr, relativeposition=True)
            wd = gaussian_filter(wd, sm)
            wdr = wd[rpos-dfitpt:rpos+dfitpt]
            mx = max(wdr)
            mxp = np.argmax(wdr) + rpos - dfitpt
            wdr = wd[mxp-dfitpt:mxp+dfitpt]
            fit = fit_gauss(np.arange(mxp-dfitpt, mxp+dfitpt, 1),
                            wdr, 0, mx, mxp, 3, czero=True)
            if fit == -1:
                print(j)
                cori[j] = -1
                ampi[j] = -1
            else:
                cori[j] = fit[1][1]
                ampi[j] = fit[1][0]
                widi[j] = fit[1][2]
            
            if debug:
                if fit[1][2] > 0.415:
                    dbi = dbi+1
                    plt.figure()
                    plt.title(fit[1][1])
                    plt.plot(np.arange(0, len(wd), 1),  wd)
                    plt.plot(np.arange(0, len(wfmj), 1),  wfmj/80000)
                    plt.plot(np.arange(mxp-dfitpt, mxp+dfitpt, 1), fit[0])
                    plt.xlim([rpos-dfitpt, rpos+dfitpt])
                    if dbi>=20: return [0,0,0]
                
        corrections.append(cori)
        amps.append(ampi)
        widths.append(widi)
        
    return np.hstack(corrections), np.hstack(amps), np.hstack(widths)

# Get number of peaks in trace using deconvolution
def getpeaknum(trace, response, h, snr, gsm, blpts, ef,
             widthlower=0.5):
    
    bl = np.mean(trace[:blpts])
    wd = wiener_deconvolution(trace-bl, response, snr, relativeposition=True)
    wd = gaussian_filter(wd, gsm)
    if ef is not None:
        lef = len(ef)
        wd[:lef] *= ef
        wd[-lef:] *= ef[::-1]
    
    pk, properties = find_peaks(wd, height=h, width=widthlower)
    
    return len(pk)

# Performs a wiener deconvolution of the signal with response
def wiener_deconvolution(signal, response, snr, relativeposition=False):
    
    if relativeposition:
        responsepos = find_peaks(response, height=max(response)/2)[0][0]
    else: responsepos = 0
        
    response = np.hstack((response, np.zeros(len(signal) - len(response))))
    H = fft(response)
    T = 1 / (H*np.conj(H)*snr)
    G = (1/H) * (1/(1+T))
    wd = np.real(ifft(fft(signal) * G))
    wd = np.hstack((wd[-responsepos:],wd[:-responsepos]))
    return wd

# Rough but faster version of getpeaks
def fastpeaks(trace, nsd, blshift, smth, bk=0, plot=False):

    bl = np.mean(trace[:blshift])
    traceshifted = trace - bl
    traceshifted = traceshifted - bk
    
    tracesm = ndimage.gaussian_filter(traceshifted, smth)
    h = nsd * np.std(tracesm[:blshift])
    peaks = find_peaks(tracesm, height=h)
    out = np.vstack((peaks[0], peaks[1]['peak_heights'])).T
    
    if plot:
        t = np.arange(0, len(trace))
        fig, ax = plt.subplots()
        ax.plot(t, tracesm)
        plt.hlines(h, t[0], t[-1])
        ax.set_title("Smoothed Trace with Threshold")
        ax.set_xlabel("Time")
        ax.set_ylabel("Amplitude")
        
    return out

# Perform MCP gain spatial correction using xy grid
def xydetmap(points, gsize, tilelim, xrange, yrange, plot=True):
    
    var = np.zeros((gsize, gsize))
    num_map = np.zeros((gsize, gsize))
    binpoints = np.zeros((0, len(points[0])))

    def f(x, t, y):
        return x[1] + x[0]*t - y 
    x0 = np.array([1, 0])
    
    c1 = points[:,0]>xrange[0]
    c2 = points[:,0]<xrange[1]
    c3 = points[:,1]>yrange[0]
    c4 = points[:,1]<xrange[1]
    
    c = c1 & c2 & c3 & c4
    
    points = points[c]
    points = points[points[:,0].argsort()]
    
    xval = np.linspace(xrange[0], xrange[1], gsize+1)
    yval = np.linspace(yrange[0], yrange[1], gsize+1)
    
    indx = np.searchsorted(points[:,0], xval)
    for i in range(0, gsize):
        subpoints = points[indx[i]:indx[i+1]]
        subpoints = subpoints[subpoints[:,1].argsort()]
        indy = np.searchsorted(subpoints[:,1], yval)
        for j in range(0, gsize):
            tilepoints = subpoints[indy[j]:indy[j+1]]
            num_map[i, j] = len(tilepoints)
            if (len(tilepoints) > tilelim):
                re_lsq = least_squares(f, x0, loss='cauchy', f_scale=0.01, args=(tilepoints[:,3], tilepoints[:,4]))
                var[i, j] = re_lsq.x[0]
                tilepoints[:,3] = tilepoints[:,3] * re_lsq.x[0]
                binpoints = np.vstack((binpoints, tilepoints))

    if plot:
        plt.figure()
        plt.imshow(var)
        ax = plt.gca()
        ax.set_title("Detector Variation Map (Raw)")
        ax.axes.get_xaxis().set_visible(False)
        ax.axes.get_yaxis().set_visible(False)
        plt.colorbar()
            
        fig, ax = plt.subplots()
        ax.imshow(num_map);
        ax.set_title("Hits by Spatial Bin")
        ax.axes.get_xaxis().set_visible(False)
        ax.axes.get_yaxis().set_visible(False)
    
    detmap = {'var': var, 'xval': xval, 'yval': yval}
    
    return binpoints, detmap, num_map

# Perform MCP gain spatial correction using radial bins
def rdetmap(points, nbin, centre, rval, tilelim, plot=True, debug=False):
    
    if np.size(rval)==1:
        rval = np.linspace(0, rval, nbin+1)
    else:
        nbin = len(rval)-1
        
    var = np.zeros(nbin)
    num_map = np.zeros(nbin)
    binpoints = np.zeros((0,7))
    
    r = np.sqrt((points[:,0]-centre[0])**2 + (points[:,1]-centre[1])**2)
    r = np.reshape(r, (len(r), 1))
    points = np.hstack((points, r))

    def f(x, t, y):
        return x[1] + x[0]*t - y 
    x0 = np.array([1, 0])
    
    points = points[points[:,0]>0]
    points = points[points[:,1]>0]
    points = points[points[:,-1].argsort()]
    
    indx = np.searchsorted(points[:,-1], rval)
    for i in range(0, nbin):
        tilepoints = points[indx[i]:indx[i+1]]
        num_map[i] = len(tilepoints)
        if (len(tilepoints) > tilelim):
            re_lsq = least_squares(f, x0, loss='cauchy', f_scale=0.001, args=(tilepoints[:,3], tilepoints[:,4]))
            
            if debug:
                slope = re_lsq.x[0]
                yint = re_lsq.x[1]
                fig, ax = plt.subplots()
                xul = max(tilepoints[:,3])
                ax.hist2d(tilepoints[:,3],tilepoints[:,4], 100);
                ax.plot([0, xul], [yint, yint+slope*xul], color='r')
                ax.set_title("Range {}:{}".format(rval[i], rval[i+1]))
            
            var[i] = re_lsq.x[0]
            tilepoints[:,3] = tilepoints[:,3] * re_lsq.x[0]
            binpoints = np.vstack((binpoints, tilepoints[:,:-1]))
                
    if plot:
        fig, ax = plt.subplots()
        ax.plot(var);
        ax.set_title("Slope by Radial Bin")
            
        fig, ax = plt.subplots()
        ax.plot(num_map);
        ax.set_title("Hits by Radial Bin")
    
    detmap = {'var': var, 'rval': rval}
    
    return binpoints, detmap, num_map

# Correct points using pre-existing spatial correction
def xycorrectpoints(points, detmap, ctrs=False, inverse=False):
    var = detmap['var']
    xval = detmap['xval']
    yval = detmap['yval']
    gsize = len(xval)-1
    corrected = []
    
    ind = 3
    if ctrs: ind = 2
    
    points = points[points[:,0].argsort()]
    indx = np.searchsorted(points[:,0], xval)
    for i in range(0, gsize):
        subpoints = points[indx[i]:indx[i+1]]
        subpoints = subpoints[subpoints[:,1].argsort()]
        indy = np.searchsorted(subpoints[:,1], yval)
        for j in range(0, gsize):
            if var[i,j] == 0: continue
            tilepoints = subpoints[indy[j]:indy[j+1]]
            if inverse:
                tilepoints[:,ind] = tilepoints[:,ind] * (1/var[i,j])
            else:
                tilepoints[:,ind] = tilepoints[:,ind] * var[i,j]
            corrected.extend(tilepoints)
    corrected = np.array(corrected)
    return corrected
      
# Bin points by arrival time, for use in movie
def timebin(points, nbin, tmin, tmax, discrete=False):
    
    points = points[points[:,2].argsort()]
    t = points[:,2]
    tmin = max(tmin, t[0])
    tmax = min(tmax, t[-1])
    
    ptr = points[points[:,2]>tmin]
    ptr = ptr[ptr[:,2]<tmax]
    t = ptr[:,2]
    
    if discrete:
        d = np.diff(np.unique(t)).min() * nbin
        ul = max(t) + d/2
        ll = min(t) - d/2
        bins = np.arange(ll, ul + d, d)
        
    else:
        ul = max(t)
        ll = min(t)
        bins = np.linspace(ll, ul, nbin)
        d = bins[1]-bins[0]

    numbin = len(bins) - 1
    
    inds = np.searchsorted(t, bins)
    bindata = list()
    for i in range(0, numbin):
        ibin = ptr[inds[i]:inds[i+1]]
        bindata.append(ibin)
    return bindata, bins[:-1]+d/2

# Converts the threshold as a fraction of the wd of the response with itself
# (including smoothing) to a non-relative value
def thresholdfromPR(response, freq, order, snr, fraction, dt, gsm=0):
    wd = wiener_deconvolution(response, response, snr)
    if gsm>0:
        wd = gaussian_filter(wd, gsm)
    else:
        wd = fftsmooth(genT(len(wd), dt), wd, freq, order)[1]
    h = max(wd) * fraction
    return h

# Generate trace background
def genBK(path, h, blpts, maxtraces=-1, dqdim=2048, ch2=False):
    
    data = readwfm(path, dqdim=dqdim, ch2=ch2)
    if ch2: data = data[0]
    
    if maxtraces==-1:
        maxtraces = len(data)
    
    bk = np.zeros(len(data[0]))
    c = 0
    for i in range(0, min(len(data), maxtraces)):
        if i % 1000 == 0: print("Trace #{} / {} ({})".format(i, len(data), c))
        tmp = data[i]
        if max(np.abs(tmp - np.mean(tmp[:blpts]))) < h:
            bk += tmp
            c += 1
    bk /= c
    print(c)
    return bk

# Generate single-hit response from pickoff data
def genPR(path, dqdim, bk, sgn, h1, h2, h3, nppr, smth,
          tc, rpr, f1, f2, fw, ntrace=-1, maxtraces=-1, loadfile=0, 
          ch2=False, simple=True, debug=False, shreturn=False):
    
    if loadfile>0:
        path = path.replace('*',str(loadfile))
    
    if type(path) is str:
        data = readwfm(path, dqdim=dqdim, ch2=ch2)
        if ch2: data = data[0]
    else: data = path
    
    if ch2:
        tc = None
        rpr = None
    
    if maxtraces==-1:
        maxtraces = len(data)
    
    if simple:
        inds = genPRinds_Simple(data, sgn, bk, tc, rpr, smth, h1, h2, h3, nppr, maxtraces, ntrace)
    else:
        inds = genPRinds(data, sgn, bk, tc, rpr, smth, h1, h2, f1, f2, nppr, maxtraces, ntrace)

    c = len(inds)
    pr = np.zeros(len(data[0]))
    shtraces = [] # for debugging
    for i in range(len(inds)):
        
        ind = inds[i]
        
        tmp = sgn * (data[ind] - bk)
    
        if tc is not None:
            tci = tc[ind]
            tmp = tmp - ndimage.shift(rpr, tci-1000, mode='wrap', order=3)
        
        peaks = find_peaks(tmp, 0.99*np.max(tmp))
        
        if len(peaks[0])==0:
            continue
        
        posr = peaks[0][0]
        
        xfit = np.arange(-1*(fw-1), fw)
        yfit = tmp[posr-1*(fw-1):posr+fw]
        
        if len(xfit)!=len(yfit):
            continue
        
        fit = fit_gauss(xfit, yfit, 0, np.max(yfit), 0, 4)
        
        if debug:
            plt.figure()
            plt.plot(tmp)
        
        if fit == -1:
            print(ind)
            continue
        
        if debug:
            plt.plot(xfit+posr, fit[0])
            plt.vlines(fit[1][2] + posr, 0, np.max(tmp))
            if i>5: return -1
        
        pos = fit[1][2] + posr
        pri = ndimage.shift(tmp, 1000-pos, mode='wrap', order=3)/c
        pr += pri
        shtraces.append(pri)
            
    print("Number of Samples: {}/{}".format(c, len(data)))
    if shreturn: return pr, shtraces
    else: return pr

def genPRinds(wfms, sgn, bk, tc, rpr, smth, h1, h2, f1, f2, nppr, maxtraces, ntrace):
    
    inds = []
    c = 0
    for i in range(0, min(len(wfms), maxtraces)):
        
        if c == ntrace: break
    
        if i % 1000 == 0: print("Trace #{} / {} ({})".format(i, len(wfms), c))
        
        tmp = sgn * (wfms[i] - bk)
        
        if tc is not None:
            tci = tc[i]
            tmp = tmp - ndimage.shift(rpr, tci-1000, mode='wrap', order=3)
        
        tmps = gaussian_filter(tmp, 2)
        peaks11 = find_peaks(tmps, h1)
        peaks12 = find_peaks(tmps, h2)
        
        if (len(peaks11[0])==nppr) and (len(peaks12[0])==1):
    
            pos1 = peaks12[0][0]
            height1 = peaks12[1]['peak_heights'][0]
            
            peaks2 = find_peaks(-1*tmps, f1*height1)
            peaks3 = find_peaks(tmps, f2*height1)
            if len(peaks2[0])==1 and len(peaks3[0])==2:
                # since there is a positive and negative secondary peak for each hit
                
                pos2 = peaks2[0][0]
                pos3 = peaks3[0][1]
                height2 = peaks2[1]['peak_heights'][0]
                height3 = peaks3[1]['peak_heights'][1]
                
                if (height1 > height2) and (height2 > height3):
                    if (pos1 < pos2) and (pos2 < pos3):
                        inds.append(i)
                        c = c + 1
    return inds

def genPRinds_Simple(wfms, sgn, bk, tc, rpr, smth, h1, h2, h3, nppr, maxtraces,
                     ntrace, debug=False):
    
    inds = []
    c = 0
    for i in range(0, min(len(wfms), maxtraces)):
        if c == ntrace: break
    
        if i % 1000 == 0: print("Trace #{} / {} ({})".format(i, len(wfms), c))
        
        tmp = sgn * (wfms[i] - bk)
        
        if tc is not None:
            tci = tc[i]
            tmp = tmp - ndimage.shift(rpr, tci-1000, mode='wrap', order=3)
        
        tmps = gaussian_filter(tmp, smth)
        peaks1 = find_peaks(tmps, h1)
        peaks2 = find_peaks(tmps, h2)
        peaks3 = find_peaks(tmps, h3)
        
        if debug:
            plt.plot(tmps)
            plt.hlines([h1, h2, h3], 0, len(tmps))
            if i > 5: break
        
        if (len(peaks1[0])==nppr) and (len(peaks2[0])==1) and (len(peaks3[0])==0):
            inds.append(i)
            c = c + 1
    
    return inds

# def genRPR(path, bk, h, nprpr, fitpts, loadfile=0, maxtraces=-1, ntrace=-1,
#            sgn=-1, dqdim=2048, ch2=False, debug=False):
    
#     if loadfile>0:
#         path = path.replace('*',str(loadfile))
    
#     if type(path) is str:
#         data = readwfm(path, dqdim=dqdim, ch2=ch2)
#         if ch2: data = data[1]
#     else: data = path
    
#     if ch2: nprpr = 1
    
#     response = np.zeros(len(data[0]))
#     if maxtraces==-1:
#         maxtraces = len(data)
    
#     c = 0
#     for i in range(0, min(len(data), maxtraces)):
#         if c == ntrace: break
#         if i % 1000 == 0: print("Trace #{} / {} ({})".format(i, len(data), c))
#         if ch2:
#             tmp = sgn * (data[i] - np.mean(data[i][:500]))
#             peaks = find_peaks(tmp, 0.99*np.max(tmp))
#         else:
#             tmp = sgn * (data[i] - bk)
#             peaks = find_peaks(tmp, h)
        
#         if not len(peaks[0])==nprpr: continue
    
#         posr = peaks[0][0]
#         xfit = np.arange(-1*(fitpts-1), fitpts)
#         yfit = tmp[posr-1*(fitpts-1):posr+fitpts]
        
#         fit = fit_gauss(xfit, yfit, 0, np.max(yfit), 0, 4, czero=True)
        
#         if debug:
#             plt.figure()
#             plt.plot(tmp)
        
#         if fit == -1:
#             continue
        
#         if debug:
#             plt.plot(xfit+posr, fit[0])
#             plt.vlines(fit[1][1] + posr, 0, np.max(tmp))
#             if i>5: return -1
        
#         pos = fit[1][1] + posr
#         response += ndimage.shift(tmp, 1000-pos, mode='wrap', order=3)
#         c = c + 1
            
#     print("Number of Samples: {}/{}".format(c, len(data)))
#     if (c==0) or (c < ntrace):
#         return np.zeros(len(response))
    
#     response = response / c
#     return response

def genRPR(path, bk, h, nprpr, fitpts, loadfile=0, maxtraces=-1, ntrace=-1,
           sgn=-1, dqdim=2048, ch2=False, debug=False):
    
    if loadfile>0:
        path = path.replace('*',str(loadfile))
    
    if type(path) is str:
        data = readwfm(path, dqdim=dqdim, ch2=ch2)
        if ch2: data = data[1]
    else: data = path

    response = np.zeros(len(data[0]))
    if maxtraces==-1:
        maxtraces = len(data)
    
    c = 0
    for i in range(0, min(len(data), maxtraces)):
        if c == ntrace: break
        if i % 1000 == 0: print("Trace #{} / {} ({})".format(i, len(data), c))
        if ch2:
            tmp = sgn * (data[i] - np.mean(data[i][:500]))
        else:
            tmp = sgn * (data[i] - bk)
    
        posr = np.argwhere(tmp >= np.max(tmp)/2)[0][0]
        #print(posr)
        xfit = np.arange(-1*(fitpts-1), fitpts)
        yfit = tmp[posr-1*(fitpts-1):posr+fitpts]
        
        fit = fit_gauss(xfit, yfit, 0, np.max(yfit), 0, 4, czero=True)
        
        if debug:
            plt.figure()
            plt.plot(tmp)
        
        if fit == -1:
            continue
        
        if debug:
            plt.plot(xfit+posr, fit[0])
            plt.vlines(fit[1][1] + posr, 0, np.max(tmp))
            if i>5: return -1
        
        pos = fit[1][1] + posr
        response += ndimage.shift(tmp, 1000-pos, mode='wrap', order=3)
        c = c + 1
            
    print("Number of Samples: {}/{}".format(c, len(data)))
    if (c==0) or (c < ntrace):
        return np.zeros(len(response))
    
    response = response / c
    return response


# Determine camera/pickoff offsets within each stack
def genOffsets(ctrpath, hitspath, stksize, wdsnr, atypehits=False, plot=False):
    
    n = len(glob.glob(ctrpath))
    
    ctrs = readctr(ctrpath, stksize=stksize)
    ngrs = getngrs(ctrs, stksize, n)
    
    if atypehits:
        nhits = getnhits_atype(hitspath, stksize, n)
    else:
        hits = np.load(hitspath)
        nhits = getnhits(hits, stksize, n)
    
    offsets = np.zeros(n)
    overlaps = np.zeros(n)

    for i in range(0, n):
        print("Processing stack {}/{}...".format(i+1, n))
    

        wd = wiener_deconvolution(nhits[i], ngrs[i], wdsnr)
        offset = np.argmax(np.abs(wd))
        overlap = np.abs(np.max(wd)/np.std(wd[offset+1:]))
        offsets[i] = offset
        overlaps[i] = overlap
        
        if plot:
            plt.figure()
            plt.plot(wd)
    
    return np.vstack((offsets, overlaps))

def getnhits_atype(hitspath, stksize, n):
    nhits = np.zeros((n, stksize))
    for i in range(0, n):
        hits = np.loadtxt(hitspath.format(i+1))
        nhits[i] = hits[:,0]
    nhits[np.isnan(nhits)] = 0
    nhits[nhits==99] = 0
    
    return nhits

# Offsets using unprocessed traces (rough)
def genOffsetsFromTraces(ctrpath, wfmpath, responsepath,
               blpts, threshold, smth, snr, stksize, ef, sgn=-1, dqdim=2048):
    
    wdsnr = 500
    ctrs = readctr(ctrpath)
    wfms = readwfm(wfmpath, dqdim=dqdim)
    response = np.load(responsepath)
    
    n = int(len(wfms)/stksize)
    ngrs = getngrs(ctrs, stksize, n)
    nhits = np.zeros((n, stksize))
    
    offsets = np.zeros(n)
    overlaps = np.zeros(n)

    for i in range(0, 10):
        print("Processing stack {}/{}...".format(i, n))

        for j in range(0, stksize):
            nhits[i,j] = getpeaknum(sgn*wfms[j+i*stksize], response, threshold, snr, smth, blpts, ef)
    
        wd = wiener_deconvolution(nhits[i], ngrs[i], wdsnr)
        offset = np.argmax(wd)
        overlap = np.round(np.abs(np.max(wd)/np.mean(wd[offset+1:])))
        offsets[i] = offset
        overlaps[i] = overlap
    
    return nhits, ngrs, np.vstack((offsets, overlaps))

# Extracts the number of hits per image from the internal format centroids
def getngrs(ctrs, stksize, nstacks):
    n = stksize*nstacks
    ngrs = np.zeros(int(n))
    stknum = ctrs[:,3].astype(int)
    for i in range(0, len(ctrs)):
        ngrs[stknum[i]] = ctrs[i,4]
    ngrs = np.resize(ngrs, (nstacks, stksize))
    return ngrs

# Extracts the number of hits per trace from the internal format hits
def getnhits(hits, stksize, nstacks):
    nhits = np.zeros((nstacks, stksize))
    for i in range(0, len(hits)):
        stk = int(hits[i, 4])
        shot = int(hits[i, 5])
        num = int(hits[i, 6])
        nhits[stk, shot] = num
    return nhits

# Returns function y(x) oversampled by a factor of 'factor'
def oversample(x, y, factor, smooth=True):
    newlen = (len(x)-1)*factor+1
    newx = np.linspace(x[0], x[-1], newlen)
    newy = np.interp(newx, x, y)
    if smooth:
        newy = ndimage.gaussian_filter(newy, int(factor/2))
    return newy

# VMI image from points data
def pts2img(pts, dim, smooth):
    
    # Definitions
    img = np.zeros((dim, dim))
    x = pts[:,0]
    y = pts[:,1]

    # Remove points outside image range
    cond = (x>0) & (x<dim) & (y>0) & (y<dim)
    x = x[cond]
    y = y[cond]

    # Count points in each pixel
    for i in range(0, len(x)):
        xi = int(x[i])
        yi = int(y[i])
        img[xi,yi] = img[xi,yi] + 1
    
    img = ndimage.gaussian_filter(img, smooth)
    return img

# Performs Butterworth filter on data (x, y)
# Filt: Either the filter to be used (as numpy array)
#   or a list [frequency, order] to generate the filter
# edgefixer: an array to multiply both ends of the output to supress edge effects
#   defaults to an array of zeros
def fftsmooth(x, y, filt, edgefixer=np.zeros(20), plot=False):
    
    # Create Butterworth Filter
    if type(filt) is list:
        f = filt[0]
        order = filt[1]
        sr = 1/(x[1]-x[0])
        bfilter = butter(order, f, 'low', analog=False, output='sos', fs=sr)     
    else:
        bfilter = filt
        
    y_butter = sosfiltfilt(bfilter, y)
    
    if edgefixer is not None:
        lef = len(edgefixer)
        y_butter[:lef] *= edgefixer
        y_butter[-lef:] *= edgefixer[::-1]
    
    if plot:
        
        plt.figure()
        plt.plot(x, y)
        plt.title("Unfiltered Data")
        plt.xlabel("Time (ns)")
        
        plt.figure()
        plt.plot(x, y_butter)
        plt.title("Data After Filter")
        plt.xlabel("Time (ns)")
        
        if type(filt) is list:
            
            b, a = butter(order, f, 'low', analog=False, fs=sr)
            w, h = freqz(b, a, fs=sr)
            
            N = len(x)
            yf = rfft(y)
            xf = rfftfreq(N, 1/sr)
            yf_butter = rfft(y_butter)
    
            plt.figure()
            plt.plot(xf, np.abs(yf))
            plt.title("FFT")
            plt.xlabel("Frequency (GHz)")
        
            plt.figure()
            plt.plot(w, np.abs(h))
            plt.title("Butterworth Filter at {} GHz (order {})".format(f, order))
            plt.xlabel("Frequency (GHz)")
            plt.ylabel("Filter Amplitude")
            
            plt.figure()
            plt.plot(xf, np.abs(yf_butter))
            plt.title("FFT of Filtered Data")
            plt.xlabel("Frequency (GHz)")
        
            plt.figure()
            plt.plot(x*1e9, np.abs(y_butter), label="Filtered")
            plt.plot(x*1e9, np.abs(y), label = "Unfiltered")
            plt.title("Filtered and Unfiltered Data (Butterworth at {} GHz, order {})".format(f, order))
            plt.xlabel("Frequency (GHz)")
        
    return [x, y_butter]

# Calculate distance between point and line
def pldist(point, slope, yint):
    x, y = point
    return (yint + slope*x - y)/np.sqrt(1 + slope**2)
    
# Fit the amplitude correlation to a line
# and get gaussian width of correlation along the line
def fitcorr(points, path, sbin, fscale=0.001, plot=True, save=True):
    
    # Linear Fit
    N = len(points)
    x0 = np.array([1, 0])
    def f(x, t, y):
            return x[1] + x[0]*t - y 
    
    lsq = least_squares(f, x0, loss='cauchy', f_scale=fscale,
                        args=(points[:,3], points[:,4]))
    
    slope, yint = lsq.x
    resid = lsq.fun
    yvar = (1/(N-2)) * np.sum(resid**2)
    
    if plot:
        ci = points[:,3]
        pi = points[:,4]
    
        fig, ax = plt.subplots()
        hist = ax.hist2d(ci, pi, 500, cmap='gray_r')
        plt.colorbar(hist[3], ax=ax)
        xul = np.percentile(ci, 99)
        ax.set_xlim((np.percentile(ci, 1), xul))
        ax.set_ylim((np.percentile(pi, 1), np.percentile(pi, 99)))
        ax.set_title("Correlation Plot")
        ax.set_ylabel("Pickoff Amplitude")
        ax.set_xlabel("Camera Intensity")
        plt.plot([0, xul], [yint, yint+slope*xul], color='r')
        plt.savefig(path + "/CorrelationFitLine.png")
    
    # Find width of distribution of points from fit line
    islope = -1/slope
    ds = np.zeros(N)
    bs = np.zeros(N)
    for i in range(0, N):
        p = points[i]
        ci = p[3]
        si = p[4]
        d = pldist([ci, si], slope, yint)
        b = np.abs(pldist([ci, si], islope, yint))
        bs[i] = b
        ds[i] = d
    sort = np.argsort(bs)
    ds = ds[sort]
    bs = bs[sort]
    
    # Bin along fit line
    bll = np.percentile(bs, 0.5)
    bul = np.percentile(bs, 98)
    if plot:
        plt.figure()
        x, y = histxy(bs, [0.8*bll, 1.2*bul], 200)
        plt.plot(x, y)
        plt.vlines([bll, bul], 0, max(y))
    
    corfit = np.zeros((sbin+1, 5))
    
    # Fit each bin
    bins = np.linspace(bll, bul, sbin+1)
    inds = np.searchsorted(bs, bins)
    binlen = np.zeros(sbin)
    for i in range(0, sbin):
        ibin = ds[inds[i]:inds[i+1]]
        binlen[i] = len(ibin)
        x, y = histxy(ibin, [-0.05, 0.05], 200)
        fit = fit_gauss(x, y, 0, max(y), 0, 0.002)
        perr = np.sqrt(np.diag(fit[2]))
        c, h, pos, w = fit[1]
        w = np.abs(w)
        dw = perr[2]
        corfit[i] = [h, pos, w, c, dw]
        if plot:
            plt.figure()
            plt.plot(x, y)
            plt.plot(x, fit[0])
            plt.title("Bin {} ({}), width: {}".format(i+1, len(ibin), w))
        
    # Fit all bins
    x, y = histxy(ds, [-0.05, 0.05], 200)
    fit = fit_gauss(x, y, 0, max(y), 0, 0.002)
    perr = np.sqrt(np.diag(fit[2]))
    c, h, pos, w = fit[1]
    w = np.abs(w)
    dw = perr[2]
    corfit[-1] = [h, pos, w, c, dw]
    if plot:
        plt.figure()
        plt.plot(x, y)
        plt.plot(x, fit[0])
        plt.title("All Bins, width: {}".format(w))

    if plot:
        plt.figure()
        ws = corfit[:-1, 2]
        plt.plot(ws/max(ws), label="Width of Distribution", lw=2, marker='o')
        #plt.plot(binlen/max(binlen), label="Population of Bin", lw=2, marker='o')
        plt.xlabel("Bin Number", fontsize=20)
        plt.ylabel("Normalized Width", fontsize=20)
        plt.tick_params(labelsize=20, width=2, length=5)
        #plt.title("Variation of Correlation Width Along Fit Line", fontsize=25)
        #leg = plt.legend(fontsize=20)
        #leg.set_draggable(True)
    
    corfit = np.hstack((corfit, np.reshape(bins, (len(bins), 1))))
    corfit = np.vstack((corfit, [slope, yint, yvar, sbin, 0, 0]))
    if save:
        np.save(path + "/corfit.npy", np.array(corfit))
    return corfit

# Histogram of data for plotting
def histxy(data, xrange=None, nbin=0):
    
    # Trim to range
    if xrange is None:
        xmin = min(data)
        xmax = max(data)
    else:
        xmin, xmax = xrange
        data = data[data<=xmax]
        data = data[data>=xmin]
    
    # Determine optimal bins
    if nbin==0:
        d = np.diff(np.unique(data)).min()
        ul = max(data) + d/2
        ll = min(data) - d/2
        bins = np.arange(ll, ul + d, d)
    else:
        bins = np.linspace(xmin, xmax, nbin)
        d = bins[1] - bins[0]
    
    # Histogram
    y, bins = np.histogram(data, bins)
    x = bins[:-1] + d/2
    return [x, y]

# 2D histogram of data for plotting
def histxy2d(datax, datay, bins):
    n, bx, by = np.histogram2d(datax, datay, bins, density=False)
    dx = bx[1]-bx[0]
    dy = by[1]-by[0]
    x = bx[:-1] + dx/2
    y = by[:-1] + dy/2
    return [x, y, n]

# Cross-correlation
def ccornorm(a, b):
    a = (a - np.mean(a)) / (np.std(a) * len(a))
    b = (b - np.mean(b)) /  np.std(b)
    return correlate(a, b)

# Generate pseudo-data from single hit amplitude distribution
# Gives 'nhits' per shot, 'nsam' shots
def shPseudo(shdata, nhits, nsam, shadsmooth=5, hist2dbins=1000,
             cutoffpercentile=99.8, plot=False):
    
    # Cut off weak tail of data at high charge
    shdata = shdata[shdata[:,3]<np.percentile(shdata[:,3], cutoffpercentile)]
    shdata = shdata[shdata[:,4]<np.percentile(shdata[:,4], cutoffpercentile)]
    
    # Generate 2d histogram of single hit amplitudes
    ci = shdata[:,3]
    pi = shdata[:,4]
    shad, x, y = np.histogram2d(ci, pi, hist2dbins)
    dx = x[1] - x[0]
    dy = y[1] - y[0]
    
    if plot:
        plt.figure()
        plt.imshow(shad, cmap='nipy_spectral_r')
    
    # Smooth the shad
    shad = gaussian_filter(shad, shadsmooth)
    
    if plot:
        plt.figure()
        plt.imshow(shad, cmap='nipy_spectral_r')
    
    # Normalize shad so it is a proper probibility distribution
    shad = shad / np.sum(shad)

    # Sample the shad to produce nsam shots with nhits each     
    # Create a 1d copy of shad
    flat = shad.flatten()

    # Sample the 1D shad using itself as the probability distribution
    ind1d = np.random.choice(flat.size, size=nhits*nsam, p=flat)
    
    # Find 2d index
    ind2d = np.unravel_index(ind1d, shad.shape)
    
    # Random offsets for the ci and pi values in each shad bin
    rndx = np.random.rand(len(ind2d[0]))*dx
    rndy = np.random.rand(len(ind2d[1]))*dy

    # Convert to ci and pi values
    ind2d = [x[ind2d[0]] + rndx, y[ind2d[1]] + rndy]
    
    if plot:
        ishad = np.histogram2d(ind2d[0], ind2d[1], hist2dbins)[0]
        plt.figure()
        plt.imshow(gaussian_filter(ishad, shadsmooth), cmap='nipy_spectral_r')
    
    ind2d = list(zip(*ind2d))
    data = [ind2d[i:i+nhits] for i in range(0, len(ind2d), nhits)]
    return data, [shad, x, y]

def genshad(points, path, nbinshad, plot=True, smooth=0, cutoffpercentile=99.8):
    
    # Cut off weak tail of data at high charge
    points = points[points[:,3]<np.percentile(points[:,3], cutoffpercentile)]
    points = points[points[:,4]<np.percentile(points[:,4], cutoffpercentile)]
    
    ci = points[:,3]
    pi = points[:,4]
    shad, x, y = np.histogram2d(ci, pi, nbinshad)
    shad = gaussian_filter(shad, smooth)
    
    if plot:
        plt.figure()
        plt.imshow(shad, cmap='nipy_spectral_r')
    
    shad = [shad, x, y]
    np.save(path + "/shad.npy", np.array(shad, dtype=object))
    return shad

# Bin points using xy grid
def xybin(points, nx, ny, dim):
    binpoints = list()    
    points = points[points[:,0].argsort()]
    
    xval = np.linspace(0, dim, nx+1)
    yval = np.linspace(0, dim, ny+1)
    
    indx = np.searchsorted(points[:,0], xval)
    for i in range(0, nx):
        subpoints = points[indx[i]:indx[i+1]]
        subpoints = subpoints[subpoints[:,1].argsort()]
        indy = np.searchsorted(subpoints[:,1], yval)
        for j in range(0, ny):
            tilepoints = subpoints[indy[j]:indy[j+1]]
            binpoints.append(tilepoints)
    return binpoints

# Bin points using radial bins
def rbin(points, nr, dim, cen):
    binpoints = list()    
    rmax = int(dim/2)
    
    r = np.zeros((len(points), 1))
    r[:,0] = np.sqrt((points[:,0]-cen[0])**2 + (points[:,1]-cen[1])**2)
    points = np.hstack((points,r))
    points = points[points[:,-1].argsort()]
    
    rval = np.linspace(0, rmax, nr+1)
    
    indx = np.searchsorted(points[:,-1], rval)
    for i in range(0, nr):
        tilepoints = points[indx[i]:indx[i+1]]
        binpoints.append(tilepoints)
    return binpoints

def weighted_average(values, weights):
    average = np.average(values, weights=weights)
    variance = np.average((values-average)**2, weights=weights)
    return (average, np.sqrt(variance))


# Generate single-hit response from pickoff data
#def genPR(path, dqdim, bk, sgn, h1, h2, nppr, smth,
#          tc, rpr, f1, f2, fw, ntrace=-1, maxtraces=-1, loadfile=0, 
#          ch2=False, simple=True, debug=False):

# Generate single-hit response from pickoff data
# using traces corresponding to supplied set of points
def pts2pr(pts, wfms, dqdim, bk, sgn, h1, h2, h3, nppr, smth,
           tc, rpr, f1, f2, fw, ntrace=-1, simple=True, shreturn=False):
     
    # Sort
    pts = pts[pts[:,4].argsort()]
    pts = np.flip(pts, 0)
    
    traces = []
    # Generate list of traces corresponding to pts
    for i in range(0, len(pts)):
        stk = int(pts[i, 5])
        stkn = int(pts[i, 6])
        traces.append(wfms[stk][stkn])
    
    return genPR(traces, dqdim, bk, sgn, h1, h2, h3, nppr, smth,
                 tc, rpr, f1, f2, fw, ntrace=ntrace, shreturn=shreturn)