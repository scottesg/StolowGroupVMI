import numpy as np
import matplotlib.pyplot as plt
from numpy.fft import fft, ifft, rfft, rfftfreq
from scipy.ndimage import gaussian_filter
from scipy.signal import find_peaks, butter, sosfiltfilt, freqz
from scipy.optimize import least_squares
from VMI3D_Fitting import res_mgauss, mgauss
from VMI3D_IO import readwfm

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

# Returns all peaks from the input trace using deconvolution with the supplied response
def getpeaks(trace, response, h, snr, filt, blpts, nmax, t, ef,
             widthlower=0.5, widthupper=2, intbd=None, d=20,
             plot=False, gfit=False, gsm=0, awf=False, fixbaseline=True):
    
    bl = np.mean(trace[:blpts])
    wd = wiener_deconvolution(trace-bl, response, snr, relativeposition=True)
    
    if gsm>0:
        wd = gaussian_filter(wd, gsm)
        if ef is not None:
            lef = len(ef)
            wd[:lef] *= ef
            wd[-lef:] *= ef[::-1]
    else:
        wd = fftsmooth(t, wd, filt, edgefixer=ef)[1]
        
    if plot:
        plt.plot(t, wd)
    
    pk, properties = find_peaks(wd, height=h, width=widthlower)
    I = properties['peak_heights']
    fwhm = properties['widths']
    
    if len(pk) == 0:
        if plot: print("No hits")
        return None
    
    if len(pk) > nmax:
        if plot: print("Too many hits: {}".format(len(pk)))
        return None

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
            
    else:
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
        for i in range(0, len(out)):
            pltpos = out[i,0]
            ax2.scatter(pltpos, max(wd), s=12, c='red')
            ax2.vlines(pltpos, min(wd), max(wd), lw=2, color='red')
        ax2.set_title("Deconvolution with Threshold")
        ax2.set_xlabel("Time (ns)")
        ax2.set_ylabel("Deconvolution Amplitude")
        ax2.set_xlim(tmin, tmax)
        
    return out, t, trace, wd

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

# Generate single-hit response from pickoff data
def genPR_OldVersion(path, blshift, h, f1=0.18, f2=0.12, bk=0,
          loadfile=0, maxtraces=-1, ntrace=-1, sgn=-1, dqdim=2048, shreturn=False):
    
    if loadfile>0:
        path = path.replace('*',str(loadfile))
    
    if type(path) is str:
        data = readwfm(path, dqdim)
    else: data = path
    
    inds = []
    shtraces = [] # for debugging
    response = np.zeros(len(data[0]))
    if maxtraces==-1:
        maxtraces = len(data)
    
    c = 0
    for i in range(0, min(len(data), maxtraces)):
        if c == ntrace: break
        if i % 1000 == 0: print("Trace #{} / {} ({})".format(i, len(data), c))
        tmp = sgn * data[i]
        tmp = tmp - np.mean(tmp[:blshift])
        tmp = tmp - bk
        tmps = gaussian_filter(tmp, 2)
        peaks1 = find_peaks(tmps, h)
        
        if not len(peaks1[0])==1: continue
    
        pos1 = peaks1[0][0]
        height1 = peaks1[1]['peak_heights'][0]
        
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
                    diff = 1000 - pos1
                    inds.append(pos1)
                    respi = np.zeros(len(data[0]))
                    if diff == 0:
                        respi = tmp
                    else:
                        respi[diff:] = tmp[:-diff]
                        respi[:diff] = tmp[-diff:]
                    shtraces.append(respi)
                    response += respi
                    c = c + 1
            
    print("Number of Samples: {}/{}".format(c, len(data)))
    if (c==0) or (c < ntrace):
        return np.zeros(len(response))
    
    response = response / c
    response = response - np.mean(response[:500])
    
    if shreturn: return response, shtraces
    else: return response

# Generate single-hit response from pickoff data
# using traces corresponding to supplied set of points
def pts2pr(pts, wfms, blshift, h, ntrace=-1, shreturn=False):
     
    # Sort
    pts = pts[pts[:,4].argsort()]
    pts = np.flip(pts, 0)
    
    traces = []
    # Generate list of traces corresponding to pts
    for i in range(0, len(pts)):
        stk = int(pts[i, 5])
        stkn = int(pts[i, 6])
        traces.append(wfms[stk][stkn])
    
    return genPR_OldVersion(traces, blshift, h, ntrace=ntrace, shreturn=shreturn)
