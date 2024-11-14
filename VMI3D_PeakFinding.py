import numpy as np
from scipy import optimize
from scipy.signal import find_peaks
from numpy.fft import fft, ifft

def gauss(x, center, amp, fwhm):
    return amp*np.exp(-(x - center)**2/(2*(fwhm/2.355)**2))

# nmax: maximum number of hits
# x0: raw signal (trace)
# t: time axis
# t0: rough position of reference peak
# twidth: rough width of reference peak
# F: average 1-hit response function
# S0: Reference and background filter
# SNR: sets minimum amplitude threshold
def fitting_routine(nmax, x0, t, t0, twidth, F, S0, SNR, dt, gausstresh, norm=False):
    
    # 1 or 2 channel
    ch = 0
    if dt < 0.3: ch=1
    if dt > 0.3: ch=2
    
    # treshold background value
    tresh = 0.5*(1/SNR)
    resmax = (0.5*tresh)**2
    
    # fit loss and solver [lsmr solver has no significant advantage]
    loss = "cauchy"
    solv = "exact"
    
    # define fitting functions
    # N=0 is just the reference and background
    # N=X includes X hits
    def funcN(N):
        if N > 11: return None # not supported
        def Func(x, t, y):
            f = gauss(t,x[0],x[1],x[2]) - y
            if N > 0: f += x[4]*F(t-x[3])
            if N > 1: f += x[6]*F(t-x[5])
            if N > 2: f += x[8]*F(t-x[7])
            if N > 3: f += x[10]*F(t-x[9])
            if N > 4: f += x[12]*F(t-x[11])
            if N > 5: f += x[14]*F(t-x[13])
            if N > 6: f += x[16]*F(t-x[15])
            if N > 7: f += x[18]*F(t-x[17])
            if N > 8: f += x[20]*F(t-x[19])
            if N > 9: f += x[22]*F(t-x[21])
            if N > 10: f += x[24]*F(t-x[23])
            return f**2
        return Func

    # apply reference+background filter
    x = ifft(fft(x0) / fft(S0)).real
    
    # maximum of 1-hit response
    arg1 = np.argmax(F(t)) # in pts
    t1 = t[arg1] # in ns
    
    # position of reference peak
    targ = int(np.round(t0/dt))  # in pts
    t0 = t[targ] # in ns
    t0max = 20 # +- bound for reference peak position fit
    
    # shift forward to avoid fitting near the edge
    x = np.roll(x, targ)
    
    # define guess for reference peak fits
    if ch==2:
        gtn = 12
        g1 = 1
    if ch==1:
        gtn = 1
        g1 = 1.1
    peaks, properties = find_peaks(x, height=1-gtn*gausstresh, width=1)
    
    if len(peaks)==0: # bad shot (can't find reference peak)
        return None
    if len(peaks)==1:
        guess0 = [t[np.argmax(x)], g1, twidth]   
    else: # the first peak is more likely to be the right one
        if t[peaks[0]] > t0-t0max:
            guess0 = [t[peaks[0]], g1, twidth]
        else: # if the first peak is too early it might be something rolled over from the end of the trace
              # try: check if this happens more than once
            guess0 = [t[peaks[1]], g1, twidth]
 
    # fit the reference gaussian
    bounds0 = [[t0-t0max, 1-12*gausstresh, twidth-0.1],
               [t0+t0max, 1+12*gausstresh, twidth+0.1]]
    p0 = optimize.least_squares(fun=funcN(0), x0=guess0, args=(t,x), bounds=bounds0, loss=loss)

    ##########
    # 1st loop
    ##########
    
    # find local maxima and filter out bad results
    Arg = [] # stores indices of bad peaks
    sig = x - gauss(t, *p0.x) # remove reference
    peaks, properties = find_peaks(sig, height=2*tresh, width=1)
    for i in range(len(peaks)): # check if peak is less than 20% of largest peak and within 5 ns of it, and reject
        if properties["peak_heights"][i] / properties["peak_heights"].max() < 0.2:
            if abs(peaks[i] - peaks[np.argmax(properties["peak_heights"])]) < 5/dt:
                Arg.append(i)
    peaks = np.delete(peaks, Arg)
    peaks = peaks[np.argsort(peaks)]
    amps0 = np.delete(properties["peak_heights"], Arg)
    amps0 = amps0[np.argsort(peaks)]
        
    # copy to use later
    x1 = x.copy()
    
    # trim fit range to include 50 pts on either side of the peaks of interest
    if len(peaks)!=0:
        argmin = np.argmin(abs(t - t[peaks[np.argmin(peaks)]]))
        argmax = np.argmin(abs(t - t[peaks[np.argmax(peaks)]]))
        arg0 = np.argmin(abs(p0.x[0] - t))
        if t[argmin] < p0.x[0]:
            if t[argmax] > p0.x[0]:
                tfit = t[argmin-20:argmax+20]
                xfit = x[argmin-20:argmax+20] 
            else:
                tfit = t[argmin-20:arg0+20]
                xfit = x[argmin-20:arg0+20] 
        else:
            tfit = t[arg0-20:argmax+20]
            xfit = x[arg0-20:argmax+20]             
    
    if len(peaks)==0:
        P = [gauss(t, *p0.x)]
        S = [0]
        
    else:
        
        if ((len(peaks)==9) or (len(peaks)>=nmax)):
            # take the 9 most intense peaks
            peaks = peaks[np.argsort(amps0)[::-1]][:9]
            peaks = peaks[np.argsort(peaks)]
            
        npeaks = len(peaks)
        guess = [guess0[0], 1, twidth]        
        boundsL = [t0-t0max, 1-gausstresh, twidth-0.1]
        boundsU = [t0+t0max, 1+gausstresh, twidth+0.1]
        
        for i in range(0, npeaks):
            guess = guess + [t[peaks[i]] - t1, abs(x[peaks[i]])]
            boundsL = boundsL + [t[peaks[i]]-3-t1, abs(x[peaks[i]])/5]
            boundsU = boundsU + [t[peaks[i]]+3-t1, abs(x[peaks[i]])*5]
        
        bounds = [boundsL, boundsU]
        
        res = optimize.least_squares(fun=funcN(npeaks),
                                     x0=guess,
                                     args=(tfit, xfit),
                                     bounds=bounds,
                                     loss=loss, tr_solver=solv)   

        P = [gauss(t, *res.x[:3])]
        S = [len(peaks), res.x[0]]
        
        for i in range(0, npeaks):
            P = P + [res.x[4+2*i]*F(t-res.x[3+2*i])]
            S = S + [res.x[3+2*i]+t1-res.x[0], res.x[4+2*i]]


    ##########
    # 2nd loop
    # Subtract current results and look for remaining hits
    ##########

    sig = x - np.sum(P, axis=0)
    peaks2, properties2 = find_peaks(sig, height=2*tresh, width=1)
    amps = properties2["peak_heights"]
    
    # remove smallest peak if below threshold
    # Try: remove all peaks below threshold
    btreslow = False
    btreshigh = False
    if len(peaks)>0:
        arg = np.argmin(S[2:][1::2])
        if S[2:][1::2][arg] < tresh: # if smallest peak is below threshold
            P2 = np.delete(P[1:], arg)
            residuals = np.sum((x - np.sum(P2, axis=0))**2)/len(t) # new residual after removing peak
            if residuals < resmax*S[0]/2: # if the residual is still good delete the peak
                S2 = np.delete(S[2:], 2*arg)
                S2 = np.delete(S2, 2*arg)
                S2 = np.insert(S2, 0, S[1])
                S = np.insert(S2, 0, len(peaks)-1)
                P = np.insert(P2, 0, P[0])             
                btreslow = True
            else:
                btreshigh = True
    
    # Do second loop if needed
    # That is, if there are still remaining peaks after subtraction
    # and if the smallest peak was not removed or it was removed and the residual is still above threshold
    if ((len(peaks2)!=0) and (btreslow==False)):
        
        if btreshigh==False: # smallest peak was not removed
            # add in the largest peak from the second check
            peaks = np.insert(peaks, 0, peaks2[np.argmax(amps)])

        else: # smallest peak was removed and the residual is still above threshold
            # remove the smallest peak and add in the largest peak from the second check
            peaks = np.delete(peaks, arg)
            peaks = np.insert(peaks, 0, peaks2[np.argmax(amps)])
            
            if len(peaks2)>1: # add in the second largest peak
                peaks2 = np.delete(peaks2, np.argmax(amps))
                amps = np.delete(amps, np.argmax(amps))
                peaks = np.insert(peaks, 0, peaks2[np.argmax(amps)])
                x1[peaks[1]] = sig[peaks[1]] # stores peak height to be accessed later (maybe do this another way?)
                
        # trim fit range to include 20 pts on either side of the peaks of interest
        argmin = np.argmin(abs(t - t[peaks[np.argmin(peaks)]]))
        argmax = np.argmin(abs(t - t[peaks[np.argmax(peaks)]]))
        arg0 = np.argmin(abs(p0.x[0] - t))
        if t[argmin] < p0.x[0]:
            if t[argmax] > p0.x[0]:
                tfit = t[argmin-20:argmax+20]
                xfit = x[argmin-20:argmax+20] 
            else:
                tfit = t[argmin-20:arg0+20]
                xfit = x[argmin-20:arg0+20] 
        else:
            tfit = t[arg0-20:argmax+20]
            xfit = x[arg0-20:argmax+20]
            
        npeaks = len(peaks)
        guess = [guess0[0], 1, twidth, t[peaks[0]] - t1, abs(sig[peaks[0]])]        
        boundsL = [t0-t0max, 1-gausstresh, twidth-0.1, t[peaks[0]]-3-t1, abs(sig[peaks[0]])/5]
        boundsU = [t0+t0max, 1+gausstresh, twidth+0.1, t[peaks[0]]+3-t1, abs(sig[peaks[0]])*5]
        
        for i in range(1, npeaks):
            guess = guess + [t[peaks[i]] - t1, abs(x1[peaks[i]])]
            boundsL = boundsL + [t[peaks[i]]-3-t1, abs(x1[peaks[i]])/5]
            boundsU = boundsU + [t[peaks[i]]+3-t1, abs(x1[peaks[i]])*5]
        
        bounds = [boundsL, boundsU]
        
        res = optimize.least_squares(fun=funcN(npeaks),
                                     x0=guess,
                                     args=(tfit, xfit),
                                     bounds=bounds,
                                     loss=loss, tr_solver=solv)   

        P = [gauss(t, *res.x[:3])]
        S = [len(peaks), res.x[0]]
        
        for i in range(0, npeaks):
            P = P + [res.x[4+2*i]*F(t-res.x[3+2*i])]
            S = S + [res.x[3+2*i]+t1-res.x[0], res.x[4+2*i]]    
            

        ##########
        # 3rd loop
        # Subtract current results and look for remaining hits
        ##########
        
        sig2 = x - np.sum(P, axis=0)
        peaks2, properties2 = find_peaks(sig2, height=2*tresh, width=1)
        amps=properties2["peak_heights"]
        
        # remove smallest peak if below threshold
        # try: remove all peaks below threshold
        btreslow = False
        btreshigh = False
        arg = np.argmin(S[2:][1::2])
        if S[2:][1::2][arg] < tresh: # if smallest peak is below threshold
            P2 = np.delete(P[1:], arg)
            residuals = np.sum((x - np.sum(P2, axis=0))**2)/len(t) # new residual after removing peak
            if residuals < resmax*S[0]/2: # if the residual is still good delete the peak
                S2 = np.delete(S[2:], 2*arg)
                S2 = np.delete(S2, 2*arg)
                S2 = np.insert(S2, 0, S[1])
                S = np.insert(S2, 0, len(peaks)-1)
                P = np.insert(P2, 0, P[0])             
                btreslow = True
            else:
                btreshigh = True

        # Do third loop if needed
        # That is, if there are still remaining peaks after subtraction
        # and if the smallest peak was not removed or it was removed and the residual is still above threshold
        if ((len(peaks2)!=0) and (btreslow==False)):
            
            # add in the largest peak from the second check
            peaks = np.insert(peaks, 0, peaks2[np.argmax(amps)])
            
            if ((btreshigh==True) and (len(peaks)>2)):
                peaks = np.delete(peaks, arg+1)
                if ((arg!=0) and len(peaks)>2): x1[peaks[2]] = sig[peaks[1]]
                sig[peaks[1]] = sig2[np.argmax(amps)]

                if len(peaks2)>1: # add in the second largest peak
                    peaks2 = np.delete(peaks2, np.argmax(amps))
                    amps = np.delete(amps, np.argmax(amps))
                    peaks = np.insert(peaks, 0, peaks2[np.argmax(amps)])
                    
            # trim fit range to include 20 pts on either side of the peaks of interest
            argmin = np.argmin(abs(t - t[peaks[np.argmin(peaks)]]))
            argmax = np.argmin(abs(t - t[peaks[np.argmax(peaks)]]))
            arg0 = np.argmin(abs(p0.x[0] - t))
            if t[argmin] < p0.x[0]:
                if t[argmax] > p0.x[0]:
                    tfit = t[argmin-20:argmax+20]
                    xfit = x[argmin-20:argmax+20] 
                else:
                    tfit = t[argmin-20:arg0+20]
                    xfit = x[argmin-20:arg0+20] 
            else:
                tfit = t[arg0-20:argmax+20]
                xfit = x[arg0-20:argmax+20]
                
            npeaks = len(peaks)
            guess = [guess0[0], 1, twidth, t[peaks[0]] - t1, abs(sig2[peaks[0]]), t[peaks[1]] - t1, abs(sig[peaks[1]])]        
            boundsL = [t0-t0max, 1-gausstresh, twidth-0.1,
                       t[peaks[0]]-3-t1, abs(sig2[peaks[0]])/5, t[peaks[1]]-3-t1, abs(sig[peaks[1]])/5]
            boundsU = [t0+t0max, 1+gausstresh, twidth+0.1,
                       t[peaks[0]]+3-t1, abs(sig2[peaks[0]])*5, t[peaks[1]]+3-t1, abs(sig[peaks[1]])*5]
            
            for i in range(2, npeaks):
                guess = guess + [t[peaks[i]] - t1, abs(x1[peaks[i]])]
                boundsL = boundsL + [t[peaks[i]]-3-t1, abs(x1[peaks[i]])/5]
                boundsU = boundsU + [t[peaks[i]]+3-t1, abs(x1[peaks[i]])*5]
            
            bounds = [boundsL, boundsU]
            
            res = optimize.least_squares(fun=funcN(npeaks),
                                         x0=guess,
                                         args=(tfit, xfit),
                                         bounds=bounds,
                                         loss=loss, tr_solver=solv)   

            P = [gauss(t, *res.x[:3])]
            S = [len(peaks), res.x[0]]
            
            for i in range(0, npeaks):
                P = P + [res.x[4+2*i]*F(t-res.x[3+2*i])]
                S = S + [res.x[3+2*i]+t1-res.x[0], res.x[4+2*i]]    
                
            ##########
            # Final Check
            ##########
            
            sig = x - np.sum(P, axis=0)
            peaks2, properties2 = find_peaks(sig, height=2*tresh, width=1)    

            # remove any duplicate peaks from the final check
            Arg = []
            for i in range(len(peaks2)):
                if np.any(peaks2[i]==peaks):
                    Arg.append(i)
            peaks2 = np.delete(peaks2, Arg)
            
            # remove smallest peak if below threshold
            # why not remove all peaks below threshold?
            btreslow = False
            arg = np.argmin(S[2:][1::2])
            if S[2:][1::2][arg] < tresh: # if smallest peak is below threshold
                P2 = np.delete(P[1:], arg)
                residuals = np.sum((x - np.sum(P2, axis=0))**2)/len(t) # new residual after removing peak
                if residuals < resmax*S[0]/2: # if the residual is still good delete the peak
                    S2 = np.delete(S[2:], 2*arg)
                    S2 = np.delete(S2, 2*arg)
                    S2 = np.insert(S2, 0, S[1])
                    S = np.insert(S2, 0, len(peaks)-1)
                    P = np.insert(P2, 0, P[0])             
                    btreslow = True
            else:
                residuals = np.sum((x - np.sum(P, axis=0))**2)/len(t)
                if residuals < resmax*S[0]/2:
                    btreslow = True
            
            # if residual is still high, discard entire shot
            if (btreslow==False):
                P = [gauss(t,*p0.x)]
                S = [99]

    return S, P

# find index of shift to align S with S0
# arg is a guess, factor*3 is the number of points to scan
# factor*10 is the range to perform least squares on
def find_arg(S0, S, arg, factor):
    
    j = np.arange(-3*factor, 3*factor+1) # range to scan
    argmin, argmax = arg - 10*factor, arg + 10*factor + 1
    
    res = np.zeros(len(j))
    for k in range(len(j)):
        S1 = np.roll(S, j[k])
        res[k] = np.sum((S1[argmin:argmax]-S0[argmin:argmax])**2)
    
    arg = np.argmin(res)
    if arg==0 or arg==len(j):
        print("WARNING - Maximum on edge of scan range!")
    return j[arg]
