import numpy as np
from VMI3D_IO import readwfm, readctr, genT
from VMI3D_Functions import (xycorrectpoints, pldist, getpeaks)
import os.path
from VMI3D_Fitting import gauss
from timeit import default_timer as timer
from scipy.signal import butter
import scipy.ndimage as ndimage
from scipy.optimize import linear_sum_assignment as lsa

# Main function to process 3DVMI data
def load3D(ctrpath, hitspath, offsetpath, outpath, # Paths
           stksize, nctrmax=10, matchcountnumber=False, nctrlist=[], # Data extraction limits
           rct=-1, act=-1, useshad=False, nprobreturn=2,
           # Correlation params (act also toggles correlation processing
           # and rct toggles assignment comparison)
           save=True, name='tmp', # Saving params
           maxstks=-1, v=True, debug=False): # Testing options
    
    # Check for pickoff/camera offsets
    if os.path.exists(offsetpath):
        offsets = np.load(offsetpath).astype(int)
    else:
        print("Warning: offsets.npy not found at path: " + offsetpath)
        offsets = None
        
    # Check for spatial correction factors
    scf = False
    if os.path.exists(outpath + "/detmap.npy"):
        print("Found spatial correction factors.")
        detmap = np.load(outpath + "/detmap.npy", allow_pickle=True).item()
        scf = True
    
    # Load correlation fit line if correlation is requested
    if act >= 0:
        if useshad: source = 'shad.npy'
        else: source = 'corfit.npy'
        if os.path.exists(outpath + "/{}".format(source)):
                corstats = np.load(outpath + "/{}".format(source), allow_pickle=True)
        else:
            print("Error: {} not found in path: ".format(source) + outpath)
            raise SystemExit()
            
    # Read in centroids and hits
    if v: print("Loading data...")
    ctrs = readctr(ctrpath, stksize, groupstks=True)
    hits = np.load(hitspath)
    nstks = len(ctrs)
    if v: print("Done.\n\n")
    
    # Track loss of ctrs from each step
    totalctrs = len(np.vstack(ctrs))
    ctrlossOffset = 0
    ctrlossXYCorrect = 0
    ctrlossNctrmax = 0
    ctrlossNohits = 0
    ctrlossNomatch = 0
    
    # Analyse each stack
    allhits = []
    allunmatched = []
    cscodes = []
    hitspershot = [[nstks, stksize, nctrmax]]
    for i in range(0, nstks):
        print("Processing stack {}...".format(i))
        if (maxstks > 0) and (i > maxstks): break
        if v: s = timer()    
    
        # Extract ith stacks
        ctrstk = ctrs[i]
        hitstk = hits[hits[:,4]==i]
        
        # Perform offset correction
        if offsets is not None:
            offset = offsets[0, i]
            quality = offsets[1, i]
            if quality < 3:
                print("Warning: Possible bad stack at index {}".format(i))
                continue
            if offset > 0:
                ctrstk[:,3] = ctrstk[:,3] + offset
                count = len(ctrstk)
                ctrstk = ctrstk[ctrstk[:,3]<stksize]
                ctrlossOffset += count - len(ctrstk)
        
        # Perform spatial correction
        if scf:
            count = len(ctrstk)
            ctrstk = xycorrectpoints(ctrstk, detmap, ctrs=True)
            ctrlossXYCorrect += count - len(ctrstk)
        
        ctrlossNctrmax += len(ctrstk[ctrstk[:,4]>nctrmax])
        
        # Processing data with n centroids per shot
        # n ranges from 1 to nctrmax
        for n in range(1, nctrmax+1):
            
            if len(nctrlist) > 0:
                if not n in nctrlist: continue
            
            # Extract indices of n hit shots
            inds = ctrstk[ctrstk[:,4]==n][:,3].astype(int)
            inds = np.unique(inds)
        
            # Analyse each shot in the stack
            hitsn = []
            for ind in inds:
                
                if debug:
                    print("\n\nStack {}, Shot {}".format(i, ind))
                
                # Extract hits and centroids for this shot
                hitsi = hitstk[hitstk[:,5]==ind]
                ctri = ctrstk[ctrstk[:,3]==ind]
                
                if len(hitsi)==0:
                    ctrlossNohits += len(ctri)
                    hitspershot.append([n, 0, 0])
                    cscodes.append([i, ind, n, 0, 0, -1])
                    if debug:
                        print("No Hits in Shot.")
                    continue
                
                # Match pickoff/camera hits
                if act < 0: # No correlation matching
                                
                    # Enforce count number matching
                    if matchcountnumber and not len(hitsi)==n: continue
                    
                    # Use most intense peak and ctr
                    ctri = ctri[ctri[:,2].argsort()][::-1]
                    amax = np.argmax(hitsi[:,1])
                    h = hitsi[amax,1]
                    pos = hitsi[amax,0]
                    
                    # Add hit to list
                    hit = [ctri[0,0], ctri[0,1], pos, ctri[0,2], h, i, ind]
                    hitsn.append(hit)
                    
                elif act >= 0: # With correlation matching
                    
                    matched, unmatched, cscode = cscorr(ctri, hitsi, corstats,
                                                               act, rct, debug=debug,
                                                               useshad=useshad,
                                                               nprobreturn=nprobreturn,
                                                               stknum=i, shotnum=ind)
                    cscodes.append([i, ind, n, len(hitsi), len(matched), cscode])
                    hitspershot.append([n, len(hitsi), len(matched)])
                    hitsn.extend(matched)
                    allunmatched.extend(unmatched)
                    ctrlossNomatch += len(unmatched)
                    
                    if debug:
                        print("cscode: {}".format(cscode))
        
            # add hits to full list
            allhits = allhits + hitsn
            if v: print("[{} hit data] Hits found: {}".format(n, len(hitsn)))
        if v: print("Total hits found so far: {}".format(len(allhits)))
        if v: print("Stack {} time: {:.2f}".format(i, timer() - s))
    
    # Convert to numpy array
    allhits = np.array(allhits)
       
    if save:
        
        # File name
        fname = "vmi3d_{}".format(name)
        if act >= 0: fname = fname + "_C"
        
        # Save data
        np.save(outpath + "/{}.pt.npy".format(fname), allhits)
        if act >= 0:
            hitspershot = np.array(hitspershot).astype('int')
            cscodes = np.array(cscodes).astype('int')
            np.save(outpath + "/{}.hps.npy".format(fname), hitspershot)
            np.save(outpath + "/{}.bad.npy".format(fname), allunmatched)
            np.save(outpath + "/{}.csc.npy".format(fname), cscodes)
        
        # Save parameter file
        file = open(outpath + "/{}.params.txt".format(fname),'w')
        file.write("Summary of 3DVMI analysis: {}\n".format(fname))
        file.write("Output Path: {}\n".format(outpath))
        file.write("Centroid Path: {}\n".format(ctrpath))
        file.write("Hits Path: {}\n".format(hitspath))
        file.write("Offsets Path: {}\n".format(offsetpath))
        
        file.write("\nSize of stacks: {}".format(stksize))
        file.write("\n\nMaximum number of centroids per image: {}".format(nctrmax))
        
        file.write("\n\nCorrelation details:\n________")
        if act >= 0:
            file.write("\nUsing pickoff-camera correlation matching with:")
            file.write("\nAbsolute Correlation Threshold: {}".format(act))
            file.write("\nRelative Correlation Threshold: {}".format(rct))
            if not useshad:
                file.write("\nAmplitude matching probability: Linear Fit")
                file.write("\nSlope = {}".format(corstats[-1,0]))
                file.write("\nY-intercept = {}".format(corstats[-1,1]))
                file.write("\nVariance = {}".format(corstats[-1,2]))
            else:
                file.write("\nAmplitude matching probability: SHAD")
            
        else:
            file.write("\nNot using pickoff-camera correlation matching")
            file.write("\nEnforcing exact match between centroid and pickoff hit numbers? {}".format(matchcountnumber))
        
        file.write("\n\nStatistics:\n________")
        file.write("\nNumber of stacks: {}".format(maxstks))
        file.write("\nTotal number of hits found... {}".format(len(allhits)))
        if act >= 0:
            ctrcount = np.sum(hitspershot[1:,0])
            hitcount = np.sum(hitspershot[1:,1])
            file.write("\nTotal number of centroids in all images... {}".format(ctrcount))
            file.write("\n\t...{:2.2f}% Assigned".format(100*len(allhits)/ctrcount))
            file.write("\nTotal number of scope hits in all images... {}".format(hitcount))
            file.write("\n\t...{:2.2f}% Assigned".format(100*len(allhits)/hitcount))
            
            file.write("\n\nTotal number of available centroids: {}".format(totalctrs))
            file.write("\nCentroid loss report...")
            
            file.write("\nOffset: {} ({:2.2f}%)"
                       .format(ctrlossOffset, 100*ctrlossOffset/totalctrs))
            file.write("\nTile Limit: {} ({:2.2f}%)"
                       .format(ctrlossXYCorrect, 100*ctrlossXYCorrect/totalctrs))
            file.write("\nCentroid Count: {} ({:2.2f}%)"
                       .format(ctrlossNctrmax, 100*ctrlossNctrmax/totalctrs))
            file.write("\nNo Hits: {} ({:2.2f}%)"
                       .format(ctrlossNohits, 100*ctrlossNohits/totalctrs))
            file.write("\nNo Match: {} ({:2.2f}%)"
                       .format(ctrlossNomatch, 100*ctrlossNomatch/totalctrs))
            
        file.close()
    
    return allhits

# Main function for performing 3DVMI amplitude matching
def cscorr(ctrs, hits, corstats, act, rct, nprobreturn=2,
           debug=False, useshad=False, probsonly=False, stknum=0, shotnum=0):  
    
    if useshad:
        # Unpack probability distribution
        shad = corstats[0]
        cbins = corstats[1]
        pbins = corstats[2]
        shadlen = len(shad)
        shadmax = np.max(shad)
    else: 
        # Unpack fit parameters
        slope = corstats[-1,0]
        yint = corstats[-1,1]
        bins = corstats[:-1,5]
        nbin = len(bins) - 1
        islope = -1/slope
    
    # Definitions
    nctrs = len(ctrs)
    nhits = len(hits)
    goodpts = []
    unmatchedctrs = []
    cscode = -1
    
    # cscode: -1: Should not be possible
    # 0: No hits/ctrs, 1: No nonzero matches, 11: No nonzero products
    # 2: No RCT compare, 3: RCT compare -  No Ambiguities, 33: RCT Compare - no probs
    # 4: RCT Compare - Ambiguities
    
    if debug:
        print("STARTING CSCORR (act={}, rct={})".format(act, rct))
        print("with {} centroids and {} hits:".format(nctrs, nhits))
        print("Centroid Amplitudes:")
        for i in range(0, nctrs): print("{:2.4f}".format(ctrs[i,2]))
        print("Hit Amplitudes:")
        for j in range(0, nhits): print("{:2.4f}".format(hits[j,1]))
    
    # Return if there are no centroids or hits
    if nhits==0 or nctrs==0:
        if debug: print("No Centroids or Hits in Shot")
        cscode = 0
        unmatchedctrs = ctrs
        if probsonly: return cscode
        return goodpts, unmatchedctrs, cscode
    
    # Calculate the event probability matrix
    if debug: s = timer()
    epm = np.zeros((nctrs, nhits))
    for i in range(0, nctrs): # For each centroid
        ci = ctrs[i,2]
        
        for j in range(0, nhits): # Try each pickoff hit
            pi = hits[j,1]
            
            if useshad:
                # Determine probability for pair
                cind = np.searchsorted(cbins, ci) - 1
                pind = np.searchsorted(pbins, pi) - 1
                
                if (min(cind, pind) >= 0) and (max(cind, pind) < shadlen):
                    epm[i,j] = shad[cind, pind] / shadmax
            
            else:
                # Determine correct bin on fit line
                linepos = np.abs(pldist([ci, pi], islope, yint))
                ind = np.searchsorted(bins, linepos) - 1
                if ind < 0: ind = 0 # Use first bin value for data below first bin
                if ind >= nbin: ind = -3 # Last bin for data greater than last bin
                
                # Assign match value from gaussian
                sigma = corstats[ind, 2]
                d0 = corstats[ind, 1]
                d = pldist([ci, pi], slope, yint)
                m = gauss(d, 0, 1, d0, sigma)
                epm[i,j] = m
        
    if debug:
        print("Time EPM: {}".format(timer() - s))
        print("EPM matrix:")
        print((1000*epm).astype(int))
        
    # Apply act
    epm[epm < act] = 0
            
    # Return if there are no non-zero matches
    epmax = np.max(epm)
    if epmax<act or np.isclose(epmax, act):
        if debug: print("No Nonzero Elements in EPM Matrix")
        cscode = 1
        unmatchedctrs = ctrs
        if probsonly: return cscode
        return goodpts, unmatchedctrs, cscode
    
    if debug:
        print("EPM matrix:")
        print((1000*epm).astype(int))

    # Find best assignments
    if debug: s = timer()
    probs, pairsets = assignmentproducts(epm, nprobreturn, act, debug)
    if debug: print("Time AP_Total: {}".format(timer() - s))

    if probsonly:
        return probs, pairsets
    
    # No nonzero products
    if len(pairsets) == 0: 
        if debug: print("No Nonzero Assignment Products")
        cscode = 11
        unmatchedctrs = ctrs
        return goodpts, unmatchedctrs, cscode
    
    if debug:
        for i in range(0, len(pairsets)):
            print("Assignment Value - {:.2f}:".format(probs[i]))
            print(pairsets[i])
    
    if rct > 0:
        if debug: s = timer()
        bestmatch, cscode = cscompare(probs, pairsets, rct)
        if debug: print("Time CSCOM: {}".format(timer() - s))
    else:
        cscode = 2
        bestmatch = pairsets[0]

    matchctrs = []
    for i in range(0, len(bestmatch)):
        c = bestmatch[i][0]
        s = bestmatch[i][1]
        if epm[c,s] < act: continue
        matchctrs.append(c)
        if debug:
            print("Centroid {} matches hit {}!".format(c+1, s+1))
        ctr = ctrs[c]
        hit = hits[s]
        goodpts.append([ctr[0], ctr[1], hit[0], ctr[2], hit[1], nctrs, nhits, stknum, shotnum])
    
    # List unmatched centroids
    unmatchedinds = list(set(range(0, nctrs)) - set(matchctrs))
    if debug:
        print("Unmatched centroids: {}".format(np.array(unmatchedinds)+1))
    unmatchedctrs = ctrs[unmatchedinds]
    
    return goodpts, unmatchedctrs, cscode

# Compare best assignment to next best assignments to determine ambiguity
# and accept overlaping pairs for ambiguous assignments
def cscompare(probs, pairsets, rct):
    
    # Apply rct scale to best prob
    rct = probs[0]*rct
        
    # Number of ambiguous assignments
    probsnz = probs[np.logical_not(np.isclose(probs, 0))]
    if len(probsnz) == 0:
        return [], 33
    
    abind = np.searchsorted(-1*probsnz, -1*(probsnz[0]-rct))
 
    # No ambiguity
    if abind == 1:
        bestmatch = pairsets[0]
        cscode = 3
        
    # Ambiguity found
    else:            
        # Find matches shared by all ambiguous assignments
        sets = []
        for i in range(0, abind):
            if probs[i]==0: continue
            ps = pairsets[i]
            pairs = [tuple(psj) for psj in ps]
            sets.append(set(pairs))
        g = set.intersection(*sets)
        bestmatch = np.array(list(g))
        cscode = 4
        
    return bestmatch, cscode

# Calculate best assignment products for a single shot from epm
def assignmentproducts(m, bestn, th, debug, rmcompete=False):
    
    if debug:
        print("Starting AssignmentProducts on matrix:")
        print(m)
        s = timer()
    
    nctrs, nhits = m.shape
    inputm = m
    
    # Find and remove zero rows/cols from matrix
    mz = np.isclose(0, m)
    msorted = False
    
    if not False in mz: # Check if matrix is empty
        if debug: print("Matrix is empty!")
        return np.zeros(bestn), []
    
    if True in mz: # Only if the matrix contains at least 1 zero
        if debug: print("Matrix contains a zero - processing.")    
    
        # Count zeros
        nzeroctrs = list(np.sum(mz, 1))
        nzerohits = list(np.sum(mz, 0))
    
        if rmcompete:
            # Negative sum of each row/col
            ctrsum = [-1*np.sum(m[i]) for i in range(0, nctrs)]
            hitsum = [-1*np.sum(m[:,i]) for i in range(0, nhits)]
            
            # Sort the matrix to move zeros to bottom/right
            ctrsort =  np.lexsort((ctrsum, nzeroctrs))
            hitsort =  np.lexsort((hitsum, nzerohits))
        else:
            ctrsort = np.argsort(nzeroctrs)
            hitsort = np.argsort(nzerohits)
        
        m = m[ctrsort][:, hitsort]
        mz = mz[ctrsort][:, hitsort]
        msorted = True
        
        if debug:
            print("Sorted Matrix: (x1000)")
            print((1000*m).astype(int))
            print("Sorted MZ:")
            print(mz)
        
        # Count empty rows/cols
        nrmc1 = nzeroctrs.count(nhits)
        nrmh1 = nzerohits.count(nctrs)
        
        # Count competing rows/cols [Not Working]
        if rmcompete:
            nrmc2 = max(nzeroctrs.count(nhits-1) - 1, 0)
            nrmc3 = np.sum(mz[-(nrmc1+nrmc2+1):,0])
            nrmh2 = max(nzerohits.count(nctrs-1) - 1, 0)
            nrmh3 = np.sum(mz[0,-(nrmh1+nrmh2+1):])
            nrmc = nrmc1 + max(nrmc2 - nrmc3, 0)
            nrmh = nrmh1 + max(nrmh2 - nrmh3, 0)
            if debug:
                print("NRMC: [{}, {}, {}]".format(nrmc1, nrmc2, nrmc3))
                print("NRMH: [{}, {}, {}]".format(nrmh1, nrmh2, nrmh3))
        else: # Only remove empty rows/cols
            nrmc = nrmc1
            nrmh = nrmh1
    
        # Remove unneeded rows/cols
        if nrmc > 0:
            if nrmh > 0:
                m = m[:-nrmc, :-nrmh]
            else: m = m[:-nrmc]
        elif nrmh > 0: m = m[:,:-nrmh]

        if debug:
            print("Removing {} ctrs and {} hits".format(nrmc, nrmh))
            print("Reduced Matrix: (x1000)")
            print((1000*m).astype(int))
            print("Time AP1_Prep: {}".format(timer() - s))

    # Calculate products
    if debug: s = timer()
    nctrs, nhits = m.shape
    products = msolver(m, th)
    if debug: print("Time AP2_Recurser: {}".format(timer() - s))
    
    # Convert to output form
    if debug: s = timer()
    probs = np.zeros(bestn)
    pairsets = []
    nprods = len(products)
    for i in range(0, min(nprods, bestn)):
        asm = products[i]
        
        # Convert indices to input matrix
        iasm = np.array(asm)
        if msorted:
            iasm[0] = ctrsort[iasm[0]]
            iasm[1] = hitsort[iasm[1]]
        
        # Filter out zero (unused) matches from assignment
        iasm = iasm.T
        filtasm = []
        for j in range(0, len(iasm)):
            if inputm[tuple(iasm[j])]>th/2:
                filtasm.append(iasm[j])
                
        if len(filtasm) > 0:
            filtasm = np.array(filtasm)
            probs[i] = np.prod(inputm[tuple(filtasm.T)])
            pairsets.append(filtasm)
    if debug: print("Time AP3_Convert: {}".format(timer() - s))
    
    return probs, pairsets

# Uses assignment problem solver to compute best product(s)
def msolver(m, th):
    
    # Prepare matrix
    m[m<th] = th
    m = np.log(m)
    
    # Find best assignment
    best = lsa(m, maximize=True)
    # Find second best here
    
    return [best]

# Calculate all assignment products recursively
# Matrix m should be square or have larger axis second
def mrecurser(m, th):
    products = []
    x, y = m.shape
    sublen = x - 1
    
    if sublen <= 0: # Recursion end condition
        for i in range(0, y):
            products.append(max(m[0,i], th))
    
    else:
        for i in range(0, y):
            submat = m[1:]
            products.append(max(m[0,i], th) * mrecurser(np.delete(submat, i, 1), th)) 
    
    return np.array(products)

# Convert index of products list from mrecurser to assignment index
def indconvert(ind):
    
    a1 = np.arange(0, len(ind), dtype=int)
    a2 = np.zeros(len(ind), dtype=int)

    track = np.arange(0, max(ind)+len(ind), dtype=int)
    for i in range(0, len(ind)):
        a2[i] = track[ind[i]]
        track = np.delete(track, ind[i])
    
    return [a1, a2]

# Perform peak-finding on stack(s) of 3DVMI pickoff traces
def peakfindall(wfmpath, outpath, response, bk,
                threshold, snr, filtfreq, filtorder, dt, nmax,
                gfit, ef, tc=None, rpr=None, blank=None, sgn=-1, fitdpts=20, widthl=0.5,
                widthu=2, maxstks=-1, save=False, gsm=0, fnum=-1, dqdim=2048, ch2=False, skipdecon=False,
                ch2_rprsub=False):

    # Create output directory
    if not os.path.exists(outpath + "hits/"):
        os.mkdir(outpath + "hits/")
    
    print("Loading traces...")
    if fnum > -1:
        wfmpath = wfmpath.replace('*','{}')
        wfmpath = wfmpath.format(fnum)
    wfms = readwfm(wfmpath, dqdim=dqdim, groupstks=True, ch2=ch2)
    if ch2: wfms = wfms[0]
    if fnum > -1: wfms = [wfms]
    t = genT(len(wfms[0][0]), dt)
    if blank is not None:
        blank = [np.searchsorted(t, blank[0]), np.searchsorted(t, blank[1])]
           
    if maxstks < 0:
        maxstks = len(wfms)
        
    if ch2 and not ch2_rprsub:
        tc = None
        rpr = None
    
    # Create butterworth filter
    sr = 1/(t[1]-t[0])
    bfilter = butter(filtorder, filtfreq, 'low', analog=False, output='sos', fs=sr)  

    peaksall = []
    for i in range(0, maxstks):
        s = timer()
        wfmstk = wfms[i]
    
        stksize = len(wfmstk)
        for j in range(0, stksize):
            
            if j % 1000 == 0: print("Shot #{} / {}".format(j, len(wfmstk)))
            
            wfm = wfmstk[j]
            
            wfm = sgn * (wfm - bk)
            if tc is not None:
                wfm = wfm - ndimage.shift(rpr, tc[stksize*(fnum-1) + j]-1000, mode='wrap', order=3)
            if blank is not None:
                wfm[blank[0]:blank[1]] = 0
                
            hits = getpeaks(wfm, response, threshold, snr, bfilter,
                            nmax, t, ef, widthlower=widthl, widthupper=widthu,
                            d=fitdpts, gfit=gfit, gsm=gsm, skipdecon=skipdecon)
            if hits is None:
                continue
            else:
                hits = hits[0]
                      
            hitsdata = np.zeros((len(hits), 7))
            hitsdata[:,:4] = hits
            hitsdata[:,4] = i
            hitsdata[:,5] = j
            hitsdata[:,6] = len(hits)
            peaksall.extend(hitsdata)
            
        time = timer() - s
        if fnum > -1:
            print("Stack {} [Time: {:.2f}]".format(fnum, time))
        else:
            print("Stack {} [Time: {:.2f}]".format(i, time))
    
    out = np.array(peaksall)
        
    if save:
        
        # Save data
        if fnum > -1:
            np.save(outpath + "hits/hits{}.npy".format(fnum), out)
        else:
            np.save(outpath + "hits/hits.npy", out)

        # Save parameter file
        if fnum > -1:
            file = open(outpath + "hits/hits{}.params.txt".format(fnum),'w')
        else:
            file = open(outpath + "hits/hits.params.txt",'w')
            
        file.write("Summary of 3DVMI peak-finding:\n")
        file.write("Output Path: {}\n".format(outpath+"hits/"))
        file.write("Traces Path: {}\n".format(wfmpath))

        file.write("\nThreshold: {}".format(threshold))
        file.write("\nSNR used in deconvolution: {}".format(snr))
        file.write("\nTime step: {} ns".format(dt))
        file.write("\nMaximum peaks per trace: {}".format(nmax))
        
        file.write("\n\nSmoothing:")
        if gsm>0:
            file.write("\nGaussian [{} points]".format(gsm))
        else:
            file.write("\nButterworth [Frequency: {}, Order: {}]".format(filtfreq, filtorder))
        
        
        file.write("\nUsed Gaussian fit: {}".format(gfit))
        if gfit:
            file.write("\nNumber of points in fit: {}".format(fitdpts))
            file.write("\nFit minimum width: {}".format(widthl))
            file.write("\nFit maximum width: {}".format(widthu))

        file.write("\n\nNumber of stacks analysed: {}".format(maxstks))
        file.write("\nTotal number of hits found... {}".format(len(out)))
        file.close()
    
    return out
    
# Analyse pseudo-experiment data
# Equivalent of load3D for pseudo-experiment
def performPseudo(data, shprobs, assignmentcutoff, act, rm=0, debug=False):
    
    allprobs = []
    allassignments = []
    
    for i in range(0, len(data)):
        if i % 100 == 0: print("Pseudoshot #{} / {}".format(i, len(data)))
        shotdata = np.array(data[i])
        
        # Shape like 'real' data that cscorr expects
        ctrs = np.zeros((len(shotdata)-rm, 3))
        ctrs[:,2] = shotdata[:len(shotdata)-rm,0]
        hits = np.zeros((len(shotdata), 2))
        hits[:,1] = shotdata[:,1]
        
        csc = cscorr(ctrs, hits, shprobs, act, 0, assignmentcutoff, probsonly=True, debug=debug)
        
        if type(csc) is int:
            if debug: print("Error Code: {}".format(csc))
            continue
        
        probs, assignments = csc
        allprobs.append(probs)
        allassignments.append(assignments)
    
    correct1 = np.zeros(len(allprobs))
    correct2 = np.zeros(len(allprobs))
    correctboth = np.zeros(len(allprobs))
    for i in range(0, len(allprobs)):
        
        if len(allassignments[i]) < 1:
            continue
        
        best = allassignments[i][0]
        correct1[i] = sum(best[:,0] == best[:,1])
        
        if len(allassignments[i]) > 1:
            second = allassignments[i][1]
            correct2[i] = sum(second[:,0] == second[:,1])
            
            cb = 0
            for b in best:
                if b[0] == b[1]:
                    for s in second:
                        if np.array_equal(b, s): cb += 1
            correctboth[i] = cb
            
    assignmentscorrect = np.array([correct1, correct2, correctboth]).T
    return np.array(allprobs), allassignments, assignmentscorrect