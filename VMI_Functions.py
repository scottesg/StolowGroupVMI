import os
import glob
import numpy as np
import scipy as sp
import re
import imutils
import imageio.v2 as imio
import matplotlib.pyplot as plt
from timeit import default_timer as timer
from scipy.signal import fftconvolve

import DAVIS

# Maps file types to a numerical value
# which is the decimal equivalent of the binary code peb
# where p=pump, e=probe, b=molecular beam
pebmap = {'.img':7, 'img':7,
          '.bgb':6, 'bgb':6, '.bgr':6, 'bgr':6,
          'imgp':5, 'bgbp':4, 'imge':3, 'bgbe':2}

# Extracts the delay in ps from the file name
def str2delay(name):
    t1 = [m.start() for m in re.finditer('\.', name)]
    t2 = [m.start() for m in re.finditer('ps', name)]
    s = name[t1[0]+1:t2[-1]]
    return float(s)

# Converts input scans string into an array of scan indices
def scans2range(scans, scansN):
    if scans[0] == -1: # returns all scans
        return np.arange(1, scansN+1)
    
    s = np.array(())
    for i in scans:
        if np.size(i)==1: # include single scan
            s = np.append(s, i)
        if np.size(i)==2: # include range of scans
            s = np.append(s, np.arange(i[0],i[1]))
        else:
            continue
    return s

# Constructs a dictionary containing all data extracted from the scans
def vmigetstats(path, scans, storeimgs=False):
    
    # Make list of files in path
    files = glob.glob(path+'/*.png')
    if len(files)==0:
        print("No .png files were found in {}".format(path))
        raise SystemExit    
    files.sort()
    
    print("Building database of files...")
    s = timer()
    
    # Get list of delays from file names
    delays = []
    for i in range(0, len(files)):
        fname = files[i]
        delay_ps = str2delay(fname[fname.rindex('/')+1:])
        if delay_ps == None:
            continue
        if not delay_ps in delays:
            delays.append(delay_ps)
    delaysN = len(delays)
    
    # Relic from Andrey's code - why reverse sort?
    if delaysN > 1:
        if delays[0] > delays[1]:
            delays.sort(reverse=True)
        else:
            delays.sort()
    #delays = np.array(delays, dtype=np.float32) # Removed to avoid floating point errors - 20200206
    
    db = []
    scansN = 1
    j = 0
    delay_ps = None
    
    # Build database of files
    for i in range(0, len(files)):
        fname = files[i]
        peb = pebmap[fname[-8:-4]]
        if peb==6:
            tmp = delay_ps
            
        delay_ps = str2delay(fname)
        if delay_ps == None:
            continue
        
        if peb==6:
            if (delay_ps==tmp) or ( ( (delay_ps==delays[-1]) or (delay_ps==delays[0]) ) and ( (tmp==delays[-1]) or (tmp==delays[0]) ) ):
                scansN = scansN + 1
        
        j = j + 1
        db.append({'idx':j, 'del':delay_ps, 'peb':peb, 'scan':scansN, 'fname':fname})
    print("Done. Time elapsed: {}".format(timer() - s));
    
    scans = scans2range(scans, scansN)
    #print(scans)
    
    print("Extracting traces...")
    s = timer()
    
    # Create mask to pull out requested scans
    mask = np.zeros(len(db))
    dbscan = [db[i]['scan'] for i in range(0,len(db))]
    for i in range(0, len(scans)):
        mask = np.logical_or(mask,(dbscan==scans[i]))
    
    # defining structures
    traces = [np.empty((0,3)) for i in range(0,7)]
    
    weights = {7:np.zeros(delaysN),
               6:np.zeros(delaysN),
               5:np.zeros(delaysN),
               4:np.zeros(delaysN),
               3:np.zeros(delaysN),
               2:np.zeros(delaysN)}
    
    avg = {7:0,
           6:0,
           5:0,
           4:0,
           3:0,
           2:0}
    
    stk = {7:[0.0 for i in range(0, delaysN)],
           6:[0.0 for i in range(0, delaysN)]}
    
    if storeimgs:
        dim = np.shape(imio.imread(db[0]['fname']))
        imgs = np.zeros((delaysN, np.size(scans), dim[0], dim[1]))
    
    niter = 0
    nimg = np.size(mask[mask==True])
    
    # Read in and store each requested image:
    for i in np.where(mask==True)[0]:
        if (niter % 50 == 0):
            print("#{} - ({}/{})".format(i, niter, nimg))
        
        case = db[i]['peb']
        data = imio.imread(db[i]['fname']).astype('float64')
        #idelay = np.where(delays==db[i]['del'])[0][0]; # Removed since delays is now a list - 20200206
        idelay = delays.index(db[i]['del']) # index of current delay
        j = delaysN * (db[i]['scan'] - 1) + idelay; # scan and delay index
        
        # (scan/delay index, image sum, image number)
        nrow = (np.double(j) + 1, np.sum(data), np.double(i) + 1)
        traces[7 - case] = np.vstack((traces[7 - case], nrow));
        
        # count number of images of each type/delay for averaging
        weights[case][idelay] = weights[case][idelay] + 1
        avg[case] = avg[case] + np.double(data)
        
        if case==7 or case==6: # img or bgb
            stk[case][idelay] = stk[case][idelay] + np.double(data)
            if storeimgs:
                iscan = np.where(scans==dbscan[i])[0][0]
                imgs[idelay, iscan] = imgs[idelay, iscan] + (2 * case - 13)*np.double(data)
        niter = niter + 1

    # perform averaging
    for i in weights:
        if not np.sum(weights[i]) == 0:
            avg[i] = avg[i] / np.sum(weights[i])
    for i in range(0, delaysN): 
        if not weights[7][i] == 0:
            stk[7][i] = stk[7][i] / weights[7][i]
        if not weights[6][i] == 0:
            stk[6][i] = stk[6][i] / weights[6][i]
    
    print("Done. Time elapsed: {}".format(timer() - s))
            
    print("Finding image centre...")
    s = timer()
    
    data = avg[7]
    if not len(avg[6])==1:
        data = data - avg[6]
        
    conv = fftconvolve(data, data)
    (x0, y0) = np.where(conv==np.max(conv))
    x0 = int(x0[0]/2)
    y0 = int(y0[0]/2)
    print("Done. Time elapsed: {}".format(timer() - s))
    print("Centre: {}, {}".format(x0,y0))


    # notes below are differences from Matlab version
    stats = {'folder':path,
             'subfolder':path[path.rindex('/')+1:],
             'delays':delays,
             'scans':scans,
             'imsize':np.shape(avg[7]),
             'vmicentre':[x0,y0],
             'traces':traces,
             'imavgs':avg,
             'imstks':stk,
             'weights':weights, # included all not just img and bgb
             'db':db, # fname is the full path
             'storeimgs':storeimgs}
    
    if storeimgs:
        stats['imgs'] = imgs.astype('int') # img and bgb reversed
             
    return stats


# Plots yield to pick out good scans
def vmiplottraces(stats):
    
    traces = stats['traces']
    pname = stats['subfolder']
    
    fig, ax = plt.subplots()
    ax.set(xlabel = "image #", ylabel = "total e yield", title = pname)
    ax.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
    markers = ['.','.','s','o','x','+']
    labels = ['111', '110', '101', '100', '011', '010']
    plotparams = {'lw':1, 'ms':5, 'mfc':'none'}
    
    for i in range(0, 7):
        if np.size(traces[i]) > 0:
            ax.plot(traces[i][:,2], traces[i][:,1], label=labels[i],
                    marker=markers[i], **plotparams)
            
    leg = ax.legend()
    leg.set_draggable(1)
    
    
# Plots various background subtractions
def vmiplotaverages(stats):
    
    img = stats['imavgs'][pebmap['img']]
    bgb = stats['imavgs'][pebmap['bgb']]
    imgp = stats['imavgs'][pebmap['imgp']]
    bgbp = stats['imavgs'][pebmap['bgbp']]
    imge = stats['imavgs'][pebmap['imge']]
    bgbe = stats['imavgs'][pebmap['bgbe']]
    subfolder = stats['subfolder']
    
    isimgp = np.size(imgp) > 1
    isimge = np.size(imge) > 1
    
    
    # Figure 1
    l = 1 + isimgp + isimge
    fig1, ax1 = plt.subplots(l,3)
    fig1.suptitle(subfolder + ": average images (1)")
    if l==1:
        ax1[0].imshow(img); ax1[0].set(title="img"); ax1[0].axis('off');
        ax1[1].imshow(bgb); ax1[1].set(title="bgb"); ax1[1].axis('off');
        ax1[2].imshow(img-bgb); ax1[2].set(title="img-bgb"); ax1[2].axis('off');
    else:
        ax1[0,0].imshow(img); ax1[0,0].set(title="img"); ax1[0,0].axis('off');
        ax1[0,1].imshow(bgb); ax1[0,1].set(title="bgb"); ax1[0,1].axis('off');
        ax1[0,2].imshow(img-bgb); ax1[0,2].set(title="img-bgb"); ax1[0,2].axis('off');
        if isimgp:
            ax1[1,0].imshow(imgp); ax1[1,0].set(title="imgp"); ax1[1,0].axis('off');
            ax1[1,1].imshow(bgbp); ax1[1,1].set(title="bgbp"); ax1[1,1].axis('off');
            ax1[1,2].imshow(imgp-bgbp); ax1[1,2].set(title="imgp-bgbp"); ax1[1,2].axis('off');
        if isimge:
            ax1[2,0].imshow(imge); ax1[2,0].set(title="imge"); ax1[2,0].axis('off');
            ax1[2,1].imshow(bgbe); ax1[2,1].set(title="bgbe"); ax1[2,1].axis('off');
            ax1[2,2].imshow(imge-bgbe); ax1[2,2].set(title="imge-bgbe"); ax1[2,2].axis('off');
    
    # Figure 2
    fig2, ax2 = plt.subplots(2,2)
    fig2.suptitle(subfolder + ": average images (2)")
    ax2[0,0].imshow(img-bgb); ax2[0,0].set(title="img-bgb"); ax2[0,0].axis('off');
    if isimgp:
        ax2[0,1].imshow(img-bgb-imgp+bgbp);
        ax2[0,1].set(title="(img-bgb)-(imgp-bgbp)"); ax2[0,1].axis('off');
    if isimge:
        ax2[1,0].imshow(img-bgb-imge+bgbe);
        ax2[1,0].set(title="(img-bgb)-(imge-bgbe)"); ax2[1,0].axis('off');
    if isimgp and isimge:
        ax2[1,1].imshow(img-bgb-imgp+bgbp-imge+bgbe);
        ax2[1,1].set(title="(img-bgb)-(imgp-bgbp)-(imge-bgbe)");
        ax2[1,1].axis('off');
    
    # Figure 3
    centre = np.flip(stats['vmicentre'])
    fig3, ax3 = plt.subplots()
    fig3.suptitle(subfolder + ": show image centre")
    ax3.imshow(img-bgb-imgp+bgbp)
    ax3.set(title="(img-bgb)-(imgp-bgbp)")
    c1 = plt.Circle(centre, 220, facecolor='none', edgecolor='red'); ax3.add_patch(c1);
    c2 = plt.Circle(centre, 150, facecolor='none', edgecolor='red'); ax3.add_patch(c2);
    c3 = plt.Circle(centre, 100, facecolor='none', edgecolor='red'); ax3.add_patch(c3);
    ax3.axvline(x=centre[0], color='red', ls=':')
    ax3.axhline(y=centre[1], color='red', ls=':')
            
# Subtract the chosen background
def vmibgsubtr(stats, peb):
    print("Subtracting background...")
    s = timer()
    
    delaysN = np.size(stats['delays'])
    imsub = stats['imstks'][pebmap['img']]
    
    # no subtraction
    if peb == '000':
        stats['imstks']['difstk'] = imsub
        imav = np.zeros(stats['imsize'])
        for i in range(0, delaysN):
            imav = imav + imsub[i] / delaysN
        stats['imavgs']['difstk'] = imav
        return stats
    
    for i in range(0, delaysN):
        
        if int(peb[0]):
            imsub[i] = imsub[i] - (stats['imavgs'][pebmap['imgp']] - stats['imavgs'][pebmap['bgbp']])
        if int(peb[1]):
            imsub[i] = imsub[i] - (stats['imavgs'][pebmap['imge']] - stats['imavgs'][pebmap['bgbe']])
        if int(peb[2]):
            imsub[i] = imsub[i] - stats['imstks'][pebmap['bgb']][i]
    
    imav = np.zeros(stats['imsize'])
    
    for i in range(0, delaysN):
        imav = imav + imsub[i] / delaysN
        
    stats['imstks']['difstk'] = imsub
    stats['imavgs']['difstk'] = imav
    
    print("Done. Time elapsed: {}".format(timer() - s))
    return stats

# Crop images around centre
def vmicropdifstk(stats, cropsize=-1):
    print("Cropping images...")
    s = timer()
    
    [x0, y0] = stats['vmicentre']
    [sx, sy] = stats['imsize']
    delaysN = np.size(stats['delays'])
    
    hc = 0
    if np.size(cropsize)==1:
        hc = np.min([x0-1, sx-x0, y0-1, sy-y0])
    else:
        hc = np.min(cropsize) / 2
    
    im = 'imstks'
    df = 'difstk'
    av = 'imavgs'
    for i in range(0, delaysN):
        stats[im][df][i] = stats[im][df][i][x0-hc:x0+hc, y0-hc+2:y0+hc+2]
    
    stats[av][df] = stats[av][df][x0-hc:x0+hc, y0-hc+2:y0+hc+2]
    stats['imsize_crop'] = np.shape(stats[im][df][0])

    print("Done. Time elapsed: {}".format(timer() - s))
    return stats

# Determines the rotation angle for alignment and applies the rotation
def vmirotate(stats, angle, plotalign=False, lockangle=False):
    
    print("Determining image rotation angle..")
    s = timer()
    
    if lockangle:
        print("Images will be rotated by {} deg *WITHOUT OPTIMIZATION*".format(angle))
        for i in range(0, np.size(stats['delays'])):
            stats['imstks']['difstk'][i] = imutils.rotate(stats['imstks']['difstk'][i], angle)
        stats['imavgs']['difstk'] = imutils.rotate(stats['imavgs']['difstk'], angle)
        stats['angle'] = angle
        return stats
    
    d = stats['imsize_crop'][0]
    im = stats['imavgs']['difstk']
    imin = im + np.flipud(np.fliplr(im)) # Symmetrize first
    ROI = DAVIS.create_circular_mask(d, d, rin=0, rout=150) # Get mask between rin and rout
    delaysN = np.size(stats['delays'])
    
    align = np.empty((0, 2))
    # Step through angles, rotate, and compare top and bottom
    for ang in np.arange(angle-30, angle+30, 0.5):
        imr = imutils.rotate(imin, ang)
        imr = imr*ROI
        al = np.mean((imr[:int(d/2),:] - np.flipud(imr[int(d/2):,:])) ** 2)
        align = np.vstack((align, [ang, al]))
    
    if plotalign: # Visualize curve for testing
        plt.plot(align[:,0],align[:,1], marker='.')
    
    # Get best angle
    rotanglei = np.where(align[:,1]==np.min(align[:,1]))
    rotangle = align[rotanglei[0],0][0]
    
    print("Done. Time elapsed: {}".format(timer() - s))
    print("Images will be rotated by {} deg".format(rotangle))
    
    # Rotate the images
    for i in range(0, delaysN):
        stats['imstks']['difstk'][i] = imutils.rotate(stats['imstks']['difstk'][i], rotangle)
        
    stats['imavgs']['difstk'] = imutils.rotate(stats['imavgs']['difstk'], rotangle)
    stats['angle'] = rotangle
    
    return stats

# Apply the inverse Able transform using the DAVIS algorithm
def vmiiabel_stk(stats, msize, angstep, radstep, smoothing):
    print("Inverting images...")
    s = timer()
    
    delays = stats['delays']
    delaysN = np.size(delays)
    
    inlen = stats['imsize_crop'][0]
    rlen = int(inlen/(2*radstep))
    
    r = radstep*np.linspace(0, rlen-1, rlen)
    
    # Get the necessary matrices for DAVIS
    (LMatrix, ms) = DAVIS.loadmats(msize, angstep, radstep, 180, inlen)
    
    Ir = np.zeros((delaysN, rlen))
    iminstk = np.zeros((delaysN, inlen, inlen))
    imout = np.zeros((delaysN, inlen, inlen))
    iminpol = np.zeros((delaysN, 180, int(inlen/(2*radstep))))
    imoutpol = np.zeros((delaysN, 180, int(inlen/(2*radstep))))
    
    # Smooth each image, symmetrize, and transform
    for i in range(0, delaysN):
        imin = stats['imstks']['difstk'][i]
        imin = sp.ndimage.gaussian_filter(imin,(smoothing,smoothing))
        imin = ((imin + np.fliplr(imin)) + 
                np.flipud((imin + np.fliplr(imin)))) / 4
        iminstk[i] = imin
        (f, delta, iminpol[i], imoutpol[i], imout[i]) = DAVIS.transform(imin, LMatrix, ms, angstep, radstep)
        Ir[i] = f[0]*r
        print("{}\t{}ps\t{}\t{}".format(i, delays[i], np.sum(imin), np.sum(Ir[i,:])))
    
    stats['imstks']['iminstk'] = iminstk
    stats['imstks']['idifstk'] = imout
    stats['imstks']['iminpol'] = iminpol
    stats['imstks']['imoutpol'] = imoutpol
    stats['Ir'] = Ir
    stats['r'] = r
    
    print("Done. Time elapsed: {}".format(timer() - s))
    
    return stats

#%%
    
def vmical(file):

    if not os.path.isfile(file):
        print("Folder {} not found".format(file))
        raise SystemExit;
    
    fid = open(file, 'r')
    lines = fid.readlines()
    fid.close()
    
    fmt = lines[0][1:-1]
    
    K = 0
    if fmt=='r1,r2,dKE':
        r1 = float(lines[1][:-1])
        r2 = float(lines[2][:-1])
        dE = float(lines[3][:lines[3].index(' ')])
        K = dE / (r2**2 - r1**2)
    else:
        print("Format not supported (yet)")
    
    return K
     
