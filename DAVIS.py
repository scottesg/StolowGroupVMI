import numpy as np
import scipy as sp
import polarTransform as pt
import imutils
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from scipy.signal import fftconvolve
from scipy.ndimage import gaussian_filter
from timeit import default_timer as timer

LegendreExpansionUpperLimit = 4 # must be even
NumLegendreTerms = int(LegendreExpansionUpperLimit / 2) + 1

# Loads matrices for the DAVIS Abel inversion
# Use this to get the matrices, then pass to DAVIS.transform
# If the matrices have not already been generated and saved,
# use DAVISmats to do so.
#
# s: full size of matrices to load
# dtdeg: angle step size (deg)
#   Degrees per element in the polar image array
# dr: radial step size (pixels)
#   Pixels per element in the polar image array
#
# amax: maximum angle, in deg
# rmax: maximum radius, in pixels
def loadmats(s, dtdeg, dr, amax, rmax):
    
    name = "{}_{}_{}".format(s, int(dtdeg), int(dr*10)) # matrix file name ending
    na = int(amax/dtdeg) # index of angle a (deg)
    nr = int(rmax/dr) # index of radius r (pixels)
    
    print("Loading matricies [{}] up to {} deg and {} pixels".format(name, amax, rmax))
    s = timer()
    
    LMatrix = np.load("DAVIS/Legendre_{}.npy".format(int(dtdeg)))[:NumLegendreTerms, :na] 
    m00 = np.load("DAVIS/m00_{}.npy".format(name))[:nr, :nr]
    m22 = np.load("DAVIS/m22_{}.npy".format(name))[:nr, :nr]
    m20 = np.load("DAVIS/m20_{}.npy".format(name))[:nr, :nr]
    m44 = np.load("DAVIS/m44_{}.npy".format(name))[:nr, :nr]
    m42 = np.load("DAVIS/m42_{}.npy".format(name))[:nr, :nr]
    m40 = np.load("DAVIS/m40_{}.npy".format(name))[:nr, :nr]
    ms = [[m00], [m20, m22], [m40, m42, m44]]
    
    print("Elapsed time: {}".format(timer()-s))
    
    return (LMatrix, ms)

# Performs an inverse Abel transform of the image using the DAVIS algorithm:
# [The Journal of Chemical Physics 148, 194101 (2018); doi: 10.1063/1.5025057]
#    
# Currently only processess the top half of the image (symmetrize image first)
# An option to use the full image could be added?
#
# imin: image to be transformed (assumes square)
# LMatrix, ms: matrices loaded using loadmats
# dtdeg: angular pixel size in degrees
#   Number of degrees per element in the polar image array
#   Smaller = better resolution
# dr: radial pixel size
#   Number of pixels per element in the polar image array
#   Smaller = better resolution
# dtdeg and dr must match the matrices loaded
def transform(imin, LMatrix, ms, dtdeg, dr):
    
    dim = len(imin)
    radius = int(dim/2)
    
    rlen = int(dim/(2*dr)) # radial axis of polar image
    alen = int(180/dtdeg) # angle axis of polar image
    
    if (rlen > len(ms[0][0])):
        print("Matrices are too small (radial axis) for input image.")
        return -1

    if (alen > len(LMatrix[0])):
        print("Matrices are too small (angle axis) for input image.")
        return -1

    # Transform to polar in format [angle, radius]
    impol = pt.convertToPolarImage(imin, finalAngle = np.pi,
                                   finalRadius = radius,
                                   angleSize=alen,
                                   radiusSize=rlen)[0]
    
    # Perform the Legendre expansion of the polar image for each r value:
    apw = np.deg2rad(dtdeg) # Angular pixel width (rad)
    angs = np.linspace(0, np.pi, alen)
    sin = np.abs(np.sin(angs))
    deltas = np.zeros((NumLegendreTerms, rlen))
    for n in range(0, NumLegendreTerms):
        for r in range(0, rlen):
            deltas[n, r] = sum(sin * LMatrix[n] * impol[:,r])
        deltas[n] *= apw * (4*n + 1)/2
        
    # Transform the Legendre coefficients using the M matrices:
    f_values = np.zeros((NumLegendreTerms, rlen))
    for k in range(NumLegendreTerms - 1, -1, -1):
        term = deltas[k]
        for i in range(2 * k + 2, LegendreExpansionUpperLimit + 1, 2):
            term -= np.matmul(ms[int(i / 2)][k][:rlen, :rlen], f_values[int(i / 2)])
        f_values[k] = np.matmul(np.linalg.inv(ms[k][k][:rlen, :rlen]), term)
    
    # Reconstruct the image:
    impolinv = np.zeros((alen, rlen))
    for r in range(0, rlen):
        for t in range(0, alen):
            impolinv[t, r] = np.dot(f_values[:, r], LMatrix[:, t])
            
    # Transform back to Cartesian with the same shape as imin:
    imout_q12 = pt.convertToCartesianImage(impolinv, center ='bottom-middle',
                                           finalAngle = np.pi,
                                           finalRadius = radius, 
                                           imageSize = (radius, dim))[0]
    
    # Reconstruct full image from top half
    imout = np.vstack((np.flipud(imout_q12), imout_q12))
    
    return (f_values, deltas, impol, impolinv, imout)

# Generate a mask to pull out points between two radii to use for determining
# the rotation angle
def create_circular_mask(h, w, center=None, rin=0, rout=None):

    if center is None: # use the middle of the image
        center = [int(w/2), int(h/2)]
    if rout is None: # use the smallest distance between the center and image walls
        rout = min(center[0], center[1], w-center[0], h-center[1])

    Y, X = np.ogrid[:h, :w]
    dist_from_center = np.sqrt((X - center[0])**2 + (Y-center[1])**2)

    mask = np.logical_and(dist_from_center <= rout, dist_from_center >=rin)
    return mask

# Prepares a single image for DAVIS transform
# Performs centre, crop, rotate, smooth, and symmetrize
def prepimage(img, rin=0, rout=500, rotsm=5, imsm=2,
              rotangle=-1, cen=[-1], sym='1111', plot=False):
    
    # Find Centre
    if len(cen)==1:
        conv = fftconvolve(img, img)
        (x0, y0) = np.where(conv==np.max(conv))
        x0 = np.int(x0[0]/2)
        y0 = np.int(y0[0]/2)
        print("Centre: {}, {}".format(x0,y0))
    else:
        x0, y0 = cen
        
    if plot:
        fig, ax = plt.subplots()
        ax.imshow(img)
        circ1 = Circle((y0,x0),800, fill=False, color='red')
        circ2 = Circle((y0,x0),200, fill=False, color='red')
        circ25 = Circle((y0,x0),150, fill=False, color='red')
        circ3 = Circle((y0,x0),20, fill=False, color='red')
        ax.add_patch(circ1)
        ax.add_patch(circ2)
        ax.add_patch(circ25)
        ax.add_patch(circ3)
        ax.set_title("Centre of Image")
    
    # Crop
    [sx, sy] = img.shape
    hc = np.min([x0-1, sx-x0, y0-1, sy-y0])
    img = img[x0-hc:x0+hc, y0-hc:y0+hc]
    
    # Rotate
    if rotangle == -1:
        [sx, sy] = img.shape
        imtmp = img
        ROI = create_circular_mask(sx, sy, rin=rin, rout=rout) # Get mask between rin and rout
        
        if plot:
            fig, ax = plt.subplots()
            ax.imshow(imtmp)
            xx = int(imtmp.shape[0]/2)
            yy = int(imtmp.shape[1]/2)
            circ1 = Circle((xx,yy),rout, fill=False, color='red')
            circ2 = Circle((xx,yy),rin, fill=False, color='red')
            ax.add_patch(circ1)
            ax.add_patch(circ2)
            ax.set_title("Mask For Rotation Angle")
        
        align = np.empty((0, 2))
        for ang in np.arange(0, 90, 0.5): # Step through angles, rotate, and compare top and bottom
            imr = imutils.rotate(imtmp, ang)
            imr = imr*ROI
            al = np.mean((imr[:int(sx/2),:] - np.flipud(imr[int(sx/2):,:])) ** 2)
            align = np.vstack((align, [ang, al]))
        align[:,1] = gaussian_filter(align[:,1], rotsm)
    
        if plot:
            plt.figure()
            plt.plot(align[:,0], align[:,1])
            plt.xlabel("Rotation Angle (deg)")
            plt.title("Determining Rotation Angle")
        
        # Get best angle
        rotanglei = np.where(align[:,1]==np.min(align[:,1]))
        rotangle = align[rotanglei[0],0][0] - 90

    if plot: print("Image will be rotated by {} deg".format(rotangle))    
    img = imutils.rotate(img, rotangle)
    
    if plot:
        plt.figure()
        plt.imshow(img)
        plt.title("Rotated Image")

    # Smooth and symmetrize
    img = sp.ndimage.gaussian_filter(img,(imsm,imsm))
    
    d = int(len(img)/2)
    img[:d,:d] = img[:d,:d]*int(sym[0])
    img[:d,d:] = img[:d,d:]*int(sym[1])
    img[d:,:d] = img[d:,:d]*int(sym[2])
    img[d:,d:] = img[d:,d:]*int(sym[3])
    
    if plot:
        plt.figure()
        plt.imshow(img)
        plt.title("Image with Selected Quadrents")
    
    img = ((img + np.fliplr(img)) + 
           np.flipud((img + np.fliplr(img)))) / np.sum([int(sym[i]) for i in range(0, len(sym))])

    if plot:
        plt.figure()
        plt.imshow(img)
        plt.title("Prepared Image")
    
    return img, [x0, y0]

    
    
    
    
    
    
    
    
    
    