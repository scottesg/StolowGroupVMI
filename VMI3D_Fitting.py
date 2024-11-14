import numpy as np
from scipy.optimize import curve_fit, least_squares
from scipy.stats import poisson

# Fit to gaussian
# x, y: data to fit.
# c: baseline, h: height, pos: x posiiton of gaussian, w: width (sigma)
# czero: lock baseline to zero
# Returns yfit array, fit params, covariance matrix
def fit_gauss(x, y, c, h, pos, w, czero=False):
    if czero:
        fitfunc = gauss_c0
        p0 = [h,pos,w]
    else:
        fitfunc = gauss
        p0 = [c,h,pos,w]
    try:
        fit_results = curve_fit(fitfunc, x, y, p0=p0)
        fit = fit_results[0]
        pcov = fit_results[1]
    except RuntimeError:
        print("Runtime Error in Gaussian Fit: Returning -1...")
        return -1
    return fitfunc(x, *fit), fit, pcov

# Fit to poission function
# x, y: data to fit
# lamb: lambda
# Returns yfit array, fit params, covariance matrix
def fit_poisson(x, y, lamb):
    try:
        fit_results = curve_fit(poisson.pmf, x, y, p0=[lamb])
        fit = fit_results[0]
        pcov = fit_results[1]
    except RuntimeError:
        print("Runtime Error in Poisson Fit: Returning -1...")
        return -1
    return poisson.pmf(x, *fit), fit, pcov

# Fit to half ellipse
# x, y: data to fit
# a, b: params
# fscale: 
# Returns fitted params
def fit_ellipse(x, y, a, b, fscale=1, fixa=False):
    try:
        if fixa:
            def f(b, x, y):
                return np.abs(ellipse(x, a, b) - y) 
            lsq = least_squares(f, [b], loss='linear', f_scale=fscale, args=(x, y))
        else:
            def f(params, x, y):
                a, b = params
                return np.abs(ellipse(x, a, b) - y) 
            lsq = least_squares(f, [a, b], loss='linear', f_scale=fscale, args=(x, y))
    except RuntimeError:
        print("Runtime Error in Ellipse Fit: Returning -1...")
        return -1
    if fixa: return [a, *lsq.x]
    else: return lsq.x

def fit_pollipse(x, y, a, b, pol, bounds=(-np.inf, np.inf), fscale=1, fixa=False, even=False):
    try:
        if fixa:
            x0 = [b, *pol]
            def f(params, x, y):
                return np.abs(pollipse(x, [a, *params], even) - y) 
            lsq = least_squares(f, x0, loss='cauchy', f_scale=fscale,
                                bounds=bounds, args=(x, y))
        else:
            x0 = [a, b, *pol]
            def f(params, x, y):
                return np.abs(pollipse(x, params, even) - y) 
            lsq = least_squares(f, x0, loss='cauchy', f_scale=fscale,
                                bounds=bounds, args=(x, y))
    except RuntimeError:
        print("Runtime Error in Ellipse Fit: Returning -1...")
        return -1
    if fixa: return [a, *lsq.x]
    else: return lsq.x
    
def fit_growth(x, y, x0):
    try:
        def f(params, x, y):
            a, b, c = params
            return np.abs(growth(x, a, b, c) - y) 
        lsq = least_squares(f, x0, loss='linear', args=(x, y))
    except RuntimeError:
        print("Runtime Error in Ellipse Fit: Returning -1...")
        return -1
    return lsq.x

def fit_growth2(x, y, x0):
    try:
        def f(params, x, y):
            a, b, c, d = params
            return np.abs(growth(x, a, b, c, d) - y) 
        lsq = least_squares(f, x0, loss='linear', args=(x, y))
    except RuntimeError:
        print("Runtime Error in Ellipse Fit: Returning -1...")
        return -1
    return lsq.x

def fit_logistic(x, y, x0):
    try:
        def f(params, x, y):
            a, b, c, d = params
            return np.abs(logistic(x, a, b, c, d) - y) 
        lsq = least_squares(f, x0, loss='linear', args=(x, y))
    except RuntimeError:
        print("Runtime Error in Ellipse Fit: Returning -1...")
        return -1
    return lsq.x

def fit_pad(x, y, x0):
    try:
        def f(params, x, y):
            a, b2 = params
            return np.abs(pad(x, a, b2) - y) 
        lsq = least_squares(f, x0, loss='linear', args=(x, y))
    except RuntimeError:
        print("Runtime Error in Ellipse Fit: Returning -1...")
        return -1
    return lsq.x

def dampedcos2(x, d, w, phase):
    y = np.exp(-1*d*x)*np.cos(w*x + phase)**2
    y[x<(np.pi/2-phase)/w] = 0
    return y

def pad(x, a, b2):
    y = a*(1 + b2*legp2(np.cos(x)))
    return y

def legp2(x):
    return 0.5*(3*x**2 - 1)

def growth2(x, a, b, c, d):
    y = a - b*np.exp(-1*c*(x-d))
    return y

def growth(x, a, b, c):
    y = a - np.exp(-1*b*(x-c))
    return y

def logistic(x, a, b, c, d):
    y = a + b/(1 + np.exp(-1*c*(x-d)))
    return y
    
# Half ellipse function
def ellipse(x, a, b):
    y2 = b**2 * (1 - (x**2)/(a**2))
    y = np.emath.sqrt(y2)
    return y

def pollipse(x, params, even=False):
    a = params[0]
    b = params[1]
    pol = params[2:]
    bpoly = b
    for i in range(0, len(pol)):
        bpoly += pol[i]*x**((even+1)*(i+1))
    y2 = (bpoly)**2 * (1 - (x/a)**2)
    y = np.emath.sqrt(y2)
    return y

# Gaussian function
def gauss(x, c, a, x0, sigma):
    return c + a*np.exp(-(x-x0)**2/(2*sigma**2))

# Gaussian function, baseline fixed to zero
def gauss_c0(x, a, x0, sigma):
    return a*np.exp(-(x-x0)**2/(2*sigma**2))

# Multi-gaussian fit
# pl: [c, n*(a, x0, sigma)] for n-gaussian fit
def mgauss(x, pl):
    c = pl[0]
    pl = pl[1:]
    n = len(pl)//3
    return c + sum([gauss(x, 0, *pl[3*i:3*i+3]) for i in range(0, n)])

# Evaluate residual for multi-gaussian fit
def res_mgauss(pl, x, y):
    diff = [mgauss(x[i], pl) - y[i] for i in range(0, len(x))]
    return diff

# Lorentzian function
def lorentz(x, a, x0, gam, c):
    return c + a*gam**2 / (gam**2 + (x - x0)**2)

# Multi-lorentzian fit
# c: baseline
# pl: list of params [a, a0, gamma] for each Lorentzian
def mlorentz(x, c, pl):
    return c + sum([lorentz(x, *pl[i], 0) for i in range(0, len(pl))])

    