#%% Imports

import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as ss
import scipy.interpolate as interpolate
from scipy import optimize
import sgolay2

#%% Load

path = r"C:\Users\Scott\Python\VMI\src\repo\thesis\figsrc/"

dpath = path+"DP_MCPGain_213_20240207_141739_mpolAs000M.npz"
spath = path+"DP_MCPGain_sim_mpol_4_12.npz"
kpath = path+"DP_MCPGain_213_20240207_141739_KAs.npy"

data = np.load(dpath)
sim = np.load(spath)
K = np.load(kpath)

smooth = 9
n0 = 5 # width of momentum slice

ke = 2.05 # eV, energy of ring
keau = ke/27.211 # in a.u.

DCSSP = data['mpol']
phi = data['phi']
theta = data['theta']
mom = data['mom']
argp = np.argmin(abs(np.sqrt(2*keau)-mom))
PHI, THETA, PP = np.meshgrid(phi, theta, mom, indexing="ij")

DCSSPsim = sim['mpol']
phisim = sim['phi']
thetasim = sim['theta']
momsim = sim['mom']
argpsim = np.argmin(abs(np.sqrt(2*keau)-momsim))
PHIsim, THETAsim, PPsim = np.meshgrid(phisim, thetasim, momsim, indexing="ij")

#%% Slices

sm = np.sum(DCSSP[:,:,argp-n0:argp+n0+1], axis=2) / (2*n0+1)

w = mom[1]-mom[0]
ws = momsim[1]-momsim[0]

n0sim = int(np.round(n0*w/ws))
smsim = np.sum(DCSSPsim[:,:,argpsim-n0sim:argpsim+n0sim+1], axis=2) / (2*n0sim+1)

if smooth > 0:
    # smooth the edges in phi
    sm[0] = ss.savgol_filter(sm[0], 11, 1)
    sm[1] = ss.savgol_filter(sm[1], 11, 1)
    sm[2] = ss.savgol_filter(sm[2], 11, 1)
    sm[-1] = ss.savgol_filter(sm[-1], 11, 1)
    sm[-2] = ss.savgol_filter(sm[-2], 11, 1)
    sm[-3] = ss.savgol_filter(sm[-3], 11, 1)
    
    smsim[0] = ss.savgol_filter(smsim[0], 11, 1)
    smsim[1] = ss.savgol_filter(smsim[1], 11, 1)
    smsim[2] = ss.savgol_filter(smsim[2], 11, 1)
    smsim[-1] = ss.savgol_filter(smsim[-1], 11, 1)
    smsim[-2] = ss.savgol_filter(smsim[-2], 11, 1)
    smsim[-3] = ss.savgol_filter(smsim[-3], 11, 1)
    
    # full smoothing
    sphere = sgolay2.SGolayFilter2(smooth, 1)(sm)
    spheresim = sgolay2.SGolayFilter2(smooth, 1)(smsim)
else:
    sphere = sm
    spheresim = smsim

argt = np.argmin(abs(90-theta))

sphereNorm = sphere/np.average(sphere[:,argt-2:argt+3], axis=1)[:,np.newaxis]
spheresimNorm = spheresim/np.average(spheresim[:,argt-2:argt+3], axis=1)[:,np.newaxis]

corr = sphere / spheresim
#corr = sphereNorm/spheresimNorm

#%% Fit Damping Function

argphi1 = np.argmin(np.absolute(75-phi))
argphi2 = np.argmin(np.absolute(105-phi))

argtheta90 = np.argmin(np.absolute(90-theta))
argtheta270 = np.argmin(np.absolute(270-theta))

argtheta1 = np.argmin(np.absolute(165-theta))
argtheta2 = np.argmin(np.absolute(195-theta))

# theta range
thetaslice = np.average(corr[:,argtheta1:argtheta2+1], axis=1)

# phi range, from theta 90 to 270
phislice = np.average(corr[argphi1:argphi2+1,argtheta90:argtheta270+1], axis=0)

# interpolate
angle = np.arange(phi.min()+90, phi.max()+90.5, 0.5)   
f1 = interpolate.interp1d(phi+90, thetaslice)
y1 = f1(angle)
f2 = interpolate.interp1d(theta[argtheta90:argtheta270+1], phislice)
y2 = f2(angle)

# fit response to damping function
def damp(r, a, a0):
    return a0*np.exp(-r**(3/2)*a)
func0 = lambda x,r,y: (damp(r, x[0], x[1]) - y)**2

guess0 = [0.5, 1] # a, a0
ang2r = np.sqrt(2*keau)/K # rmax
rpix = abs(np.sin(np.radians(angle))) * ang2r # convert angle into radius (pixels)
args = (rpix, 0.5*(y1+y2)) # fit to average of both x and y
p0 = optimize.least_squares(func0, x0=guess0, args=args)

# save
rint = np.linspace(0.1, 174, 10000)
np.savez("DP_MCPGain_out_rc.npz", Rcorr=damp(rint, *p0.x), R=rint)

#%% Start Figure

figmode = 3
fig = plt.figure()
fig.set_tight_layout(True)
col1 = 'black'
col2 = 'white'
pos1 = 0.03
pos2 = 0.96
letsize = 15

if figmode==1:
    
    FSt = 12
    FSlb = 10
    FSlg = 6
    FStk = 10
    
    ax1 = plt.subplot2grid((6, 2), (0, 0), rowspan=2)
    ax2 = plt.subplot2grid((6, 2), (0, 1), rowspan=2)
    ax3 = plt.subplot2grid((6, 2), (2, 0), rowspan=2)
    ax4 = plt.subplot2grid((6, 2), (2, 1), rowspan=2)
    ax5 = plt.subplot2grid((6, 2), (4, 0), rowspan=2)
    ax6 = plt.subplot2grid((6, 2), (4, 1))
    ax7 = plt.subplot2grid((6, 2), (5, 1))
    plt.gcf().set_size_inches(8, 8)

elif figmode==2:

    FSt = 12
    FSlb = 10
    FSlg = 10
    FStk = 10
    
    ax1 = plt.subplot2grid((4, 4), (0, 0))
    ax2 = plt.subplot2grid((4, 4), (0, 1))
    ax3 = plt.subplot2grid((4, 4), (1, 0))
    ax4 = plt.subplot2grid((4, 4), (1, 1))
    ax5 = plt.subplot2grid((4, 4), (0, 2), rowspan=2, colspan=2)
    ax6 = plt.subplot2grid((4, 4), (2, 0), rowspan=2, colspan=2)
    ax7 = plt.subplot2grid((4, 4), (2, 2), rowspan=2, colspan=2)

elif figmode==3:
    
    FSt = 20
    FSlb = 15
    FSlg = 12
    FStk = 12
    
    ax1 = plt.subplot2grid((3, 4), (0, 0))
    ax2 = plt.subplot2grid((3, 4), (0, 1))
    ax3 = plt.subplot2grid((3, 4), (1, 0))
    ax4 = plt.subplot2grid((3, 4), (1, 1))
    ax5 = plt.subplot2grid((3, 4), (0, 2), rowspan=2, colspan=2)
    ax6 = plt.subplot2grid((3, 4), (2, 0), colspan=2)
    ax7 = plt.subplot2grid((3, 4), (2, 2), colspan=2)

#%% Panel 1

p1 = ax1.pcolor(PHI[:,:,argp], THETA[:,:,argp], sphere/np.percentile(sphere, 95), cmap="seismic", vmax=1, vmin=0)
ax1.hlines((90, 250), 0, 180, color='black', ls='-', lw=2)
ax1.set_xticks([0, 45, 90, 135, 180])
ax1.set_yticks([0, 90, 180, 270, 360])
ax1.set_xlabel("$\phi$' (\N{DEGREE SIGN})", fontsize=FSlb)
ax1.set_ylabel("$\\theta$' (\N{DEGREE SIGN})", fontsize=FSlb)

#ax1.axes.get_xaxis().set_visible(False)

cm = fig.colorbar(p1)
cm.set_label("Normalized Yield", fontsize=FSlb)

ax1.tick_params(labelsize=FStk)

t1 = ax1.text(pos1, pos2, '(a)', ha='left', va='top', color=col1, transform=ax1.transAxes, fontsize=letsize)
t1.set_bbox(dict(facecolor=col2, alpha=0.8, edgecolor=col2))

#%% Panel 2

p2 = ax2.pcolor(PHI[:,:,argp], THETA[:,:,argp], sphereNorm/sphereNorm.max(), cmap="seismic", vmin=0.2)
ax2.hlines((theta[argt-2], theta[argt+2]), 0, 180, color='black', ls='--', lw=1)
ax2.set_xticks([0, 45, 90, 135, 180])
ax2.set_yticks([0, 90, 180, 270, 360])
ax2.set_xlabel("$\phi$' (\N{DEGREE SIGN})", fontsize=FSlb)
ax2.set_ylabel("$\\theta$' (\N{DEGREE SIGN})", fontsize=FSlb)

#ax2.axes.get_xaxis().set_visible(False)
#ax2.axes.get_yaxis().set_visible(False)

cm = fig.colorbar(p2)
cm.set_label("Normalized Yield", fontsize=FSlb)

ax2.tick_params(labelsize=FStk)

t2 = ax2.text(pos1, pos2, '(b)', ha='left', va='top', color=col1, transform=ax2.transAxes, fontsize=letsize)
t2.set_bbox(dict(facecolor=col2, alpha=0.8, edgecolor=col2))

#%% Panel 3

p3 = ax3.pcolor(PHI[:,:,argp], THETA[:,:,argp], spheresim/np.percentile(spheresim, 95), cmap="seismic", vmax=1, vmin=0)
ax3.set_xticks([0, 45, 90, 135, 180])
ax3.set_yticks([0, 90, 180, 270, 360])
ax3.set_xlabel("$\phi$' (\N{DEGREE SIGN})", fontsize=FSlb)
ax3.set_ylabel("$\\theta$' (\N{DEGREE SIGN})", fontsize=FSlb)

cm = fig.colorbar(p3)
cm.set_label("Normalized Yield", fontsize=FSlb)

ax3.tick_params(labelsize=FStk)

t3 = ax3.text(pos1, pos2, '(c)', ha='left', va='top', color=col1, transform=ax3.transAxes, fontsize=letsize)
t3.set_bbox(dict(facecolor=col2, alpha=0.8, edgecolor=col2))

#%% Panel 4

p4 = ax4.pcolor(PHI[:,:,argp], THETA[:,:,argp], spheresimNorm/spheresimNorm.max(), cmap="seismic", vmin=0.7)
ax4.hlines((theta[argt-2], theta[argt+2]), 0, 180, color='black', ls='--', lw=1)
ax4.set_xticks([0, 45, 90, 135, 180])
ax4.set_yticks([0, 90, 180, 270, 360])
ax4.set_xlabel("$\phi$' (\N{DEGREE SIGN})", fontsize=FSlb)
ax4.set_ylabel("$\\theta$' (\N{DEGREE SIGN})", fontsize=FSlb)

cm = fig.colorbar(p4)
cm.set_label("Normalized Yield", fontsize=FSlb)

ax4.tick_params(labelsize=FStk)

t4 = ax4.text(pos1, pos2, '(d)', ha='left', va='top', color=col1, transform=ax4.transAxes, fontsize=letsize)
t4.set_bbox(dict(facecolor=col2, alpha=0.8, edgecolor=col2))

#%% Panel 5

p5 = ax5.pcolor(PHI[:,:,argp], THETA[:,:,argp], corr/corr.max(), cmap="seismic", vmax=1, vmin=0)
ax5.set_xticks([0, 45, 90, 135, 180])
ax5.set_yticks([0, 90, 180, 270, 360])
ax5.set_xlabel("$\phi$' (\N{DEGREE SIGN})", fontsize=FSlb)
ax5.set_ylabel("$\\theta$' (\N{DEGREE SIGN})", fontsize=FSlb)
ax5.hlines((theta[argtheta1], theta[argtheta2]), 0, 180, color='black', ls='--', lw=2)
ax5.vlines((phi[argphi1], phi[argphi2]), 90, 270, color='black', ls='--', lw=2)
ax5.hlines((90, 270), phi[argphi1], phi[argphi2], color='black', ls='--', lw=2)
ax5.vlines((0, 180), theta[argtheta1], theta[argtheta2], color='black', ls='--', lw=2)

cm = fig.colorbar(p5)
cm.set_label("Normalized Ratio", fontsize=FSlb)

t5 = ax5.text(pos1-0.01, pos2+0.02, '(e)', ha='left', va='top', color=col1, transform=ax5.transAxes, fontsize=letsize)
t5.set_bbox(dict(facecolor=col2, alpha=0.8, edgecolor=col2))

#%% Panel 6

ln1 = ax6.plot(theta[argtheta90:argtheta270+1], phislice, label="$\\theta$'", color='red', lw=2)
ax6.set_xlabel("$\\theta$' (\N{DEGREE SIGN})", fontsize=FSlb)
ax6.set_ylabel("Relative Gain (arb)", fontsize=FSlb)
ax6.tick_params(labelsize=FStk)

ax6b = ax6.twiny()
ln2 = ax6b.plot(phi, thetaslice, label="$\phi$'", color='blue', lw=2)
ax6b.set_xlabel("$\phi$' (\N{DEGREE SIGN})", fontsize=FSlb)
ax6b.tick_params(labelsize=FStk)

lns = ln1+ln2
labs = [l.get_label() for l in lns]
ax6.legend(lns, labs, loc='upper center', fontsize=FSlg)

t6 = ax6.text(pos1-0.02, pos2, '(f)', ha='left', va='top', color=col1, transform=ax6.transAxes, fontsize=letsize)

#%% Panel 7

ang180 = np.argmin(abs(180-angle))
ang1 = abs(np.sin(np.radians(angle[ang180:])))
ang2 = abs(np.sin(np.radians(angle[:ang180+1])))

# ax7.plot(ang2*ang2r, 0.5*(y1+y2)[:ang180+1], label="X,Y > 0", color='red', lw=2)
# ax7.plot(ang1*ang2r, 0.5*(y1+y2)[ang180:], label="X,Y < 0", color='blue', lw=2)
# ax7.plot(ang1*ang2r, damp(ang1*ang2r, *p0.x), c="k", ls="--", label="Fit")
ax7.plot(ang1*ang2r, 0.25*((y1+y2)[ang180:] + (y1+y2)[:ang180+1][::-1]), label="Average Gain", color='black', lw=2)
ax7.plot(ang1*ang2r, damp(ang1*ang2r, *p0.x), c="k", ls="--", label="Fit")
ax7.set_xlabel("Radius (pixels)", fontsize=FSlb)
ax7.set_ylabel("Relative Gain (arb)", fontsize=FSlb)
ax7.set_xlim(0, 172)

ax7.legend(loc='upper center', fontsize=FSlg)

ax7.tick_params(labelsize=FStk)

t7 = ax7.text(pos1-0.02, pos2, '(g)', ha='left', va='top', color=col1, transform=ax7.transAxes, fontsize=letsize)