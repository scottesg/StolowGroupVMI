import numpy as np
import os
import matplotlib.pyplot as plt
from timeit import default_timer as timer
from matplotlib.ticker import FormatStrFormatter
import keyboard

from VMI3D_IO import imgfromfile, readroiframe
from VMI3D_Functions import pts2img, extractrois
from VMI3D_Centroiding import centroidFrameROIs

EVT_PATH = r"C:\Program Files\EVT\eSDK\Examples\EVT_Py/"

TESTSOURCE_ROI = r"C:\Users\Scott\Python\VMI\documents\DSC\20250528_202054\run2/roisposb1.single"
TESTSOURCE_IMG = r"C:\Users\Scott\Python\VMI\documents\DSC\testimages\run26/evtdata1.uint8"

SOURCETYPE_CAM = 1
SOURCETYPE_ROI = 2
SOURCETYPE_IMG = 3

DISPLAY_FILTERED_IMAGE = True
IMAGEDIM = 512
ROIDIM = 31
MAXROIS = 20
IMAGEBLANK = np.zeros((IMAGEDIM, IMAGEDIM))

FONTSIZE_T = 15
FONTSIZE_LB = 12
FONTSIZE_TK = 12

HITCOUNTWINDOW = 100

EXTROI_SMOOTH = 5
EXTROI_SIZE = 5
EXTROI_THRESH = 0.2

FRAMECTR_SMOOTH = 5
FULLCTR_SMOOTH = 2
PAUSETIME = 0.1

HISTBINS = np.arange(0.5, 15.5)

class LivePreview:
    
    def __init__(self):
        
        self.isRunning = False
        self.source = None
        self.framenum = 1
        self.hitcount = []
        self.hitcounttime = []
        self.ctrs = []
        self.timezero = timer()
        
        fig = plt.figure(figsize=(14,9))
               
        ax1 = plt.subplot2grid((2, 3), (0, 0))
        self.draw_rawimage(ax1, None)
        
        ax2 = plt.subplot2grid((2, 3), (0, 1))
        self.draw_framectrs(ax2, None)
        
        ax3 = plt.subplot2grid((2, 3), (0, 2))
        self.draw_fullctrs(ax3, None)
        
        ax4 = plt.subplot2grid((2, 3), (1, 0))
        self.draw_hitcountdist(ax4)
        
        ax5 = plt.subplot2grid((2, 3), (1, 1), colspan=2)
        self.draw_hitcountplot(ax5)
        
        fig.subplots_adjust(wspace=0.22, hspace=0.15, left=0.08, right=0.95, top=0.95, bottom=0.08)
        
        self.fig = fig
        self.axs = [ax1, ax2, ax3, ax4, ax5]
        
    def draw_rawimage(self, ax, img):
        
        ax.cla()
        
        if img is None:
            ax.imshow(IMAGEBLANK)
        else:
            ax.imshow(img)
        
        if DISPLAY_FILTERED_IMAGE:
            title = "Filtered Image"
        else:
            title = "Raw Image"
        
        ax.axes.get_xaxis().set_visible(False)
        ax.axes.get_yaxis().set_visible(False)
        ax.set_title(title, fontsize=FONTSIZE_T)
        ax.set_aspect(1/ax.get_data_ratio())
        ax.tick_params(labelsize=FONTSIZE_TK)
        
    def draw_framectrs(self, ax, ctrs):
        
        ax.cla()
        
        ctrnum = 0
        
        if (ctrs is None) or (len(ctrs)==0):
            ax.imshow(IMAGEBLANK)
        else:
            ctrnum = len(ctrs)
            img = pts2img(ctrs, IMAGEDIM, FRAMECTR_SMOOTH)
            ax.imshow(img)

        ax.axes.get_xaxis().set_visible(False)
        ax.axes.get_yaxis().set_visible(False)
        ax.set_title("Frame Centroids [N=%d]"%ctrnum, fontsize=FONTSIZE_T)
        ax.set_aspect(1/ax.get_data_ratio())
        ax.tick_params(labelsize=FONTSIZE_TK)
        
    def draw_fullctrs(self, ax, ctrs):
        
        ax.cla()
        
        if (ctrs is None) or (len(ctrs)==0):
            ax.imshow(IMAGEBLANK)
        else:
            img = pts2img(ctrs, IMAGEDIM, FULLCTR_SMOOTH)
            ax.imshow(img)
        
        ax.axes.get_xaxis().set_visible(False)
        ax.axes.get_yaxis().set_visible(False)
        ax.set_title("Cumulative Centroids", fontsize=FONTSIZE_T)
        ax.set_aspect(1/ax.get_data_ratio())
        ax.tick_params(labelsize=FONTSIZE_TK)
    
    def draw_hitcountdist(self, ax):
        
        ax.cla()
        
        ax.hist(self.hitcount, HISTBINS, density=True,
                edgecolor='black', linewidth=1.5)
        ax.set_xticks(np.arange(1, HISTBINS[-1], 3))
        ax.set_xticks(np.arange(1, HISTBINS[-1], 1), minor=True)
        ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
        ax.set_ylabel("Counts (density)", fontsize=FONTSIZE_LB)
        ax.set_xlabel("Centroids", fontsize=FONTSIZE_LB)
        ax.set_title("Hit Count Distribution", fontsize=FONTSIZE_T)
        ax.tick_params(labelsize=FONTSIZE_TK)
        
    def draw_hitcountplot(self, ax):
        
        ax.cla()
        
        subhc = self.hitcount
        subt = self.hitcounttime
        if self.framenum > HITCOUNTWINDOW:
            subhc = self.hitcount[-HITCOUNTWINDOW:]
            subt = self.hitcounttime[-HITCOUNTWINDOW:]
                
        ax.plot(subt, subhc, color='black', lw=2, marker='.')
        ax.set_ylabel("Centroids", fontsize=FONTSIZE_LB)
        ax.set_xlabel("Time (s)", fontsize=FONTSIZE_LB)
        ax.set_title("Hit Count Over Time", fontsize=FONTSIZE_T)
        ax.tick_params(labelsize=FONTSIZE_TK)
    
    def set_source(self, srctype, source):
        
        if (srctype==SOURCETYPE_ROI) or (srctype==SOURCETYPE_IMG):        
            try:
                self.source = open(source)
            except:
                print("ERROR: Unable to open image source")
                self.isRunning = False
            else:
                self.isRunning = True
                self.srctype = srctype
        
        elif srctype==SOURCETYPE_CAM:
            cwd = os.getcwd()
            os.chdir(EVT_PATH)
            from EVT_Py import EVT_Py, EVT_Util
            #os.chdir(cwd)
            
            evt_context = EVT_Py.EvtContext()
            
    def get_next(self):
        
        framedata = None
        
        if self.srctype==SOURCETYPE_ROI:
            try:
                framedata = readroiframe(self.source, self.framenum,
                                         ROIDIM, MAXROIS)
            except:
                print("ERROR: No data from source")
                self.isRunning = False
            else:
                self.framenum += 1
        elif self.srctype==SOURCETYPE_IMG:
            try:
                framedata = imgfromfile(self.source, IMAGEDIM)
            except:
                print("ERROR: No data from source")
                self.isRunning = False
            else:
                self.framenum += 1
        else:
            print("Source type not supported")
            self.isRunning = False
        
        return framedata
        
    def update(self, framedata):
        
        img = None
        
        if self.srctype==SOURCETYPE_IMG:
            img = framedata[0]
            xpos, ypos, rois, nrois, imgs = extractrois(img, EXTROI_SMOOTH,
                                                        EXTROI_SIZE, EXTROI_THRESH,
                                                        MAXROIS, ROIDIM)
            if DISPLAY_FILTERED_IMAGE:
                img = imgs
            
        elif self.srctype==SOURCETYPE_ROI:
            xpos, ypos, rois, nrois = framedata
        
        self.hitcount.append(nrois)
        self.hitcounttime.append(timer()-self.timezero)
        
        ctrs = centroidFrameROIs(xpos, ypos, rois, int(nrois))
        self.ctrs.extend(ctrs)
        
        self.draw_rawimage(self.axs[0], img)
        self.draw_framectrs(self.axs[1], np.array(ctrs))
        self.draw_fullctrs(self.axs[2], np.array(self.ctrs))
        self.draw_hitcountdist(self.axs[3])
        self.draw_hitcountplot(self.axs[4])
        
        plt.pause(PAUSETIME)
        
    def reset(self):
        
        plt.pause(0.5)
        
        self.framenum = 1
        self.hitcount = []
        self.hitcounttime = []
        self.ctrs = []
        self.timezero = timer()
        
    def pause(self):
        
        plt.pause(0.5)
        paused = True
        
        while paused:
            if keyboard.is_pressed("p"):
                print("Command: un(p)ause")
                paused = False
                plt.pause(0.4)
            plt.pause(0.1)
    
    def print_commands(self):
        
        print("LivePreview Commands:")
        print("(e)xit")
        print("(r)eset")
        print("(p)ause [or un(p)ause]")
    
    def cleanup(self):
        
        if self.source:
            self.source.close()
            
if __name__ == "__main__":

    pv = LivePreview()

    #pv.set_source(SOURCETYPE_ROI, TESTSOURCE_ROI)
    pv.set_source(SOURCETYPE_IMG, TESTSOURCE_IMG)
    #pv.set_source(SOURCETYPE_CAM, None)

    pv.print_commands()
    
    while pv.isRunning:
        
        framedata = pv.get_next()
        
        if framedata:
            pv.update(framedata)
            
        if keyboard.is_pressed("e"):
            print("Command: (e)xit")
            pv.isRunning = False
        
        if keyboard.is_pressed("r"):
            print("Command: (r)eset")
            pv.reset()
            
        if keyboard.is_pressed("p"):
            print("Command: (p)ause")
            pv.pause()
            
    pv.cleanup()
    