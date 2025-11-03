import numpy as np
import os
import matplotlib.pyplot as plt
from timeit import default_timer as timer
from matplotlib.ticker import FormatStrFormatter
import keyboard
import ctypes
from PIL import Image

from VMI3D_IO import imgfromfile, readroiframe
from VMI3D_Functions import pts2img, extractrois
from VMI3D_Centroiding import centroidFrameROIs

# Import EVT SDK
EVT_PATH = r"C:\Program Files\EVT\eSDK\Examples\EVT_Py/"
os.chdir(EVT_PATH)
from EVT_Py import EVT_Py

TESTSOURCE_ROI = r"C:\Python\testdata\run26/roispos1.single"
TESTSOURCE_IMG = r"C:\Python\testdata\run26/evtdata1.uint8"

SOURCETYPE_CAM_FREE = 1
SOURCETYPE_CAM_TRIG = 2
SOURCETYPE_ROI = 3
SOURCETYPE_IMG = 4

NUM_ALLOCATED_FRAMES = 1
CAM_EXPOSURE_FREE = 20000 #us
CAM_EXPOSURE_TRIG = 900 #us
CAM_FRAMERATE = 10
CAM_OFFSET_X = 552
CAM_OFFSET_Y = 356
CAM_GAIN = 256

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
EXTROI_THRESH = 0.15

FRAMECTR_SMOOTH = 5
FULLCTR_SMOOTH = 2
PAUSETIME = 0.01

HISTBINS = np.arange(0.5, 20.5)

class LivePreview:
    
    def __init__(self):
        
        self.isRunning = False
        self.source = None
        self.context = None
        self.stream_open = False
        self.framenum = 1
        self.hitcount = []
        self.hitcounttime = []
        self.ctrs = []
        self.timezero = timer()
        self.prevtime = self.timezero
        self.frameratelive = 0
        
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
            title = "Filtered Image [%d fps]"%self.frameratelive
        else:
            title = "Raw Image [%d fps]"%self.frameratelive
        
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
        
        elif (srctype==SOURCETYPE_CAM_FREE) or (srctype==SOURCETYPE_CAM_TRIG):
                        
            # Locate camera
            evt_context = EVT_Py.EvtContext()
            device_list = evt_context.list_devices(False)
            num_cameras = len(device_list.dev_infos)
            
            print(f"Number of detected cameras: {num_cameras}")
            for dev in device_list.dev_infos:
                print(f"\tCam ID: {dev.camera_id} IP: {dev.ip_address}")
            
            if num_cameras == 0:
                print("ERROR: Unable to find camera")
                self.isRunning = False
            else:
                # Initialize camera
                first_dev_info = device_list.dev_infos[0]
                print(f"Setting up cam: {first_dev_info.camera_id}")
                open_camera_params = evt_context.create_open_camera_params()
                cam = evt_context.open_camera(first_dev_info, open_camera_params)
                
                self.context = evt_context
                self.source = cam
                self.srctype = srctype
            
                self.setup_camera()
                self.source.open_stream()
                self.stream_open = True

                for _ in range(NUM_ALLOCATED_FRAMES):
                    frame = self.source.allocate_frame()
                    self.source.queue_frame(frame)
                
                self.source.execute_command("AcquisitionStart")
                self.isRunning = True
        
        else:
            print("Source type not supported")
            self.isRunning = False
            
    def setup_camera(self):
        
        self.source.set_param_uint32("Width", IMAGEDIM)
        self.source.set_param_uint32("Height", IMAGEDIM)
        self.source.set_param_uint32("OffsetX", CAM_OFFSET_X)
        self.source.set_param_uint32("OffsetY", CAM_OFFSET_Y)
        self.source.set_param_uint32("Gain", CAM_GAIN)
        self.source.set_param_bool("HCG", True)
        
        if self.srctype==SOURCETYPE_CAM_FREE:
            self.source.set_param_uint32("Exposure", CAM_EXPOSURE_FREE)
            self.source.set_param_uint32("FrameRate", CAM_FRAMERATE)
            self.source.set_enum_str("TriggerMode", "Off")
        elif self.srctype==SOURCETYPE_CAM_TRIG:
            self.source.set_param_uint32("Exposure", CAM_EXPOSURE_TRIG)
            self.source.set_enum_str("TriggerMode", "On")
            self.source.set_enum_str("TriggerSource", "Hardware")
            self.source.set_enum_str("GPI_End_Exp_Event", "Falling_Edge")
        else:
            print("Source type not supported")
            self.isRunning = False
                
    def get_next(self):
        
        framedata = None
        
        try:
            if self.srctype==SOURCETYPE_ROI:
                framedata = readroiframe(self.source, self.framenum,
                                         ROIDIM, MAXROIS)
            
            elif self.srctype==SOURCETYPE_IMG:
                framedata = imgfromfile(self.source, IMAGEDIM)
            
            elif (self.srctype==SOURCETYPE_CAM_FREE) or (self.srctype==SOURCETYPE_CAM_TRIG):
                frame = self.source.get_frame()
                self.source.queue_frame(frame)
                framedata = self.convert_frame(frame)
            else:
                print("Source type not supported")
                self.isRunning = False
        except:
            print("ERROR: No data from source")
            self.isRunning = False
        else:
            self.framenum += 1

        return framedata
    
    def convert_frame(self, frame):
        
        image_mode = "L"
        convert_bit_flag = EVT_Py.EvtBitConvert.EVT_CONVERT_NONE
        convert_format = EVT_Py.EvtPixelFormat.GVSP_PIX_MONO8
        conversion_frame = self.source.allocate_convert_frame(frame.width, frame.height, 
                                                              convert_format, convert_bit_flag,
                                                              EVT_Py.EvtColorConvert.EVT_CONVERT_NONE)
        
        frame.convert(conversion_frame, convert_bit_flag,
                      EVT_Py.EvtColorConvert.EVT_CONVERT_NONE,
                      self.source.get_lines_reorder_handle())

        # Copy the image data to a python managed buffer
        img_bytes = bytes((ctypes.c_char * conversion_frame.buffer_size).from_address(conversion_frame.img_ptr))
    
        # Free the newly allocated conversion frame now that we're done with it
        self.source.release_frame(conversion_frame)
    
        im = Image.frombytes(image_mode, (frame.width, frame.height), img_bytes, 'raw')
        
        return im
        
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
            
        elif (self.srctype==SOURCETYPE_CAM_FREE) or (self.srctype==SOURCETYPE_CAM_TRIG):
            img = np.array(framedata)
            xpos, ypos, rois, nrois, imgs = extractrois(img, EXTROI_SMOOTH,
                                                        EXTROI_SIZE, EXTROI_THRESH,
                                                        MAXROIS, ROIDIM)

        curtime = timer()
        self.frameratelive = round(1.0/(curtime - self.prevtime))
        self.prevtime = curtime
        
        self.hitcount.append(nrois)
        self.hitcounttime.append(curtime-self.timezero)
        
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
            if (self.srctype==SOURCETYPE_ROI) or (self.srctype==SOURCETYPE_IMG):  
                self.source.close()
            elif (self.srctype==SOURCETYPE_CAM_FREE) or (self.srctype==SOURCETYPE_CAM_TRIG):
                if self.stream_open:
                    self.source.execute_command("AcquisitionStop")
                    self.source.close_stream()
                self.context.close_camera(self.source)
            
if __name__ == "__main__":

    pv = LivePreview()
    frames = []

    #pv.set_source(SOURCETYPE_ROI, TESTSOURCE_ROI)
    #pv.set_source(SOURCETYPE_IMG, TESTSOURCE_IMG)
    pv.set_source(SOURCETYPE_CAM_FREE, None)
    #pv.set_source(SOURCETYPE_CAM_TRIG, None)

    pv.print_commands()
    
    while pv.isRunning:
        
        framedata = pv.get_next()
        
        frames.append(framedata)
        
        if framedata:
            pv.update(framedata)
        else:
            print("ERROR: Can't get next frame!")
            pv.isRunning = False
            
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
    