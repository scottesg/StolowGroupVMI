import ctypes
import time
import os
import numpy as np
from VMI3D_IO import readwfm, rawdaqtovolts
import atsapi as ats
from EVT_Py import EVT_Py

def writevmi3dparams(datadir, nstacks, nframes, iexposure, imgsize,
                     roix0, roiy0, storeimgs, storertcs, dclvl, rtclvl):

    # Args: Run driectory, number of stacks, frames/stack, exposure(ms), image size (typically 512),
    # x offset, y offset, store full images? (y/n), store rtcs? (y/n), dc level in data, threshold for sensing ROIs
    
    file = open('V:/data/vmi3drunparamsrtc.txt','w')   
    
    file.write('data_folder %s\n'%datadir)
    file.write('nstacks %d\n'%nstacks)
    file.write('nframes %d\n'%nframes)
    file.write('exposure_us %d\n'%iexposure)
    file.write('imagesize %d\n'%imgsize)
    file.write('roix0 %d\n'%roix0)
    file.write('roiy0 %d\n'%roiy0)
    file.write('storeimgs %d\n'%storeimgs)
    file.write('storertcs %d\n'%storertcs)
    file.write('dclvl %f\n'%dclvl)
    file.write('rtclvl %f\n'%rtclvl)
    
    file.close()
    
def camconfig(params):
    
    # Locate camera
    evt_context = EVT_Py.EvtContext()
    device_list = evt_context.list_devices(False)
    num_cameras = len(device_list.dev_infos)
    
    print(f"Number of detected cameras: {num_cameras}")
    for dev in device_list.dev_infos:
        print(f"\tCam ID: {dev.camera_id} IP: {dev.ip_address}")
    
    if num_cameras == 0:
        print("ERROR: Unable to find camera")
    else:
        # Initialize camera
        first_dev_info = device_list.dev_infos[0]
        print(f"Setting up cam: {first_dev_info.camera_id}")
        open_camera_params = evt_context.create_open_camera_params()
        cam = evt_context.open_camera(first_dev_info, open_camera_params)
        
        funcs = {
            "UInt32": cam.set_param_uint32,
            "Bool": cam.set_param_bool,
            "Enum": cam.set_enum_str       
            }
        
        # Set params
        print("Setting Parameters:")
        for p in params:
            func = funcs[p[0]]
            name = p[1]
            val = p[2]
            print("{}: {}".format(name, val))
            func(name, val)
            print("...Done!")
            
    print("Closing camera...")
    evt_context.close_camera(cam)
    
    return

# Generate intensity cross corr. For current camera and daq datasets
# assumes camera data are rtc in blposb1 and alzardata files
# for each image and frame in the current folder this generates an
# intensity metric for the daq frame (integration of the waveform) and for
# the image sum of all roi  z values in the rtc.
# if the acquisition system is working properly, these two metrics should be highly correlated
def camdaqhgtxcorrn(npts, nimg, daqdc, blbspath, daqpath):
    
    d = np.hstack(readwfm(daqpath, npts, ch2=True))
    d = -1*(rawdaqtovolts(d)+daqdc) # this is a minor adjustment; not really needed
    
    p = np.fromfile(blbspath, dtype=np.single)
    p = np.reshape(p, (nimg, 60)).T
    
    ps = np.sum(p, 0) # metric for camera roi intensities
    valc = p[0]
    vald = np.sum(d, 1) # metric for daq frame intensities
    
    xcorro1 = np.correlate(valc, vald, "full")
    xcorro2 = np.correlate(ps, vald, "full")
    
    return (xcorro1, xcorro2, valc, vald, p, d)

def fstdiff(y):

    n = len(y)
    yo = np.zeros(n)
    for i in range(n-1):
      yo[i] = y[i+1]-y[i]
    
    yo[n-1] = yo[n-2]
    
    return yo

# Configures a board for acquisition
def ConfigureBoard(board, threshold, twochannel=True):

    global samplesPerSec
    if twochannel:
        samplesPerSec = 2000000000.0
        board.setCaptureClock(ats.INTERNAL_CLOCK,
                              ats.SAMPLE_RATE_2000MSPS,
                              ats.CLOCK_EDGE_RISING,
                              0)
    else:
        samplesPerSec = 4000000000.0
        board.setCaptureClock(ats.INTERNAL_CLOCK,
                              ats.SAMPLE_RATE_4000MSPS,
                              ats.CLOCK_EDGE_RISING,
                              0)
    
    board.inputControlEx(ats.CHANNEL_A,
                         ats.DC_COUPLING,
                         ats.INPUT_RANGE_PM_400_MV,
                         ats.IMPEDANCE_50_OHM)
    
    board.inputControlEx(ats.CHANNEL_B,
                         ats.DC_COUPLING,
                         ats.INPUT_RANGE_PM_400_MV,
                         ats.IMPEDANCE_50_OHM)
    
    board.setTriggerOperation(ats.TRIG_ENGINE_OP_J,
                              ats.TRIG_ENGINE_J,
                              ats.TRIG_EXTERNAL,
                              ats.TRIGGER_SLOPE_POSITIVE,
                              threshold,
                              ats.TRIG_ENGINE_K,
                              ats.TRIG_DISABLE,
                              ats.TRIGGER_SLOPE_POSITIVE,
                              128)

    board.setExternalTrigger(ats.DC_COUPLING,
                             ats.ETR_2V5_50OHM)# ESG: From ats.ETR_TTL)

    triggerDelay_sec = 0
    triggerDelay_samples = int(triggerDelay_sec * samplesPerSec + 0.5)
    board.setTriggerDelay(triggerDelay_samples)

    # NOTE: The board will wait for a for this amount of time for a
    # trigger event.  If a trigger event does not arrive, then the
    # board will automatically trigger. Set the trigger timeout value
    # to 0 to force the board to wait forever for a trigger event.
    #
    # IMPORTANT: The trigger timeout value should be set to zero after
    # appropriate trigger parameters have been determined, otherwise
    # the board may trigger if the timeout interval expires before a
    # hardware trigger event arrives.
    board.setTriggerTimeOut(0)

    board.configureAuxIO(ats.AUX_OUT_TRIGGER,
                         0)

def AcquireData(board, dg, npts, nframes, savepath, acqpath, usecamera=True):
        
    # Shift the data in the frame
    # Maximum pretrigger for Alazar card is 4088
    if npts>2*4088:
        preTriggerSamples = 4088
        postTriggerSamples = npts-4088
    else:
        preTriggerSamples = int(npts/2)
        postTriggerSamples = int(npts/2)

    recordsPerTransfer = 400
    transfersPerAcquisition = int(nframes/400)

    channels = ats.CHANNEL_A | ats.CHANNEL_B # ESG: From ats.CHANNEL_A, needed for two channels        

    # Calculate the number of enabled channels from the channel mask
    channelCount = 0
    for c in ats.channels:
        channelCount += (c & channels == c)
        
    # Create a data file if required
    dataFile = None
    if savepath:
        dataFile = open(savepath, 'wb')

    # Compute the number of bytes per record and per buffer
    memorySize_samples, bitsPerSample = board.getChannelInfo()
    bytesPerSample = (bitsPerSample.value + 7) // 8
    samplesPerRecord = preTriggerSamples + postTriggerSamples
    samplesPerTransfer = samplesPerRecord * recordsPerTransfer * channelCount
    bytesPerTransfer = bytesPerSample * samplesPerTransfer
    bytesPerAcquisition = bytesPerTransfer * transfersPerAcquisition

    postedBufferCount = 4

    # Allocate page aligned memory
    sample_type = ctypes.c_uint8
    if bytesPerSample > 1:
        sample_type = ctypes.c_uint16

    data = ats.DMABuffer(board.handle, sample_type, bytesPerAcquisition)

    # Set the record size
    board.setRecordSize(preTriggerSamples, postTriggerSamples)

    recordsPerAcquisition = recordsPerTransfer * transfersPerAcquisition

    # Configure the board to make an NPT AutoDMA acquisition
    board.beforeAsyncRead(channels,
                          -preTriggerSamples,
                          samplesPerRecord,
                          recordsPerTransfer,
                          recordsPerAcquisition,
                          ats.ADMA_EXTERNAL_STARTCAPTURE | ats.ADMA_NPT | ats.ADMA_FIFO_ONLY_STREAMING)


    # Post DMA buffers to board
    totalBuffersPosted = 0
    while totalBuffersPosted < postedBufferCount and totalBuffersPosted < transfersPerAcquisition:
        board.postAsyncBuffer(data.addr + totalBuffersPosted * bytesPerTransfer, bytesPerTransfer )
        totalBuffersPosted += 1

    start = time.time() # Keep track of when acquisition started
    try:
        board.startCapture() # Start the acquisition
        # Note: this function is nonblocking; returns immediately
        # at this point the daq card is waiting for a trigger pulse stream from the DG535
        
        # Run camera C program
        if usecamera: # usecamera will be true for VMI acquisition; false for DAQ acq only
            # after some initialization, the C program sets DG535
            # to EXT mode to start the pulse stream for triggering image and daq acuisition
            print("Running C++ Script [{}]...".format(acqpath))
            os.system(acqpath)    
        else:
            # start trigger set DG535 to EXT mode to start the pulse stream for triggering
            pass#dg.trigger_on()

        print("Capturing %d transfers. Press <enter> to abort" %
              transfersPerAcquisition)
        transfersCompleted = 0
        bytesTransferred = 0
        while (transfersCompleted < transfersPerAcquisition and not
               ats.enter_pressed()):
            # Wait for the buffer at the head of the list of available
            # buffers to be filled by the board.
            board.waitAsyncBufferComplete(data.addr + transfersCompleted * bytesPerTransfer, timeout_ms=5000)
            transfersCompleted += 1
            bytesTransferred += bytesPerTransfer

            # Add the buffer to the end of the list of available buffers.
            if totalBuffersPosted < transfersPerAcquisition:
                board.postAsyncBuffer(data.addr + bytesPerTransfer * totalBuffersPosted, bytesPerTransfer)
                totalBuffersPosted += 1
    finally:
        board.abortAsyncRead()
    
    # ESG
    if not usecamera:
        pass#dg.trigger_off()
      
    # Compute the total transfer time, and display performance information.
    transferTime_sec = time.time() - start
    print("Capture completed in %f sec" % transferTime_sec)
    bytesPerSec = 0
    recordsPerSec = 0
    if transferTime_sec > 0:
        transfersPerSec = transfersCompleted / transferTime_sec
        bytesPerSec = bytesTransferred / transferTime_sec
        recordsPerSec = recordsPerTransfer * transfersCompleted / transferTime_sec
    print("Captured %d transfers (%.2f transfers per sec)" %
          (transfersCompleted, transfersPerSec))
    print("Captured %d records (%.2f records per sec)" %
          (recordsPerTransfer * transfersCompleted, recordsPerSec))
    print("Transferred %d bytes (%.2E bytes per sec)" %
          (bytesTransferred, bytesPerSec))

    # Optionaly save data to file
    if dataFile:
        data.buffer.tofile(dataFile)

def DAS2C_start(threshold, dg, npts, nframes, savepath, acqpath, twochannel=True, usecamera=True):
    
    #dg.trigger_off()
    board = ats.Board(systemId = 1, boardId = 1)
    ConfigureBoard(board, threshold, twochannel)
    AcquireData(board, dg, npts, nframes, savepath, acqpath, usecamera=usecamera)