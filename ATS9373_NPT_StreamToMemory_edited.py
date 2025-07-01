#-------------------------------------------------------------------------------------------------
#
# Copyright (c) 2008-2023 AlazarTech, Inc.
#
# AlazarTech, Inc. licenses this software under specific terms and conditions. Use of any of the
# software or derivatives thereof in any product without an AlazarTech digitizer board is strictly
# prohibited.
#
# AlazarTech, Inc. provides this software AS IS, WITHOUT ANY WARRANTY, EXPRESS OR IMPLIED,
# INCLUDING, WITHOUT LIMITATION, ANY WARRANTY OF MERCHANTABILITY OR FITNESS FOR A PARTICULAR
# PURPOSE. AlazarTech makes no guarantee or representations regarding the use of, or the results of
# the use of, the software and documentation in terms of correctness, accuracy, reliability,
# currentness, or otherwise; and you rely on the software, documentation and results solely at your
# own risk.
#
# IN NO EVENT SHALL ALAZARTECH BE LIABLE FOR ANY LOSS OF USE, LOSS OF BUSINESS, LOSS OF PROFITS,
# INDIRECT, INCIDENTAL, SPECIAL OR CONSEQUENTIAL DAMAGES OF ANY KIND. IN NO EVENT SHALL
# ALAZARTECH'S TOTAL LIABILITY EXCEED THE SUM PAID TO ALAZARTECH FOR THE PRODUCT LICENSED
# HEREUNDER.
#
#-------------------------------------------------------------------------------------------------'''
#
# This program demonstrates how to configure an ATS9373 to make a stream to memory
# NPT acquisition. In this scheme, a buffer large enough to contain the large
# acquisition is allocated, and segments of this buffer are posted in turn to the board 
# as DMA buffers.
#

from __future__ import division
import ctypes
import numpy as np
import os
import signal
import sys
import time
import subprocess # ESG

sys.path.append(os.path.join(os.path.dirname(__file__), '../..', 'Library'))
import atsapi as ats

samplesPerSec = None

# Configures a board for acquisition
def ConfigureBoard(board):
    # TODO: Select clock parameters as required to generate this
    # sample rate
    #
    # For example: if samplesPerSec is 100e6 (100 MS/s), then you can
    # either:
    #  - select clock source INTERNAL_CLOCK and sample rate
    #    SAMPLE_RATE_100MSPS
    #  - or select clock source FAST_EXTERNAL_CLOCK, sample rate
    #    SAMPLE_RATE_USER_DEF, and connect a 100MHz signal to the
    #    EXT CLK BNC connector
    
    # ESG
    global usecamera
    usecamera = True
    
    global samplesPerSec    
    # ESG: edit next line depending on number of channels
    #samplesPerSec = 4000000000.0
    samplesPerSec = 2000000000.0 # 2 channels
    
    # ESG
    global nframes
    (datadir, nstacks, nframes, iexposure, imgsize,
    roix0, roiy0, storeimgs, storertcs, dclvl, rtclvl) = readvmi3dparams()
    
    board.setCaptureClock(ats.INTERNAL_CLOCK,
                          ats.SAMPLE_RATE_2000MSPS, # ESG: 4000MSPS for 4GHz operation
                          ats.CLOCK_EDGE_RISING,
                          0)
    
    # TODO: Select channel A input parameters as required.
    board.inputControlEx(ats.CHANNEL_A,
                         ats.DC_COUPLING,
                         ats.INPUT_RANGE_PM_400_MV,
                         ats.IMPEDANCE_50_OHM)
    
    
    # TODO: Select channel B input parameters as required.
    board.inputControlEx(ats.CHANNEL_B,
                         ats.DC_COUPLING,
                         ats.INPUT_RANGE_PM_400_MV,
                         ats.IMPEDANCE_50_OHM)
    
    
    threshold = 2000 # ESG
    
    # TODO: Select trigger inputs and levels as required.
    board.setTriggerOperation(ats.TRIG_ENGINE_OP_J,
                              ats.TRIG_ENGINE_J,
                              ats.TRIG_EXTERNAL, # EST: From TRIG_CHAN_A
                              ats.TRIGGER_SLOPE_POSITIVE,
                              threshold, # ESG: From 150
                              ats.TRIG_ENGINE_K,
                              ats.TRIG_DISABLE,
                              ats.TRIGGER_SLOPE_POSITIVE,
                              128)

    # TODO: Select external trigger parameters as required.
    board.setExternalTrigger(ats.DC_COUPLING,
                             ats.ETR_TTL)

    # TODO: Set trigger delay as required.
    triggerDelay_sec = 0
    triggerDelay_samples = int(triggerDelay_sec * samplesPerSec + 0.5)
    board.setTriggerDelay(triggerDelay_samples)

    # TODO: Set trigger timeout as required.
    #
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

    # Configure AUX I/O connector as required
    board.configureAuxIO(ats.AUX_OUT_TRIGGER,
                         0)
    

def AcquireData(board):
    # Make an AutoDMA acquisition from dual-ported memory.

    # No pre-trigger samples in NPT mode
    preTriggerSamples = 2048 # ESG: From 0, to shift the data in the frame

    # TODO: Select the number of samples per record.
    # NOTE: The number of bytes per DMA buffer *must* be a multiple 
    # of 4096, this is because DMA buffers must be page-aligned.
    postTriggerSamples = 2048 # ESG: From 4096, to shift the data in the frame

    # TODO: Select the number of records per channel per DMA buffer.
    recordsPerTransfer = 400

    # TODO: Select the number of DMA buffers (i.e. transfers) per acquisition.
    transfersPerAcquisition = nframes/400 # ESG: From 50

    # TODO: Select which channels to capture (A, B, or both).
    channels = [ats.CHANNEL_A, ats.CHANNEL_B] # ESG: From ats.CHANNEL_A, needed for two channels        

    # TODO: Should data be saved to file?
    saveData = True # ESG: From False


    # Calculate the number of enabled channels from the channel mask
    channelCount = 0
    for c in ats.channels:
        channelCount += (c & channels == c)
        
    # Create a data file if required
    dataFile = None
    if saveData:
        dataFile = open(os.path.join(os.path.dirname(__file__),
                                     "c:/data/daqdata.uint16"), 'wb') # ESG: Changed file name from data.bin

    # Compute the number of bytes per record and per buffer
    memorySize_samples, bitsPerSample = board.getChannelInfo()
    bytesPerSample = (bitsPerSample.value + 7) // 8
    samplesPerRecord = preTriggerSamples + postTriggerSamples
    samplesPerTransfer = samplesPerRecord * recordsPerTransfer * channelCount
    bytesPerTransfer = bytesPerSample * samplesPerTransfer
    bytesPerAcquisition = bytesPerTransfer * transfersPerAcquisition

    # TODO: Select number of DMA buffers to allocate
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
        # ESG: Note: this function is nonblocking; returns immediately
        # at this point the daq card is waiting for a trigger pulse stream from the DG535
        
        # ESG: run camera C program
        if usecamera: # usecamera will be true for VMI acquisition; false for DAQ acq only
            # after some initialization, the C program sets DG535
            # to EXT mode to start the pulse stream for triggering image and daq acuisition
            subprocess.run('C:\data\currentevtacq.exe')    
        else:
            # start trigger set DG535 to EXT mode to start the pulse stream for triggering
            subprocess.run('C:\data\setdgtrig.exe 1')

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

            # TODO: Process sample data in this buffer. Data is available
            # as a NumPy array at buffer.buffer

            # NOTE:
            #
            # While you are processing this buffer, the board is already
            # filling the next available buffer(s).
            #
            # You MUST finish processing this buffer and post it back to the
            # board before the board fills all of its available DMA buffers
            # and on-board memory.
            #
            # Samples are arranged in the buffer as follows:
            # S0A, S0B, ..., S1A, S1B, ...
            # with SXY the sample number X of channel Y.
            #
            # A 12-bit sample code is stored in the most significant bits of
            # each 16-bit sample value.
            #
            # Sample codes are unsigned by default. As a result:
            # - 0x0000 represents a negative full scale input signal.
            # - 0x8000 represents a ~0V signal.
            # - 0xFFFF represents a positive full scale input signal.

            # Add the buffer to the end of the list of available buffers.
            if totalBuffersPosted < transfersPerAcquisition:
                board.postAsyncBuffer(data.addr + bytesPerTransfer * totalBuffersPosted, bytesPerTransfer)
                totalBuffersPosted += 1

    finally:
        board.abortAsyncRead()
    
    # ESG
    if not usecamera:
        subprocess.run('C:\data\setdgtrig.exe 2') # stop trigger
      
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
  
# ESG
def readvmi3dparams():
    params = []
    
    def param(line, dtype):
        space = line.find(' ')
        return dtype(line[space+1:])
    
    with open("c:/data/vmi3drunparamsrtc.txt") as f:
        for i in range(11):
            
            if i==0: dtype=str
            elif i<9: dtype=int
            else: dtype=float
            
            line = f.readline()
            params.append(param(line, dtype))
    
    return params

if __name__ == "__main__":
    
    # ESG: stop trigger
    subprocess.run('c:\data\setdgtrig.exe 2')
    
    board = ats.Board(systemId = 1, boardId = 1)
    ConfigureBoard(board)
    AcquireData(board)