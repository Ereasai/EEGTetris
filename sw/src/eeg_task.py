import time
import psychopy
import pylsl
import random
import numpy as np
from psychopy import visual, core, constants, event, sound
import threading
import sys
import os

from itertools import chain
from math import atan2, degrees

from datetime import datetime

WINDOW = None
DRAW_OBJECTS = []
DRAW_THR = None

CLOCK = core.Clock()

MARKER_OUTLET = None

PROMPT_FRAMERATE = 10
PROMPT_DURATION = 2

# creates a psychopy ShapeStim object, for creating a fixation cross
# in the window

def InitFixation(size=50):

    return visual.ShapeStim(
        win=WINDOW,
        vertices=((0, -10), (0, 10), (0,0), (-10, 0), (10, 0)), 
        lineWidth=1,
        closeShape=False,
        lineColor="black",
        fillColor="black"
    )

def CreateSequence(n):
    # List of movement prompts
    movements = ['LEFT','RIGHT'] # TODO: add correct names for stimulus
    # Duplicate movements by a factor of n(the argument) to create a longer sequence
    seq = movements*n
    
    # Randomize order of sequence
    random.seed()
    random.shuffle(seq)

    return seq

def CreateMarker(task):
    myMap = {"LEFT" : 0, "LEFT_END" : 1, "RIGHT" : 2, "RIGHT_END" : 3}
    MARKER_OUTLET.push_sample(str(myMap[task]))

def RunParadigm():    
    vidStim = psychopy.visual.MovieStim(WINDOW, 
                                        filename='./resources/death-corridor-death.gif', 
                                        size=[100,100], pos=(100, 0), autoStart=False)
    taskStim = psychopy.visual.TextStim(WINDOW, text='',
                                     units='norm', alignText='center', color="black");
    fixation = InitFixation(10)
    beep = sound.Sound('./resources/beep.wav')
    
    totalChunk = 2
    for chunk in range(totalChunk):

        sequence = CreateSequence(1)

        for task in sequence:
            # FIXATION
            fixation.draw()
            WINDOW.flip()
            core.wait(2)

            # CUE
            beep.play()
            CreateMarker(task)
            core.wait(random.uniform(0.5, 0.8))

            # TASK WITH VIDEO
            taskStim.text = task
            vidStim.seek(0)
            vidStim.play()
            CLOCK.reset()
            while CLOCK.getTime() < 2: # stimulus length
                vidStim.draw()
                taskStim.draw()
                WINDOW.flip()

            # REST
            CreateMarker(task + "_END")
            WINDOW.flip()
            core.wait(random.uniform(4.5, 6.5))

        if chunk == totalChunk-1: # on last iteration, exit.
            break

        # LONG BREAK
        taskStim.text = f'You can rest, press SPACE to do next chunk. (progress: {chunk+1}/{totalChunk})'
        taskStim.draw()
        WINDOW.flip()
        event.waitKeys(keyList=['space']) 
    
    # FINISHED
    taskStim.text = "Task completed. Press ANY KEY to exit."
    taskStim.draw()
    WINDOW.flip()
    event.waitKeys(keyList=None) 

if __name__ == "__main__":
    
    WINDOW = psychopy.visual.Window(
        screen=0,
        size=[600, 400], # add
        units="pix",
        fullscr=False,
        color="white",
        gammaErrorPolicy="ignore"
    )

    # create marker stream
    info = pylsl.stream_info('EMG_Markers', 'Markers', 1, 0, pylsl.cf_string, 'unsampledStream');
    MARKER_OUTLET = pylsl.stream_outlet(info, 1, 1)
    
    while not MARKER_OUTLET.have_consumers():
        # warning message for LabRecorder
        warning_msg = psychopy.visual.TextStim(WINDOW, 
                                            text='Make sure that LabRecorder is connected with EMG_Markers. Trial will begin autoamtically when you start recording in LabRecorder.', 
                                            units='norm', alignText='center', color=[1,0,0]);
        warning_msg.draw()
        WINDOW.flip()
        core.wait(0.2)



    # event.waitKeys(keyList=["space"]) # wait for user

    # RUN SEQEUENCE OF TRIALS
    RunParadigm()

    # cleanup
    WINDOW.close()
    core.quit()

    ####################################################### END OF MAIN




# Calculate number of frames in ms milliseconds, given ms and fs (frame rate)
def MsToFrames(ms, fs):
    dt = 1000 / fs
    return np.round(ms / dt).astype(int)



"""
Unused Methods from aeroMus

    DegToPix
    listFlatten
    InitPhotosensor

    TODO
    left right cross
"""