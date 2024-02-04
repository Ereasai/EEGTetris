import time
import psychopy
import pylsl
import random
import numpy as np
from psychopy import event
from psychopy import visual
import threading
import sys
import os

from itertools import chain
from math import atan2, degrees

from datetime import datetime

WINDOW = None
EEG_INLET = None
MARKER_OUTLET = None

BG_COLOR = [-1,-1,-1]

PROMPT_FRAMERATE = 10
PROMPT_DURATION = 2

# creates a psychopy ShapeStim object, for creating a fixation cross
# in the window

def InitFixation(size=50):
    return psychopy.visual.ShapeStim(
            win=WINDOW,
            units='pix',
            size=size,
            fillColor=[1, 1, 1],
            lineColor=[1, 1, 1],
            lineWidth=0.01,
            vertices='cross',
            name='off', # Used to determine state
            pos=[0, 0]
        )

def CreateSequence(n):
    # List of movement prompts
    movements = ['CLENCH FIST','SNAP'] # TODO: add correct names for stimulus
    # Duplicate movements by a factor of n(the argument) to create a longer sequence
    seq = movements*n
    
    # Randomize order of sequence
    random.seed()
    random.shuffle(seq)

    # Add rests between each movement prompt in seq to restSeq
    restSeq = []
    for s in seq:
        restSeq.extend(['REST', s])

    return restSeq
    
# def RunParadigm():
#     met = psychopy.visual.TextStim(WINDOW, text = 'X', units = 'norm', alignText = 'center');
#     met.setHeight(0.1);
#     met.pos = (-0.4, 0)
#     met.draw()

#     vidSize = [100, 100]
#     vidStim = psychopy.visual.MovieStim(WINDOW, filename = './resources/death-corridor-death.gif', size = vidSize, pos=(100, 0), autoStart = True)

#     sequence = CreateSequence(2)

#     fix = InitFixation();
#     fix.draw()
    
#     metadataFile = open(metadata, "w")

#     for item in sequence:
#         met.setText(item)
#         met.draw()
#         WINDOW.flip()

#         times = EEG_INLET.pull_sample()


#         metadataFile.write(str(times[1]) + ", " + item + "\n")

#         for x in range(PROMPT_FRAMERATE):
#             vidStim.draw()
#             WINDOW.flip()
#             time.sleep(PROMPT_DURATION / PROMPT_FRAMERATE)

def RunParadigm():
    for i in range(0,10):
        MARKER_OUTLET.push_sample(str(i))
        time.sleep(1)

        

#     WINDOW.flip() # swap buffer

"""
LSL Thread

Open `fpath` and save data points from EEG inlet. Saves the file in CVS file
format. (time, channel_1, ..., channel_8)

TODO: consider buffer implementaton 
"""
def lsl_thread(fpath):
    print("LSL Thread started.")
    while True:
        sample, times = EEG_INLET.pull_sample()
        with open(fpath, "a") as fo:
            fo.write(f"{str(times)}, {str(sample)[1:-1]}\n")

    
if __name__ == "__main__":
    
    WINDOW = psychopy.visual.Window(
        screen=0,
        size=[600, 400], # add
        units="pix",
        fullscr=False,
        color=BG_COLOR,
        gammaErrorPolicy="ignore"
    )

    # create eeg stream & inlet
    # eeg_streams = pylsl.resolve_stream('type', 'EEG')
    # EEG_INLET = pylsl.stream_inlet(eeg_streams[0], recover = False)
    # print("Inlet created.")
    # time_str = datetime.now().strftime("%Y_%m_%d_%H%M%S")
    # lsl thread
    # lsl = threading.Thread(target=lsl_thread, args=(f"./results/{time_str}_data.csv",))
    # lsl.daemon = True   # this thread does not affect the termination of the process.
                        # if main finishes, the whole process will end, regardless of whether lsl is done.
    # lsl.start()

    info = pylsl.stream_info('EMG_Markers', 'Markers', 1, 0, pylsl.cf_string, 'unsampledStream');
    MARKER_OUTLET = pylsl.stream_outlet(info, 1, 1)
    
    input("waiting")
    # RUN SEQEUENCE OF TRIALS
    # metadata = f"./results/{time_str}_metadata.csv"
    print("starting paradigm")
    RunParadigm()

    time.sleep(2)

    # END OF MAIN




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