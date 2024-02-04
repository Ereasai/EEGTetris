# import time       # DO NOT USE time.sleep() for timing, use core.wait() 
import pylsl
import random
import numpy as np
from psychopy import visual, core, constants, event, sound

WINDOW = None
CLOCK = core.Clock()
MARKER_OUTLET = None

"""
Create a randomized sequence of tasks. There will be `n` of each task.
"""
def CreateSequence(n):
    
    movements = ['LEFT','RIGHT'] 
    seq = movements*n
    
    # randomize order of sequence
    random.seed()
    random.shuffle(seq)

    return seq

def RunParadigm():

    def CreateMarker(task):
        myMap = {"LEFT" : 0, "LEFT_END" : 1, "RIGHT" : 2, "RIGHT_END" : 3}
        MARKER_OUTLET.push_sample(str(myMap[task]))

    # SET UP STIMULI
    # TODO: there will be gifs for each task. 
    vidStim = visual.MovieStim(WINDOW, filename='./resources/death-corridor-death.gif', 
                                       size=[100,100], 
                                       pos=(100, 0), 
                                       autoStart=False)
    vidStim.play() # file is not loaded in until it is played. 
                   # removes delay when being displayed for the first time.
    taskStim = visual.TextStim(WINDOW, text='', 
                                       units='norm', 
                                       alignText='center', 
                                       color="black");
    fixation = visual.ShapeStim(WINDOW, vertices=((0, -10), (0, 10), (0,0), (-10, 0), (10, 0)), 
                                        lineWidth=1,  
                                        closeShape=False,  
                                        lineColor="black",
                                        fillColor="black")
    beep = sound.Sound('./resources/beep.wav')
    
    # PARADIGM
    totalChunk = 2
    taskPerChunk = 1
    for chunk in range(totalChunk):

        sequence = CreateSequence(taskPerChunk)

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
    
    WINDOW = visual.Window(
        screen=0,
        size=[600, 400],
        units="pix",
        fullscr=False,
        color="white",
        gammaErrorPolicy="ignore"
    )

    # create marker stream
    info = pylsl.stream_info('EMG_Markers', 'Markers', 1, 0, pylsl.cf_string, 'unsampledStream');
    MARKER_OUTLET = pylsl.stream_outlet(info, 1, 1)
    
    # warning message for LabRecorder
    warning_msg = visual.TextStim(WINDOW, text='Make sure that LabRecorder is connected with EMG_Markers. \n \
                                                Trial will begin automatically when you start recording in LabRecorder.',
                                          units='norm', alignText='center', color=[1,0,0], bold=True);
    warning_msg.draw()
    WINDOW.flip()

    # wait for markerstream to be used by LabRecorder
    while not MARKER_OUTLET.have_consumers():
        core.wait(0.2)

    # RUN SEQEUENCE OF TRIALS
    RunParadigm()

    # cleanup
    WINDOW.close()
    core.quit()

    ####################################################### END OF MAIN