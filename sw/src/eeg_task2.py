"""
the following code was adapted from eeg_task.py by
    EMMA CHEN
"""
import pylsl
import random
import numpy as np
from psychopy import visual, core, constants, event, prefs, sound

WINDOW = None
CLOCK = core.Clock()
MARKER_OUTLET = None

"""
Pushes a given event onto the marker stream. MARKER_OUTLET must be set up.
"""
def CreateMarker(event):
        # marker_id = utils.get_marker_number(event)
        MARKER_OUTLET.push_sample(pylsl.vectorstr([event]));

"""
Create a randomized sequence of tasks. There will be `n` of each task.
"""
def CreateSequence(n):
    
    movements = ['LEFT_ARM','RIGHT_ARM'] 
    seq = movements*n
    
    # randomize order of sequence
    random.seed()
    random.shuffle(seq)

    return seq

def RunParadigm():

    # SET UP STIMULI
    taskStim = visual.TextStim(WINDOW, text='', 
                                       units='norm', 
                                       alignText='center', 
                                       color="black")
    messageStim = visual.TextStim(WINDOW, text='', 
                                       units='norm', 
                                       alignText='center', 
                                       color="black")
    fixation = visual.ShapeStim(WINDOW, vertices=((0, -10), (0, 10), (0,0), (-10, 0), (10, 0)), 
                                        lineWidth=1,  
                                        closeShape=False,  
                                        lineColor="black",
                                        fillColor="black")
    beep = sound.Sound(r'sw\src\resources\beep.wav')
    
    # PARADIGM
    totalChunk = 6
    taskPerChunk = 5
    for chunk in range(totalChunk):

        sequence = CreateSequence(taskPerChunk)

        for task in sequence:
            # task = LEFT_ARM, RIGHT_ARM

            # FIXATION
            CreateMarker(task)
            fixation.draw()
            WINDOW.flip()
            core.wait(2)

            # CUE
            beep.play()
            CreateMarker('CUE')
            core.wait(random.uniform(0.5, 0.8))

            # TASK
            CreateMarker('MI_START')

            if task == 'LEFT_ARM':
                taskStim.text = f'< < imagine {task} < <'
                taskStim.draw()
                WINDOW.flip()
                core.wait(4)

            if task == 'RIGHT_ARM':
                taskStim.text = f'> > imagine {task} > >'
                taskStim.draw()
                WINDOW.flip()
                core.wait(4)

            WINDOW.flip()
            CreateMarker('MI_END')

            core.wait(random.uniform(0.5, 0.85))

        if chunk == totalChunk-1: # on last iteration, exit.
            break

        # LONG BREAK
        messageStim.text = f'You have reached a break! Press SPACE to start next chunk. (progress: {chunk+1}/{totalChunk})'
        messageStim.draw()
        WINDOW.flip()
        event.waitKeys(keyList=['space']) 
    
    # FINISHED
    messageStim.text = "Task completed. Press ANY KEY to exit."
    messageStim.draw()
    WINDOW.flip()
    event.waitKeys(keyList=None) 

########################################################### MAIN
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
    info = pylsl.stream_info('EEG_Markers', 'Markers', 1, 0, pylsl.cf_string, 'unsampledStream');
    MARKER_OUTLET = pylsl.stream_outlet(info, 1, 1)
    
    # warning message for LabRecorder
    warning_msg = visual.TextStim(WINDOW, text='Make sure that LabRecorder is connected with EEG_Markers. \n \
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

########################################################### END OF MAIN