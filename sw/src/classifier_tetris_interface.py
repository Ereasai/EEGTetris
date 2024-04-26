"""
This program is intended to be used as an interface between the classifier and the game.
The classifier will have arbitrary outputs, i.e. 0, 1. We now map these keystrokes into
correct keystrokes for the game Tetris. 
"""

import keyboard
import pylsl
import pygetwindow as gw
import time

GAME_WINDOW_NAME = 'Tetris'

result_inlet = None
info = pylsl.resolve_stream('name', "Classifier_Result_Out")
result_inlet = pylsl.stream_inlet(info[0], recover=False)
print(f'mapper has received the {info[0].type()} inlet.')

while True:
    res, time = result_inlet.pull_sample(timeout=0)

    found = False
    wins = gw.getWindowsWithTitle(GAME_WINDOW_NAME)
    for win in wins:
        if win.title == GAME_WINDOW_NAME and win.isActive:
            found = True
            break

    if not found:
        print("Tetris window not open. Waiting for it to be open again.")
        continue

    if res is not None:
        mvt = res[0]
        print(f'i got {mvt}')
        if (mvt == 'left'):
            keyboard.press_and_release('a') # rotate CCW
        elif (mvt == 'right'):
            keyboard.press_and_release('right') # move right
        
        # otherwise do nothing.