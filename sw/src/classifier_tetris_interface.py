"""
This program is intended to be used as an interface between the classifier and the game.
The classifier will have arbitrary outputs, i.e. 0, 1. We now map these keystrokes into
correct keystrokes for the game Tetris. 
"""

import keyboard
import pylsl

result_inlet = None
info = pylsl.resolve_stream('name', "Classifier_Result_Out")
result_inlet = pylsl.stream_inlet(info[0], recover=False)
print(f'mapper has received the {info[0].type()} inlet.')

keyboard.remap_hotkey("0", "left")
keyboard.remap_hotkey("1", "right")

while True:
    res, time = result_inlet.pull_sample(timeout=0)

    if res is not None:
        mvt = res[0]
        print(f'i got {mvt}')
        # keyboard.press_and_release(mvt)