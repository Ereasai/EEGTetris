"""
This program is intended to be used as an interface between the classifier and the game.
The classifier will have arbitrary outputs, i.e. 0, 1. We now map these keystrokes into
correct keystrokes for the game Tetris. 
"""

import keyboard

keyboard.remap_hotkey("0", "left")
keyboard.remap_hotkey("1", "right")

while True:
    keyboard.wait()