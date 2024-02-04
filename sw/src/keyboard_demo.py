import keyboard

keyboard.remap_hotkey("0", "left")
keyboard.remap_hotkey("1", "right")

while True:
    keyboard.wait()