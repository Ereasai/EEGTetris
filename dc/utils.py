"""
Utilities file.
Contains useful functions to be shared by other files.
"""

from enum import Enum

"""
This defines how we encode each markers with numbers. 
This is needed because LSL marker inlet requires us to send in a number, not a
string. 
"""
class Marker_Enum(Enum):
    LEFT_ARM = 0
    LEFT_ARM_END = 1
    RIGHT_ARM = 2
    RIGHT_ARM_END = 3
    LEFT_LEG = 4
    LEFT_LEG_END = 5
    RIGHT_LEG = 6
    RIGHT_LEG_END = 7
    CUE = 8
    MI_START = 9
    
"""
Given name of the marker, it returns the corresponding code.
"""
def get_marker_number(task_name):
    try:
        return Marker_Enum[task_name.upper()].value
    except KeyError:
        raise ValueError(f"{task_name} is not a valid marker name")

"""
Given the number of the marker, it returns the corresponding name.
"""
def get_marker_name(task_code):
    try:
        return Marker_Enum(task_code).name
    except ValueError:
        raise ValueError(f"{task_code} is not a valid marker value")