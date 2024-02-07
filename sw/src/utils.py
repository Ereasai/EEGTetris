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
    LEFT = 0
    LEFT_END = 1
    RIGHT = 2
    RIGHT_END = 3
    
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