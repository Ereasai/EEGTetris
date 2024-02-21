import numpy as np
from scipy import signal

def vis_preprocess(epoch, Wn=[0.1, 30], Fs=300, n=4):
    """ Band-passes EEG epoch with Butterworth filter and returns time and 
    filtered signal for online plotting

    Parameters
    ----------
    epoch: [(t, [ch1, ch2, ..., ch8]), ...]
        list of tuples with elements time and the eight EEG channels from the DSI-7
        length of list is eq. to no. of timepoints sampled
    
    Wn: list of len 2
        list containing band-pass frequencies, default 0.1 to 30 Hz
    
    Fs: int
        sampling rate, default 300 (ie. Fs of DSI-7 headset)
    
    n: int 
        order of filter

    Returns
    -------
    t: np.arr (t, 1)
        2D arr with t as no. of timepoints
        preserved from input epoch

    fsg: np.arr (t, 8)  # read as 'filtered_signal'
        2D arr w/ t as no. of timepoints, 8 channels of DSI-7 headset across columns
    """
    t  = np.array([tup[0] for tup in epoch]).reshape((-1, 1))
    sg = np.array([tup[1] for tup in epoch])

    sos = signal.butter(n, Wn, btype='band', fs=Fs, output='sos')
    fsg = signal.sosfilt(sos, sg, axis=0)

    return t, fsg