from PyQt5.QtWidgets import QApplication, QVBoxLayout, QWidget, QLabel, QProgressBar
from PyQt5.QtCore import QThread, pyqtSignal
import pyqtgraph as pg
import numpy as np
import pylsl
import time
from scipy import signal
import sys
import random
import mne

import joblib 
from custom_transformers import FilterBank


def vis_mne(epoch, Fs=300, l_freq=0.1, h_freq=30):

    fsg = mne.filter.filter_data(epoch, sfreq=Fs, l_freq=l_freq, h_freq=h_freq)

    return fsg

def vis_preprocess(epoch, Wn=[7, 30], Fs=300, n=4):
    sos = signal.butter(n, Wn, btype='band', fs=Fs, output='sos')
    fsg = signal.sosfilt(sos, epoch, axis=0)

    return fsg

def init_lsl():
    """
    Sets up the LSL inlets and outlets.
    * `eeg_inlet` is the incoming EEG stream.
    * `result_outlet` is the outgoing result stream.
    
    Returns (`eeg_inlet`, `result_outlet`)
    """
    print("Looking for an EEG stream...")
    streams = pylsl.resolve_stream('type', 'EEG')
    inlet = pylsl.StreamInlet(streams[0])
    print("Stream found and connected.")

    info = pylsl.stream_info('Classifier_Result_Out', 'Markers', 1, 0, pylsl.cf_string, 'unsampledStream')
    result_outlet = pylsl.stream_outlet(info, 1, 1)
    print("Created result outlet.")

    return inlet, result_outlet

def classify(epoch_data, pipeline):
    """
    Input:
    * `epoch_data` is a numpy array of shape (n_channels, n_samples).
    
    Output:
    * classification result of the epoch.
    """
    
    result = pipeline.predict(epoch_data)
    # we expect the result to be either 0 or 1 or 2.

    return ['left', 'right', 'baseline'][result]

def update_data_and_plot(samples):
    """
    Input
    * `samples` is an numpy array of shape (n_channels, n_samples)

    It will update the `data_buffer` by shifting it (discarding n_samples of oldest samples)
    and pasting in `samples` in the front.

    """
    global data_buffer, curves, window_size, n_channels
    offset = 500 # graphing: offset between each channel

    # samples has shape (n_channels, n_samples).
    n_samples = samples.shape[1]
    data_buffer = np.roll(data_buffer, n_samples, axis=1) # shift --> by n_samples.
    data_buffer[:,0:n_samples] = samples

    # draw each channel on the plot.
    for i in range(n_channels):
        # plot backwards to make it seem like new data comes in from the right.
        curves[i].setData(np.arange(window_size),           # 0 1 2 ... window_size-1
                          data_buffer[i,::-1] + i * offset) # x[N-1] ... x[0]

def plot_epoch (epoch):
    global curves_epoch, n_channels
    offset = 500
    n_samples = epoch.shape[1]

    epoch = vis_mne(epoch) # preprocessor
    
    for i in range(n_channels):
        curves_epoch[i].setData(np.arange(n_samples), 
                                epoch[i, ::-1] + i * offset)

class DataThread(QThread):
    data_signal = pyqtSignal(np.ndarray)
    epoch_signal = pyqtSignal(np.ndarray)
    classification_signal = pyqtSignal(str)
    timer_signal = pyqtSignal(int)

    def __init__(self, epoch_size=900, classify_interval=100, pipeline=None):
        super(DataThread, self).__init__()
        self.classify_interval = classify_interval
        self.last_classify_time = 0
        self.epoch_size = epoch_size
        self.pipeline = pipeline

    def run(self):
        while True:
            samples, timestamps = eeg_inlet.pull_chunk(timeout=1.0, max_samples=100)
            
            # do not proceed if we are getting no data.
            if (len(timestamps)==0):
                continue

            samples = np.array(samples).T # (n_samples, n_channels)
            samples = samples[:,::-1] # it comes in as oldest to youngest.
            # but we want youngest to be in the front, and oldest in the back.
            self.data_signal.emit(samples) 

            current_time = int(time.time() * 1000)  # current time in ms

            # update GUI timer, it wants "remaining time"
            self.timer_signal.emit(self.classify_interval + self.last_classify_time - current_time)

            time_elapsed = current_time - self.last_classify_time

            # only classify if enough time has passed.
            if time_elapsed < self.classify_interval:
                continue
            
            self.last_classify_time = current_time;

            # the front entries are the recent
            epoch_data = data_buffer[:, :self.epoch_size]
            self.epoch_signal.emit(epoch_data) # for plotting epoch.

            result = classify(epoch_data, self.pipeline) # run pipeline!

            result_outlet.push_sample(pylsl.vectorstr([result])) # broadcast the result to LSL.
            
            self.classification_signal.emit(result) # update GUI for result.


if __name__ == '__main__':
    # GUI setup
    app = QApplication([])
    mainWidget = QWidget()  # Main widget that holds the layout
    layout = QVBoxLayout()  # Vertical layout
    
    win = pg.GraphicsLayoutWidget()
    pg.setConfigOptions(antialias=True)

    plot = win.addPlot(title="Live EEG Stream")
    plot2 = win.addPlot(title="epoch (preprocessed)")
    timer_progbar = QProgressBar(maximum=500)
    classification_label = QLabel("Classification: Not yet classified")
    classification_label.setStyleSheet("font: 30px")

    layout.addWidget(win)  # Add window containing plots to the layout
    layout.addWidget(classification_label)  # Add the label to the layout
    layout.addWidget(timer_progbar)

    mainWidget.setLayout(layout)  # Set the layout on the main widget
    mainWidget.show()  # Show the main widget

    # --------------------------------------------

    # set globals
    n_channels = 8
    window_size = 1000 # buffer contains 1000 samples
    interval = 500 # ms
    epoch_size = 900

    data_buffer = np.zeros((8, 1000)) # buffer

    ### LOAD PIPELINE ###
    # pipeline = joblib.load('./csp_lda_pipeline.joblib')
    class DumbClassifier():
        """
        Placeholder classifier that guesses between three options.
        """
        def __init__(self):
            pass
        def predict(self, X):
            # ignore X, I'm guessing. 
            assert (X.shape == (8,900))
            return random.randint(0,2)
    
    pipeline = DumbClassifier()

    #####################

    # setup intlets, and outlets
    # the inlet is incoming EEG data
    # the outlet is the classification results
    eeg_inlet, result_outlet = init_lsl()



    # this sets up each 'line' of the channels
    curves = [plot.plot(pen=pg.intColor(i)) for i in range(n_channels)]
    curves_epoch = [plot2.plot(pen=pg.intColor(i)) for i in range(n_channels)]
    plot.setXRange(100, epoch_size)
    plot.setMouseEnabled(x=False, y=False)

    # set data thread and signal connections
    # signals are callbacks
    data_thread = DataThread(epoch_size=epoch_size, classify_interval=interval, pipeline=pipeline)
    data_thread.data_signal.connect(update_data_and_plot)
    data_thread.classification_signal.connect(
        lambda result : classification_label.setText(f"Classification: {result}"))
    data_thread.timer_signal.connect(
        lambda time : timer_progbar.setValue(interval - time))
    data_thread.epoch_signal.connect(plot_epoch)
    data_thread.start()

    sys.exit(app.exec_())