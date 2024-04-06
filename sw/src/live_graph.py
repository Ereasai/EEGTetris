from PyQt5.QtWidgets import QApplication, QVBoxLayout, QWidget, QLabel, QProgressBar, QLineEdit
from PyQt5.QtGui import QIntValidator
from PyQt5.QtCore import QThread, pyqtSignal
import pyqtgraph as pg
import numpy as np
import pylsl
import time
from scipy import signal
import sys
import random
import mne

from joblib import load
from custom_transformers import FilterBank


def vis_mne(epoch, Fs=300, l_freq=0.1, h_freq=30):
    t  = np.array([tup[0] for tup in epoch]).reshape((-1, 1))
    sg = np.array([tup[1] for tup in epoch])

    fsg = mne.filter.filter_data(sg.T, sfreq=Fs, l_freq=l_freq, h_freq=h_freq)

    return t, fsg.T

def vis_preprocess(epoch, Wn=[7, 30], Fs=300, n=4):
    t  = np.array([tup[0] for tup in epoch]).reshape((-1, 1))
    sg = np.array([tup[1] for tup in epoch])

    sos = signal.butter(n, Wn, btype='band', fs=Fs, output='sos')
    fsg = signal.sosfilt(sos, sg, axis=0)

    return t, fsg


pipeline = None
eeg_inlet = None
result_outlet = None

# Assuming you have a classifier function defined somewhere in your code
def classify(epoch_data):
    channel_data = [channels for _, channels in epoch_data]
    channel_data_transposed = np.array(channel_data).T
    channel_data_transposed = channel_data_transposed[0:4]
    
    # add an extra dimension to represent a single epoch to classify
    single_trial_formatted = np.expand_dims(channel_data_transposed, axis=0)

    result = pipeline.predict(single_trial_formatted)
    # result looks like [trial1_result, trial2_result, ...]
    # since we only pass in single trial, [trial_result]

    return ['left', 'right', 'up'][random.randint(0,2)]
    # return ['left','right'][int(result[0].item())]

def init_lsl():
    print("Looking for an EEG stream...")
    streams = pylsl.resolve_stream('type', 'EEG')
    inlet = pylsl.StreamInlet(streams[0])
    print("Stream found and connected.")

    info = pylsl.stream_info('Classifier_Result_Out', 'Markers', 1, 0, pylsl.cf_string, 'unsampledStream')
    result_outlet = pylsl.stream_outlet(info, 1, 1)
    print("Created result outlet.")

    return inlet, result_outlet

class DataThread(QThread):
    data_signal = pyqtSignal(np.ndarray)
    classification_signal = pyqtSignal(str)
    timer_signal = pyqtSignal(int)
    epoch_signal = pyqtSignal(list)

    def __init__(self, epoch_length=1000, classify_interval=100):
        super(DataThread, self).__init__()
        self.epoch_length = epoch_length  # Number of samples in an epoch
        self.classify_interval = classify_interval  # Interval to classify and reset in ms
        self.last_classify_time = 0
        self.epoch_data = []

    def run(self):
        while True:
            samples, timestamps = eeg_inlet.pull_chunk(timeout=1.0, max_samples=10)
            if timestamps:
                current_time = int(time.time() * 1000)  # Current time in ms

                self.data_signal.emit(np.array(samples))
                
                self.epoch_data.extend([(t, s) for t, s in zip(timestamps, samples)])

                # update timer
                self.timer_signal.emit(self.classify_interval + self.last_classify_time - current_time)

                if current_time - self.last_classify_time >= self.classify_interval:

                    self.last_classify_time = current_time;
                    
                    result = classify(self.epoch_data)
                    result_outlet.push_sample(pylsl.vectorstr([result]))
                    
                    self.classification_signal.emit(result)
                    self.epoch_signal.emit(self.epoch_data)

                    self.epoch_data.clear()

def update_data_and_plot(samples):
    global data, curves, data_length, offset
    # Process and plot the chunk of data efficiently
    for ch_index in range(8):
        for sample in samples:
            data[ch_index] = np.roll(data[ch_index], -1)
            data[ch_index][-1] = sample[ch_index]  # Assuming samples are structured [sample][channel]
        curves[ch_index].setData(np.arange(data_length), data[ch_index] + ch_index * offset)

def plot_epoch (epoch):
    global data, curves_epoch, data_length, offset

    # sort by time
    epoch = sorted(epoch, key=lambda x: x[0])

    # apply preprocessing
    # x, ys = vis_mne(epoch)
    # x = x.reshape(-1)

    # no preprocessing
    x  = np.array([tup[0] for tup in epoch]).reshape(-1)
    ys = np.array([tup[1] for tup in epoch])
    
    # update each curve (8 channels => 8 curves)
    for ch_index in range(8):
        y_offset = ch_index * offset
        curves_epoch[ch_index].setData(x, ys[:,ch_index] + y_offset)
        

if __name__ == '__main__':
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
    # offset_line_edit = QLineEdit()
    # offset_line_edit.setValidator(QIntValidator())

    layout.addWidget(win)  # Add window containing plots to the layout
    layout.addWidget(classification_label)  # Add the label to the layout
    layout.addWidget(timer_progbar)
    # layout.addWidget(offset_line_edit)
    
    mainWidget.setLayout(layout)  # Set the layout on the main widget
    mainWidget.show()  # Show the main widget

    # set globals
    num_channels = 8
    data_length = 500
    offset = 500
    data = [np.zeros(data_length) for _ in range(num_channels)] # 500 sample window to be displayed.

    pipeline = load('./csp_lda_pipeline.joblib')

    curves = [plot.plot(pen=pg.intColor(i)) for i in range(num_channels)]
    curves_epoch = [plot2.plot(pen=pg.intColor(i)) for i in range(num_channels)]

    # setup outlets
    eeg_inlet, result_outlet = init_lsl()

    # set data thread and signal connections
    data_thread = DataThread(epoch_length=1000, classify_interval=500)
    data_thread.data_signal.connect(update_data_and_plot)
    data_thread.classification_signal.connect(
        lambda result : classification_label.setText(f"Classification: {result}"))
    data_thread.timer_signal.connect(
        lambda time : timer_progbar.setValue(500 - time))
    data_thread.epoch_signal.connect(plot_epoch)
    data_thread.start()

    sys.exit(app.exec_())