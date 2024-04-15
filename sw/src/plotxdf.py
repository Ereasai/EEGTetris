# mostly written by CHATGPT

import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def plot_eeg_with_event_markers(eeg_file_path, markers_file_path):
    # Load the EEG data file
    eeg_data = pd.read_csv(eeg_file_path)

    # Load the Markers data file
    markers_data = pd.read_csv(markers_file_path)

    # Plotting the EEG data with event markers
    plt.figure(figsize=(18, 20))

    # Plot each EEG channel with actual event names as labels
    for i, column in enumerate(eeg_data.columns[1:], start=1):
        plt.subplot(len(eeg_data.columns)-1, 1, i)
        plt.plot(eeg_data['Timestamp'], eeg_data[column], label=f'{column}')
        plt.ylabel('Amplitude')
        
        # Adding markers with actual event names from the second column
        for _, row in markers_data.iterrows():
            plt.axvline(x=row['Timestamp'], color='r', linestyle='--')
            # marker_id = np.rint().astype(int) # obtain marker id
            eventName = row['Channel_1']
            plt.text(row['Timestamp'], plt.ylim()[1], eventName, color='r', verticalalignment='bottom', rotation=45)

        plt.legend()

    plt.xlabel('Timestamp')
    plt.suptitle('EEG Data with Event Markers Labeled by Event Names')
    plt.tight_layout(rect=[0, 0.03, 1, 0.97])  # Adjust layout to make room for the title
    plt.show()

# Replace 'path_to_your_eeg_data.csv' and 'path_to_your_markers_data.csv' with the actual paths to your files

if (len(sys.argv) < 2):
    print('Usage: py plotxdf.py [eeg_filename] [markers_filename]')
    print('The file path are assumed to start in ./xdf-results.')
    print('So if you just used xdfToCSV, you do not have to specify the length.')
    exit()

eeg_fname = sys.argv[1]
marker_fname = sys.argv[2]
plot_eeg_with_event_markers(f'./xdf-results/{eeg_fname}', f'./xdf-results/{marker_fname}')