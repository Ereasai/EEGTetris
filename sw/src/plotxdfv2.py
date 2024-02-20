import pandas as pd
import matplotlib.pyplot as plt

eeg_data = pd.read_csv('./xdf-results/t0_eeg.csv') ## EEG DATA
markers_data = pd.read_csv('./xdf-results/t0_markers.csv') # MARKERS 

# Convert timestamps to relative time in seconds for easier interpretation
eeg_data['Relative_Time'] = eeg_data['Timestamp'] - eeg_data['Timestamp'].iloc[0]
markers_data['Relative_Time'] = markers_data['Timestamp'] - eeg_data['Timestamp'].iloc[0]

# Identify epochs (start and end times) and label them with the task name
epochs = []
for index, row in markers_data.iterrows():
    if 'END' not in row['Channel_1']:  # Check if it's a start of an epoch
        task_name = row['Channel_1']
        for next_index in range(index + 1, len(markers_data)):
            if markers_data.iloc[next_index]['Channel_1'] == task_name + '_END':
                start_time = row['Relative_Time']
                end_time = markers_data.iloc[next_index]['Relative_Time']
                epochs.append((start_time, end_time, task_name))
                break  # Break after finding the matching end marker


# Plotting with vertical offsets and different colors
plt.figure(figsize=(15, 10))
offset = 0  # Starting offset
offset_step = 1000  # Step to separate each channel for clarity
color_palette = plt.cm.tab10.colors  # Using tab10 colormap for distinct colors

for i, color in zip(range(1, 9), color_palette):
    plt.plot(eeg_data['Relative_Time'], eeg_data[f'Channel_{i}'] + offset, 
             label=f'Channel {i}', color=color, linewidth=1)
    offset += offset_step

# Highlight epochs and label them with the task name
for start_time, end_time, task_name in epochs:
    plt.axvspan(start_time, end_time, color='yellow', alpha=0.3)
    # Place the task name label at the start of the epoch
    plt.text((start_time + end_time)/2, offset, task_name.replace('_', ' '), ha='center', va='bottom')

# Dotted lines for each event (no labels)
label_height = offset + 50  # Adjust label height to be above the channels
for index, marker in markers_data.iterrows():
    plt.axvline(x=marker['Relative_Time'], color='k', linestyle='--')
    

plt.xlabel('Time (seconds)')
plt.ylabel('Amplitude + Offset')
plt.show()