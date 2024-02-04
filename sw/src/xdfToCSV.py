import os
import pyxdf
import pandas as pd

# Define the source and target directories
source_dir = './xdf'
target_dir = './xdf-results'

# Create the target directory if it doesn't exist
if not os.path.exists(target_dir):
    os.makedirs(target_dir)

# List all XDF files in the source directory
xdf_files = [f for f in os.listdir(source_dir) if f.endswith('.xdf')]

for xdf_file in xdf_files:
    # Construct the full path to the current XDF file
    xdf_path = os.path.join(source_dir, xdf_file)
    
    # Load the XDF file
    streams, header = pyxdf.load_xdf(xdf_path)
    
    # Assuming the first stream contains the EEG data you're interested in
    eeg_data = streams[0]['time_series']
    timestamps = streams[0]['time_stamps']
    channel_names = ['Channel_' + str(i+1) for i in range(eeg_data.shape[1])]
    
    # Create a DataFrame
    df = pd.DataFrame(eeg_data, columns=channel_names)
    df.insert(0, 'Timestamp', timestamps)
    
    # Construct the CSV file name based on the XDF file name
    csv_file_name = xdf_file.replace('.xdf', '.csv')
    csv_path = os.path.join(target_dir, csv_file_name)
    
    # Save the DataFrame to a CSV file
    df.to_csv(csv_path, index=False)
    
    print(f'Data from {xdf_file} successfully saved to {csv_path}')
