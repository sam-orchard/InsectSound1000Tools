import os
import re
import natsort
import pandas as pd
import numpy as np

from datetime import datetime
from scipy.io import wavfile


input_dir = '/Users/samorchard/Documents/InsectSound1000_full/'

output_path = './InsectSound1000_metadata.csv'

regex = re.compile(r'(\d+-\d+-\d+)_([a-zA-Z]+_[a-zA-Z]+)')
date_format = '%Y%m%d-%H-%M'

columns = ['Filename', 'Date', 'Time', 'Species', 'Length (s)', 'Sample Rate (Hz)', 'Max Amp.']

df = pd.DataFrame(columns=columns)

for file in natsort.natsorted(os.listdir(input_dir)):
    row_dict = {}
    match = regex.match(file)
    if match is not None:
        date_time, species = match.groups()
        date_time_object = datetime.strptime(date_time, date_format)
        species = species.replace('_', ' ')

        # Add to row_dict
        row_dict['Filename'] = file
        row_dict['Date'] = date_time_object.date()
        row_dict['Time'] = date_time_object.time()
        row_dict['Species'] = species
    else:
        print(f'WARNING: Info could not be extracted from file {file}')

    sample_rate, data = wavfile.read(os.path.join(input_dir, file))
    max_amplitude = np.max(data)
    num_samples, channels = data.shape
    length = num_samples / sample_rate

    row_dict['Length (s)'] = length
    row_dict['Sample Rate (Hz)'] = sample_rate
    row_dict['Max Amp.'] = max_amplitude

    # Could probably be done in a way to avoid having to create a dict, then a list?
    df.loc[len(df)] = list(row_dict.values())

df['Date'] = df['Date'].apply(lambda x: x.strftime('%Y/%m/%d'))
df['Time'] = df['Time'].apply(lambda x: x.strftime('%H:%M'))

df.to_csv(output_path, index=False)




