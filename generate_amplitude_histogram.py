import os

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

output_dir = './amplitude_histogram/log_max_amplitude'
if not os.path.isdir(output_dir):
    os.mkdir(output_dir)

metadata_file_path = './InsectSound1000_metadata.csv'

df = pd.read_csv(metadata_file_path)

species_list = df['Species'].unique()

# If True, calculates the dbFS of the max amplitude. Otherwise, just use max amplitude values.
use_dbFS = True

for species in species_list:
    species_df = df[df['Species'] == species].reset_index(drop=True)
    species_df['dbFS'] = 20 * np.log10(species_df['Max Amp.'] / species_df['Max Amp.'].max())

    plt.figure()
    plt.ylabel('Count')
    if use_dbFS:
        plt.title(f'{species} dbFS')
        plt.xlabel('dbFS')
        # plt.hist(species_df['dbFS'])
        plt.scatter(range(0, len(species_df['dbFS'])), species_df['dbFS'].sort_values())
        filename = f"{output_dir}/{species.replace(' ', '_')}_dbFS_histogram.png"
    else:
        plt.title(f'{species} Log Max. Amplitude')
        plt.xlabel('$log_10$(Max. Amplitude)')
        plt.hist(np.log10(species_df['Max Amp.']))
        filename = f"{output_dir}/{species.replace(' ', '_')}_log_max_amplitude_histogram.png"


    # plt.savefig(filename)
plt.show()


