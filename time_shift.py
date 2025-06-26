"""
Pads a single channel audio file to specified duration
Applies time-shift to result and saves as a new .wav file
"""

import librosa
import random
import os
import numpy as np
import soundfile as sf

from math import floor, ceil
from audiomentations import Shift

# random.seed(5) # Seed for shifting training dataset
random.seed(10) # Seed for shifting test dataset

input_dir = '/Users/samorchard/Documents/InsectSound1000_training/single_channel_test'
output_dir = '/Users/samorchard/Documents/InsectSound1000_training/clean_single_channel_shifted_test'

labels = [
    'bombus',
    'episyrphus',
]

sample_rate = 16000 # Sample rate of input audio
duration = 3 # Duration in seconds

min_shift = -0.6
max_shift = 0.6

for label in labels:
    species_input_dir = os.path.join(input_dir, label)
    species_output_dir = os.path.join(output_dir, label)

    if not os.path.exists(species_output_dir):
        os.mkdir(species_output_dir)

    for file in os.listdir(species_input_dir):
        sig, rate = librosa.load(os.path.join(species_input_dir, file), sr=sample_rate, mono=True, res_type='kaiser_fast')

        # Pad the signal to be 3s in length
        total_samples = duration * sample_rate
        samples_to_add = total_samples - sig.size

        # pad_width argument is a tuple of samples to add (before, after). (0, N) will just add the samples
        # to the end of the signal. For center, it should be (N/2, N/2).
        padded_sig = np.pad(sig, (floor(samples_to_add / 2), ceil(samples_to_add / 2)), 'constant')

        # Randomly time-shift the signal
        shift = Shift(min_shift=min_shift, max_shift=max_shift, rollover=False, p=1.0)
        shifted_sig = shift(padded_sig, sample_rate)

        sf.write(os.path.join(species_output_dir, file), shifted_sig, sample_rate)