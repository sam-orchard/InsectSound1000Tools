import os
import numpy as np
import librosa
import soundfile as sf

np.random.seed(18)

test_or_train = 'train'
species_list = [
    'bombus',
    'episyrphus'
]

sample_rate = 16000

for species in species_list:
    for dataset in ['train', 'test']:

        input_dir = f'../InsectSound1000_training/{dataset}/{species}'
        output_dir = f'../InsectSound1000_training/single_channel_{dataset}/{species}'

        audio_files = os.listdir(input_dir)
        for f in audio_files:
            # Select a random number between 0 and 3 corresponding to the index of the selected channel
            channel_index = np.random.randint(0, 4)
            sig, sr = librosa.load(os.path.join(input_dir, f), sr=sample_rate, mono=False)

            # Rewrite the selected channel as a new wav file
            sf.write(os.path.join(output_dir, f), sig[channel_index], sample_rate)