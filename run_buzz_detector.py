import librosa
import pickle
import os
import shutil

import numpy as np

buzz_dest = '/Users/samorchard/Documents/InsectSound1000_training/automated_buzz_filter/test_buzzes/episyrphus'
no_buzz_dest = '/Users/samorchard/Documents/InsectSound1000_training/automated_buzz_filter/test_non_buzzes/episyrphus'
source_dir = '/Users/samorchard/Documents/InsectSound1000_training/single_channel_test/episyrphus'

# Load the buzz detector model saved as pickle
with open('buzz_detector.pkl', 'rb') as f:
    buzz_detector = pickle.load(f)

def extractMFCCs(path):
    files = [f for f in os.listdir(path) if not f.startswith('.')]
    samples = []
    for f in files:
        sig, sr = librosa.load(os.path.join(path, f), sr=16000, mono=True)
        sig = sig.flatten('F')[:sig.shape[0]]
        mfcc_mean = [np.mean(feature) for feature in librosa.feature.mfcc(y=sig, sr=16000)]
        samples.append((f, mfcc_mean))
    return samples

print('Extracting MFCCs...')
samples = extractMFCCs(source_dir)
print('Running prediction model...')
for sample in samples:
    src_file_path = os.path.join(source_dir, sample[0])
    pred = buzz_detector.predict(np.array(sample[1]).reshape(1, -1))
    if pred[0] == 1:
        shutil.copy(src_file_path, os.path.join(buzz_dest, sample[0]))
    else:
        shutil.copy(src_file_path, os.path.join(no_buzz_dest, sample[0]))


