import librosa
import matplotlib.pyplot as plt
import numpy as np

filename = '2023427-16-31_Bombus_terrestris_000000_s61_ch3'
clean_input_file = f'/Users/samorchard/Documents/InsectSound1000_training/automated_buzz_filter/test_buzzes/bombus/{filename}.wav'
noise_input_file = f'/Users/samorchard/Documents/InsectSound1000_training/automated_buzz_filter/augmented_test_buzzes/bombus/{filename}.wav'

sig, sr = librosa.load(clean_input_file, sr=16000, mono=True)

S = librosa.feature.melspectrogram(y=sig, sr=sr, n_mels=128, fmax=8000)

fig, ax = plt.subplots()
S_dB = librosa.power_to_db(S, ref=np.max)
img = librosa.display.specshow(S_dB, x_axis='time', y_axis='mel', sr=sr, fmax=8000, ax=ax)
fig.colorbar(img, ax=ax, format='%+2.0f dB')

sig, sr = librosa.load(noise_input_file, sr=16000, mono=True)

S = librosa.feature.melspectrogram(y=sig, sr=sr, n_mels=128, fmax=8000)

fig, ax = plt.subplots()
S_dB = librosa.power_to_db(S, ref=np.max)
img = librosa.display.specshow(S_dB, x_axis='time', y_axis='mel', sr=sr, fmax=8000, ax=ax)
fig.colorbar(img, ax=ax, format='%+2.0f dB')

plt.show()
