import os
from scipy.fft import fft


for f in no_buzz_filepaths:
    sig, rate = librosa.load(f, sr=16000, mono=True)
    T = 1/16000
    N = len(sig)
    sig_fft = fft(sig)
    xf = fftfreq(N, T)[:N//2]
    energy = np.sum(np.abs(sig_fft[:N//2][np.where((xf >= 100) & (xf <= 400))])**2)
    if energy > 100:
            buzzes_detected.append(f)
    else:
            no_buzzes_detected.append(f)


for f in no_buzz_filepaths:
    sig, rate = librosa.load(f, sr=16000, mono=True)
    T = 1/16000
    N = len(sig)
    sig_fft = fft(sig)
    xf = fftfreq(N, T)[:N//2]
    energy = np.sum(np.abs(sig_fft[:N//2][np.where((xf >= 100) & (xf <= 400))])**2)
    if energy > max_no_buzz_energy:
        max_no_buzz_energy = energy
        max_no_buzz_energy_filepath = f


for f in buzz_filepaths:
    sig, rate = librosa.load(f, sr=16000, mono=True)
    T = 1/16000
    N = len(sig)
    sig_fft = fft(sig)
    xf = fftfreq(N, T)[:N//2]
    energy = np.sum(np.abs(sig_fft[:N//2][np.where((xf >= 100) & (xf <= 400))])**2)
    buzz_energies.append(energy)