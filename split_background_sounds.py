import os
import librosa
import soundfile as sf

split_duration = 3 # 3 second duration for output files
sample_rate = 48000
noise_path = '/Users/samorchard/Documents/background_soundscapes/silwood_data/augment_test'
output_path = '/Users/samorchard/Documents/InsectSound1000_training/background_test_full_dataset'

if not os.path.isdir(output_path):
    os.mkdir(output_path)

for f in os.listdir(noise_path):
    if f.lower().endswith('.wav'):
        file_path = os.path.join(noise_path, f)
        file_name, file_extension = os.path.splitext(file_path)
        sig, sr = librosa.load(file_path, sr=sample_rate, mono=True)
        chunk_size = split_duration * sample_rate # Size of audio chunks in samples
        n_chunks = int(sig.size / chunk_size) # int to always round down to max number of full 3-second chunks
        for i in range(0, n_chunks):
            chunk = sig[i * chunk_size:(i+1) * chunk_size]
            sf.write(os.path.join(output_path, f.replace(file_extension, f'_{i}{file_extension}')), chunk, sample_rate)
