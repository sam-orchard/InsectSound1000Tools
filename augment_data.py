import os
import librosa
import soundfile as sf  # Do we want to use librosa and soundfile? Or just use scipy.io.wavfile which seems more robust?
import random # Audiomentations uses builtin random module for generating random numbers

from audiomentations import AddBackgroundNoise, PolarityInversion

random.seed(97) # Set the seed for the random module so audiomentation processes are repeatable

def augment_audio_file(sig, sample_rate, noise_directory, min_absolute_rms_db=-45, max_absolute_rms_db=-15):
    # Create transform
    transform = AddBackgroundNoise(
        sounds_path=noise_directory,
        noise_transform=PolarityInversion(),
        noise_rms='absolute',
        min_absolute_rms_db=min_absolute_rms_db,
        max_absolute_rms_db=max_absolute_rms_db,
        p=1.0
    )
    augmented_data = transform(sig, sample_rate)

    return augmented_data

# def augment_audio_file(sig, sample_rate, noise_directory):
#     # Create transform
#     transform = AddBackgroundNoise(
#         sounds_path=noise_directory,
#         noise_transform=PolarityInversion(),
#         noise_rms='none', # Set to anything that isn't relative or absolute so it adds the noise at the original volume
#         p=1.0
#     )
#     augmented_data = transform(sig, sample_rate)
#
#     return augmented_data


input_dir = '/Users/samorchard/Documents/InsectSound1000_training/clean_single_channel_shifted_test'
output_dir = '/Users/samorchard/Documents/InsectSound1000_training/augmented_with_varied_amplitude_test'
noise_directory = '/Users/samorchard/Documents/background_soundscapes/silwood_data/augment_test'

labels = [
    'bombus',
    'episyrphus',
]

for label in labels:
    species_input_dir = os.path.join(input_dir, label)
    species_output_dir = os.path.join(output_dir, label)

    if not os.path.exists(species_output_dir):
        os.mkdir(species_output_dir)

    for file in os.listdir(species_input_dir):
        sig, rate = librosa.load(os.path.join(species_input_dir, file), sr=16000, mono=True, res_type='kaiser_fast')

        # TODO: Confirm that setting the seed within the augment_audio_file function won't cause exactly the same
        # transformation to be applied to every file. The seed should instead ensure the same augmentation occurs for
        # a given file every time it is run, but these augmentations should differ between files
        augmented_data = augment_audio_file(sig, rate, noise_directory=noise_directory, min_absolute_rms_db=-45, max_absolute_rms_db=-25)
        # augmented_data = augment_audio_file(sig, rate, noise_directory=noise_directory)
        sf.write(os.path.join(species_output_dir, file), augmented_data, rate)



