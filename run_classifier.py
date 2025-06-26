import librosa
import pickle
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import classification_report, ConfusionMatrixDisplay

episyrphus_test_dir = '/Users/samorchard/Documents/InsectSound1000_training/augmented_with_varied_amplitude_test/episyrphus'
bombus_test_dir = '/Users/samorchard/Documents/InsectSound1000_training/augmented_with_varied_amplitude_test/bombus'

# Load the buzz detector model saved as pickle
with open('augmented_buzz_classifier.pkl', 'rb') as f:
    buzz_classifier = pickle.load(f)

def extractMFCCs(path):
    files = [f for f in os.listdir(path) if not f.startswith('.')]
    samples = []
    for f in files:
        sig, sr = librosa.load(os.path.join(path, f), sr=16000, mono=True)
        sig = sig.flatten('F')[:sig.shape[0]]
        mfcc_mean = [np.mean(feature) for feature in librosa.feature.mfcc(y=sig, sr=16000)]
        samples.append(mfcc_mean)
    return samples

print('Extracting MFCCs...')
episyrphus_samples = extractMFCCs(episyrphus_test_dir)
bombus_samples = extractMFCCs(bombus_test_dir)

bombus_buzz_df = pd.DataFrame(bombus_samples)
bombus_buzz_df['Class'] = 'bombus'
episyrphus_buzz_df = pd.DataFrame(episyrphus_samples)
episyrphus_buzz_df['Class'] = 'episyrphus'

df = pd.concat([bombus_buzz_df, episyrphus_buzz_df], axis=0)

X = df.drop('Class', axis=1)
y_true = df['Class']

print('Running prediction model...')
y_pred = buzz_classifier.predict(X)

cm_display = ConfusionMatrixDisplay.from_predictions(y_true, y_pred, labels=['bombus', 'episyrphus'])

clf_report = classification_report(y_true, y_pred, output_dict=True)
print(pd.DataFrame(clf_report))

plt.show()
# for sample in samples:
#     print(sample)
#     src_file_path = os.path.join(test_dir, sample[0])
#     pred = buzz_classifier.predict(np.array(sample[1]).reshape(1, -1))
#     print(pred)


