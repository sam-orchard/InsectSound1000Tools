import os
import librosa
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import classification_report, roc_curve, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn import svm

bombus_buzz_path = '/Users/samorchard/Documents/InsectSound1000_training/augmented_with_varied_amplitude_train/bombus'
episyrphus_buzz_path = '/Users/samorchard/Documents/InsectSound1000_training/augmented_with_varied_amplitude_train/episyrphus'

def extractMFCCs(path):
    files = [f for f in os.listdir(path) if not f.startswith('.')]
    samples = []
    for f in files[::4]:
        sig, sr = librosa.load(os.path.join(path, f), sr=16000, mono=True)
        sig = sig.flatten('F')[:sig.shape[0]]
        mfcc_mean = [np.mean(feature) for feature in librosa.feature.mfcc(y=sig, sr=16000)]
        samples.append(mfcc_mean)
    return samples

bombus_buzz_samples = extractMFCCs(bombus_buzz_path)
episyrphus_buzz_samples = extractMFCCs(episyrphus_buzz_path)

bombus_buzz_df = pd.DataFrame(bombus_buzz_samples)
bombus_buzz_df['Class'] = 'bombus'
episyrphus_buzz_df = pd.DataFrame(episyrphus_buzz_samples)
episyrphus_buzz_df['Class'] = 'episyrphus'

df = pd.concat([bombus_buzz_df, episyrphus_buzz_df], axis=0)
print(df.head())

X = df.drop('Class', axis=1)
y = df['Class']

scaler = MinMaxScaler()
scaler.fit(X)
X_scaled = scaler.transform(X)


#
# scaler = StandardScaler()
# scaler.fit(X)
# X_scaled = scaler.transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
print(X_train.shape, X_test.shape)
#
svm_params = {'C': [0.001, 0.01, 0.1, 1, 10], 'gamma': [0.01, 0.001, 0.0001], 'kernel': ['rbf', 'poly', 'sigmoid', 'linear']}

svm_grid = GridSearchCV(svm.SVC(probability=True), svm_params, refit=True, verbose=3, cv=StratifiedKFold(5))

svm_pipeline = Pipeline([('scaler', StandardScaler()), ('clf', svm_grid)])
svm_pipeline.fit(X_train, y_train)
y_pred = svm_pipeline.predict(X_test)

clf_report = classification_report(y_test, y_pred, output_dict=True)
print(pd.DataFrame(clf_report))

probs = svm_pipeline.predict_proba(X_test)
print(probs)
fpr, tpr, thresholds = roc_curve(y_test, probs[:, 1], pos_label='bombus')
auc = roc_auc_score(y_test,probs[:, 1])
print(auc)
plt.figure()
plt.plot(fpr, tpr)
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.show()

with open('augmented_buzz_classifier.pkl', 'wb') as f:
    pickle.dump(svm_pipeline, f)