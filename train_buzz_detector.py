import os
import librosa
import pickle
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, StratifiedKFold
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import classification_report, roc_curve, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn import svm
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.decomposition import PCA

bombus_buzz_path = '/Users/samorchard/Documents/InsectSound1000_training/training_buzzes/bombus'
bombus_no_buzz_path = '/Users/samorchard/Documents/InsectSound1000_training/training_non_buzzes/bombus'
episyrphus_buzz_path = '/Users/samorchard/Documents/InsectSound1000_training/training_buzzes/episyrphus'
episyrphus_no_buzz_path = '/Users/samorchard/Documents/InsectSound1000_training/training_non_buzzes/episyrphus'

def get_scaled_tsne_embeddings(features, perplexity=30.0):
    embedding = TSNE(n_components=2, perplexity=perplexity, random_state=42).fit_transform(features)
    # scaler = MinMaxScaler()
    # scaler.fit(embedding)
    # return np.array(scaler.transform(embedding))
    return np.array(embedding)

def extractMFCCs(path):
    files = [f for f in os.listdir(path) if not f.startswith('.')]
    samples = []
    for f in files:
        sig, sr = librosa.load(os.path.join(path, f), sr=16000, mono=True)
        sig = sig.flatten('F')[:sig.shape[0]]
        mfcc_mean = [np.mean(feature) for feature in librosa.feature.mfcc(y=sig, sr=16000)]
        samples.append(mfcc_mean)
    return samples

# bombus_buzz_samples = extractMFCCs(bombus_buzz_path)
# bombus_no_buzz_samples = extractMFCCs(bombus_no_buzz_path)

episyrphus_buzz_samples = extractMFCCs(episyrphus_buzz_path)
episyrphus_no_buzz_samples = extractMFCCs(episyrphus_no_buzz_path)

buzz_samples = episyrphus_buzz_samples
# buzz_samples = bombus_buzz_samples + episyrphus_buzz_samples
no_buzz_samples = episyrphus_no_buzz_samples
# no_buzz_samples = bombus_no_buzz_samples + episyrphus_no_buzz_samples

# print(len(buzz_samples), len(bombus_buzz_samples), len(episyrphus_buzz_samples))
# print(len(no_buzz_samples), len(bombus_no_buzz_samples), len(episyrphus_no_buzz_samples))

buzz_df = pd.DataFrame(buzz_samples)
buzz_df['Class'] = 1
no_buzz_df = pd.DataFrame(no_buzz_samples)
no_buzz_df['Class'] = 0

df = pd.concat([buzz_df, no_buzz_df], axis=0)
print(df.head())

X = df.drop('Class', axis=1)
y = df['Class']
labels = df['Class'].apply(lambda x: 'Buzz' if x == 1 else 'No Buzz')
print(labels)

scaler = MinMaxScaler()
scaler.fit(X)
X_scaled = scaler.transform(X)

tsne_embeddings = get_scaled_tsne_embeddings(X_scaled, perplexity=50)
tsne_result_df = pd.DataFrame({'tsne_1': tsne_embeddings[:, 0], 'tsne_2': tsne_embeddings[:, 1], 'label': labels})
print(tsne_result_df)

fig, ax = plt.subplots(1)
sns.scatterplot(x='tsne_1', y='tsne_2', hue='label', data=tsne_result_df, ax=ax, s=120)
# lim = (tsne_result_df.min() - 5, tsne_result_df.max() + 5)
# ax.set_xlim(lim)
# ax.set_ylim(lim)
ax.set_aspect('equal')
ax.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.0)

#
# scaler = StandardScaler()
# scaler.fit(X)
# X_scaled = scaler.transform(X)
pca = PCA(n_components=2)
pca.fit(X_scaled)
X_pca = pca.transform(X_scaled)

plt.figure()
plt.title('PCA')
plt.scatter(X_pca[y==0, 0], X_pca[y==0, 1], label='No Buzz')
plt.scatter(X_pca[y==1, 0], X_pca[y==1, 1], label='Buzz')
plt.gca().set_aspect("equal")
plt.xlabel("First principal component")
plt.ylabel("Second principal component")
plt.legend()

#
#
kmeans = KMeans(n_clusters=2)
kmeans.fit(X_scaled)
plt.figure()
plt.scatter(tsne_embeddings[:, 0], tsne_embeddings[:, 1], c=kmeans.labels_, alpha=0.5)
#
dbscan = DBSCAN()
clusters=dbscan.fit_predict(X_scaled)
plt.figure()
plt.scatter(tsne_embeddings[:, 0], tsne_embeddings[:, 1], c=clusters)
#
agg = AgglomerativeClustering(n_clusters=2)
assignment = agg.fit_predict(X_scaled)
plt.figure()
plt.scatter(tsne_embeddings[:, 0], tsne_embeddings[:, 1], c=assignment, alpha=0.5)

plt.show()
#
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
print(X_train.shape, X_test.shape)
#
# svm_params = {'C': [0.001, 0.01, 0.1, 1, 10], 'gamma': [0.01, 0.001, 0.0001], 'kernel': ['rbf', 'poly', 'sigmoid', 'linear']}
#
# svm_grid = GridSearchCV(svm.SVC(probability=True), svm_params, refit=True, verbose=3, cv=StratifiedKFold(5))
#
# svm_pipeline = Pipeline([('scaler', StandardScaler()), ('clf', svm_grid)])
# svm_pipeline.fit(X_train, y_train)
# y_pred = svm_pipeline.predict(X_test)
#
#
# clf_report = classification_report(y_test, y_pred, output_dict=True)
# print(pd.DataFrame(clf_report))
#
# probs = svm_pipeline.predict_proba(X_test)
# print(probs)
# fpr, tpr, thresholds = roc_curve(y_test, probs[:, 1])
# auc = roc_auc_score(y_test,probs[:, 1])
# print(auc)
# plt.figure()
# plt.plot(fpr, tpr)
# plt.xlabel("False Positive Rate")
# plt.ylabel("True Positive Rate")
# plt.show()
#
# with open('buzz_detector.pkl', 'wb') as f:
#     pickle.dump(svm_pipeline, f)