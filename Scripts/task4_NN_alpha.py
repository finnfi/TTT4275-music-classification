from song_features import readGenreClassData, getPointsAndClasses
from KNN import error_rate

from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import MinMaxScaler

import matplotlib.pyplot as plt

import numpy as np


# Import song feature
songs_dict = readGenreClassData("Data/GenreClassData_30s.txt")

# Choose features and genres:
features    = [ "zero_cross_rate_mean","zero_cross_rate_std","rmse_mean","rmse_var","spectral_centroid_mean","spectral_centroid_var","spectral_bandwidth_mean",
                "spectral_bandwidth_var","spectral_rolloff_mean","spectral_rolloff_var","spectral_contrast_mean","spectral_contrast_var","spectral_flatness_mean",
                "spectral_flatness_var","chroma_stft_1_mean","chroma_stft_2_mean","chroma_stft_3_mean","chroma_stft_4_mean","chroma_stft_5_mean","chroma_stft_6_mean",
                "chroma_stft_7_mean","chroma_stft_8_mean","chroma_stft_9_mean","chroma_stft_10_mean","chroma_stft_11_mean","chroma_stft_12_mean","chroma_stft_1_std",
                "chroma_stft_2_std","chroma_stft_3_std","chroma_stft_4_std","chroma_stft_5_std","chroma_stft_6_std","chroma_stft_7_std","chroma_stft_8_std","chroma_stft_9_std",
                "chroma_stft_10_std","chroma_stft_11_std","chroma_stft_12_std","tempo","mfcc_1_mean","mfcc_2_mean","mfcc_3_mean","mfcc_4_mean","mfcc_5_mean","mfcc_6_mean",
                "mfcc_7_mean","mfcc_8_mean","mfcc_9_mean","mfcc_10_mean","mfcc_11_mean","mfcc_12_mean","mfcc_1_std","mfcc_2_std","mfcc_3_std","mfcc_4_std","mfcc_5_std",
                "mfcc_6_std","mfcc_7_std","mfcc_8_std","mfcc_9_std","mfcc_10_std","mfcc_11_std","mfcc_12_std"]
genres      = ["pop","metal", "disco", "blues", "reggae", "classical", "rock", "hiphop", "country", "jazz"]

# Extract training
X_train, y_train, ids_train  = getPointsAndClasses(songs_dict,features, genres, "Train")

# Scale
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)

# Initialise arrays for error rates
error_rates_train = []
error_rates_test = []

# Make list of alphas to test
alphas = np.logspace(-5,2,num=8, base = 10.0)

# Test for each alpha
for alpha in alphas:
    #Create classifier
    clf = MLPClassifier(solver='lbfgs', alpha=alpha,activation="logistic", max_iter=100000,verbose=True,
                                    hidden_layer_sizes=(40,), random_state=1)
    clf.fit(X_train, y_train)

    # Evaluate for alpha
    X_test, y_test, ids_test  = getPointsAndClasses(songs_dict,features, genres, "Test")
    X_test = scaler.transform(X_test)

    y_pred_test = clf.predict(X_test)
    y_pred_train = clf.predict(X_train)
    cm_test = confusion_matrix(y_test, y_pred_test)
    cm_train = confusion_matrix(y_train, y_pred_train)
    error_rates_test.append(error_rate(cm_test))
    error_rates_train.append(error_rate(cm_train))

# Plot result for each alpha
fig = plt.figure()
ax = fig.add_subplot()
ax.plot(alphas, np.array(error_rates_train), color="red",label="Training set")
ax.plot(alphas, np.array(error_rates_test), color="blue", label="Test set")
ax.legend()
ax.set_xlabel("\u03B1")
ax.set_ylabel("Error rate")
ax.set_xscale('log')
plt.show()