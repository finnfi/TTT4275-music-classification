from song_features import genre_id_to_string, readGenreClassData, getPointsAndClasses
from KNN import error_rate

from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import SelectKBest

import numpy as np
import copy
import matplotlib.pyplot as plt


# Choose features and genres:
features    = [ "zero_cross_rate_mean","zero_cross_rate_std","rmse_mean","rmse_var","spectral_centroid_mean","spectral_centroid_var","spectral_bandwidth_mean",
                "spectral_bandwidth_var","spectral_rolloff_mean","spectral_rolloff_var","spectral_contrast_mean","spectral_contrast_var","spectral_flatness_mean",
                "spectral_flatness_var","chroma_stft_1_mean","chroma_stft_2_mean","chroma_stft_3_mean","chroma_stft_4_mean","chroma_stft_5_mean","chroma_stft_6_mean",
                "chroma_stft_7_mean","chroma_stft_8_mean","chroma_stft_9_mean","chroma_stft_10_mean","chroma_stft_11_mean","chroma_stft_12_mean","chroma_stft_1_std",
                "chroma_stft_2_std","chroma_stft_3_std","chroma_stft_4_std","chroma_stft_5_std","chroma_stft_6_std","chroma_stft_7_std","chroma_stft_8_std","chroma_stft_9_std",
                "chroma_stft_10_std","chroma_stft_11_std","chroma_stft_12_std","tempo","mfcc_1_mean","mfcc_2_mean","mfcc_3_mean","mfcc_4_mean","mfcc_5_mean","mfcc_6_mean",
                "mfcc_7_mean","mfcc_8_mean","mfcc_9_mean","mfcc_10_mean","mfcc_11_mean","mfcc_12_mean","mfcc_1_std","mfcc_2_std","mfcc_3_std","mfcc_4_std","mfcc_5_std",
                "mfcc_6_std","mfcc_7_std","mfcc_8_std","mfcc_9_std","mfcc_10_std","mfcc_11_std","mfcc_12_std"]
genres          = ["pop","metal", "disco", "blues", "reggae", "classical", "rock", "hiphop", "country", "jazz"]

# Choose how many features to use
k = 14

# Import song feature
songs_dict = readGenreClassData("Data/GenreClassData_30s.txt")

# Init dict for best models for each genre
best_gmms = dict() # Dict from genre to best gmm for that genre

# Create scalers and kbest transform
X_train, y_train, ids_train  = getPointsAndClasses(songs_dict,features, genres, "Train")
scaler  = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
k_best  = SelectKBest(k=k)
X_train = k_best.fit_transform(X_train, y_train)

# Print names of feature chosen
feature_map = k_best.get_feature_names_out()
for f in feature_map:
    idx = int(f[1:])
    print(features[idx])

# Define # of components
model_sizes = {'pop': 1, 'metal': 1, 'disco': 8, 'blues':2, 'reggae':1, 'classical':1, 'rock': 4, 'hiphop': 6, 'country': 1, 'jazz':3}

# Create appropriate model
for genre in genres:
    # Here we fit a GM
    X_train, y_train, ids_train  = getPointsAndClasses(songs_dict,features, [genre], "Train")
    X_train = scaler.transform(X_train)
    X_train = k_best.transform(X_train)
    gmm = GaussianMixture(n_components=model_sizes[genre], covariance_type="full",init_params='kmeans',random_state=1)
    gmm.fit(X_train)
    best_gmms[genre] = copy.deepcopy(gmm)

# Get test set
X_test, y_test, ids  = getPointsAndClasses(songs_dict,features, genres, "Test")
# Scale and extract k best features
X_test = scaler.transform(X_test)
X_test = k_best.transform(X_test)

# Compute log likelihood for every function
log_likelihood = np.zeros((len(y_test),len(genres)))
for genre, model in best_gmms.items():
    log_likelihood[:,genres.index(genre)]  = model.score_samples(X_test)

# Find index of best log likelihood
maxVal_ind_rowise = np.argmax(log_likelihood, axis=1)

# Create confusion matrix
cm = np.zeros((len(genres), len(genres)))
for i in range(len(y_test)):
    cm[genres.index(genre_id_to_string(y_test[i])), maxVal_ind_rowise[i]] += 1

# Display confusion matrices and print error rate
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=genres)
disp.plot(cmap=plt.cm.Blues,xticks_rotation=45)
plt.show()

print("Error rate: ", error_rate(cm))

