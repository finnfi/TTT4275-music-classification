from song_features import readGenreClassData, getPointsAndClasses
from KNN import KNNClassifier

from sklearn.feature_selection import SelectKBest
from sklearn.metrics import ConfusionMatrixDisplay

import numpy as np
import matplotlib.pyplot as plt

# Import song feature
songs_dict  = readGenreClassData("Data/GenreClassData_30s.txt")

# Define genres to use
genres      = ["pop","metal", "disco", "blues", "reggae", "classical", "rock", "hiphop", "country", "jazz"]

# Define feature pools: 
feature_pool_1 = ["spectral_centroid_mean","mfcc_1_mean","spectral_rolloff_mean","tempo"]
feature_pool_2 = ["zero_cross_rate_mean","zero_cross_rate_std","rmse_mean","rmse_var","spectral_centroid_mean","spectral_centroid_var","spectral_bandwidth_mean",
"spectral_bandwidth_var","spectral_rolloff_mean","spectral_rolloff_var","spectral_contrast_mean","spectral_contrast_var","spectral_flatness_mean",
"spectral_flatness_var","chroma_stft_1_mean","chroma_stft_2_mean","chroma_stft_3_mean","chroma_stft_4_mean","chroma_stft_5_mean","chroma_stft_6_mean",
"chroma_stft_7_mean","chroma_stft_8_mean","chroma_stft_9_mean","chroma_stft_10_mean","chroma_stft_11_mean","chroma_stft_12_mean","chroma_stft_1_std",
"chroma_stft_2_std","chroma_stft_3_std","chroma_stft_4_std","chroma_stft_5_std","chroma_stft_6_std","chroma_stft_7_std","chroma_stft_8_std","chroma_stft_9_std",
"chroma_stft_10_std","chroma_stft_11_std","chroma_stft_12_std","tempo","mfcc_1_mean","mfcc_2_mean","mfcc_3_mean","mfcc_4_mean","mfcc_5_mean","mfcc_6_mean",
"mfcc_7_mean","mfcc_8_mean","mfcc_9_mean","mfcc_10_mean","mfcc_11_mean","mfcc_12_mean","mfcc_1_std","mfcc_2_std","mfcc_3_std","mfcc_4_std","mfcc_5_std",
"mfcc_6_std","mfcc_7_std","mfcc_8_std","mfcc_9_std","mfcc_10_std","mfcc_11_std","mfcc_12_std"]

# Get points and classes
X_train, y_train, ids_train  = getPointsAndClasses(songs_dict,feature_pool_2, genres, "Train")

# Define k_best object
k_best= SelectKBest(k="all")
k_best.fit(X_train, y_train)

# Get scores for each feature and index
scores = np.array(k_best.scores_)
scores_and_index = [(i, scores[i]) for i in range(len(scores))]
scores_and_index = sorted(scores_and_index, key=lambda tup: tup[1], reverse=True)

# Find best features
features = []
joker_taken = False

for si in scores_and_index:
    idx = si[0]
    feature = feature_pool_2[idx]

    if feature in feature_pool_1:
        features.append(feature)
    elif joker_taken is False: 
        features.append(feature)
        joker_taken = True

    if len(features) == 4:
        break

print("Chosen featues: ", features)

# Get points and classes
X_train, y_train, ids_train     = getPointsAndClasses(songs_dict,features, genres, "Train")
X_test, y_test, ids_test        = getPointsAndClasses(songs_dict,features, genres, "Test")

# Do knn classification
knn = KNNClassifier(X_train, y_train, ids_train, features, 5 ,"min_max")
confusion_matrix_our, confusion_matrix_list_our, error_rate = knn.evaluate(X_test.copy(),y_test.copy(),ids_test.copy())  

print("Error rate: ", error_rate)

# Plot confusion matrices
disp = ConfusionMatrixDisplay(confusion_matrix=confusion_matrix_our, display_labels=knn.classes)
disp.plot(cmap=plt.cm.Blues,xticks_rotation=45)
plt.show()