from song_features import getPointsAndClasses
from song_features import readGenreClassData
from KNN import KNNClassifier
from itertools import combinations
from sklearn.model_selection import StratifiedKFold
import numpy as np
from sklearn.metrics import ConfusionMatrixDisplay
from matplotlib import pyplot as plt
from sklearn.preprocessing import MinMaxScaler



# Import song feature
songs_dict  = readGenreClassData("Data/GenreClassData_30s.txt")

# Define genres to use
genres      = ["pop","metal", "disco", "blues", "reggae", "classical", "rock", "hiphop", "country", "jazz"]

# Define splits to use in kfold cross-validation
n_splits = 4

# Find best feature space: 
feature_pool_1 = ["spectral_centroid_mean","mfcc_1_mean","spectral_rolloff_mean","tempo"]
feature_pool_2 = ["zero_cross_rate_mean","zero_cross_rate_std","rmse_mean","rmse_var","spectral_centroid_mean","spectral_centroid_var","spectral_bandwidth_mean",
"spectral_bandwidth_var","spectral_rolloff_mean","spectral_rolloff_var","spectral_contrast_mean","spectral_contrast_var","spectral_flatness_mean",
"spectral_flatness_var","chroma_stft_1_mean","chroma_stft_2_mean","chroma_stft_3_mean","chroma_stft_4_mean","chroma_stft_5_mean","chroma_stft_6_mean",
"chroma_stft_7_mean","chroma_stft_8_mean","chroma_stft_9_mean","chroma_stft_10_mean","chroma_stft_11_mean","chroma_stft_12_mean","chroma_stft_1_std",
"chroma_stft_2_std","chroma_stft_3_std","chroma_stft_4_std","chroma_stft_5_std","chroma_stft_6_std","chroma_stft_7_std","chroma_stft_8_std","chroma_stft_9_std",
"chroma_stft_10_std","chroma_stft_11_std","chroma_stft_12_std","tempo","mfcc_1_mean","mfcc_2_mean","mfcc_3_mean","mfcc_4_mean","mfcc_5_mean","mfcc_6_mean",
"mfcc_7_mean","mfcc_8_mean","mfcc_9_mean","mfcc_10_mean","mfcc_11_mean","mfcc_12_mean","mfcc_1_std","mfcc_2_std","mfcc_3_std","mfcc_4_std","mfcc_5_std",
"mfcc_6_std","mfcc_7_std","mfcc_8_std","mfcc_9_std","mfcc_10_std","mfcc_11_std","mfcc_12_std"]
feature_pool_1_combinations = list(combinations(feature_pool_1,3))

error_rate = 1
chosen_features = []

for comb in feature_pool_1_combinations:
    for feature in feature_pool_2:
        features = list(comb) + [feature]
        X_train, y_train, ids_train  = getPointsAndClasses(songs_dict,features, genres, "Train")
        scaler = MinMaxScaler()
        X_train = scaler.fit_transform(X_train)
        skf = StratifiedKFold(n_splits=n_splits)
        skf.get_n_splits(X_train, y_train)
        errors = np.zeros(n_splits)
        ei = 0
        for train_index, test_index in skf.split(X_train, y_train):
            knn = KNNClassifier(X_train[train_index,:], y_train[train_index], np.array(ids_train)[train_index], features, 5 ,"min_max")
            cm, cm_list, er = knn.evaluate(X_train[test_index,:].copy(), y_train[test_index].copy(), np.array(ids_train)[test_index].copy())
            errors[ei] = er
            ei = ei + 1
        
        avg_error = np.average(errors)
        if avg_error < error_rate:
            error_rate = avg_error
            chosen_features = features


print("Selected features and average error rate: ")
print(chosen_features)
print(error_rate)

# Run on test set
X_train, y_train, ids_train  = getPointsAndClasses(songs_dict,chosen_features, genres, "Train")
X_test, y_test, ids_test  = getPointsAndClasses(songs_dict,chosen_features, genres, "Test")

knn = KNNClassifier(X_train, y_train, ids_train,chosen_features, 5 ,"min_max")
cm, cm_list, er = knn.evaluate(X_test, y_test, np.array(ids_test).copy())

print("Error rate: ", er)

# Plot confusion matrices
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=knn.classes)
disp.plot(cmap=plt.cm.Blues,xticks_rotation=45)
plt.show()
