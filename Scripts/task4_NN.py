from sklearn.neural_network import MLPClassifier
from plotter import plot_histogram, plot_3D_feature_space
from song_features import readGenreClassData, getPointsAndClasses
from KNN import KNNClassifier, error_rate
from KNN_scikit import KNNSciKitClassifier
from itertools import combinations
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from sklearn.model_selection import StratifiedKFold


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

# #Extract training and test set
# X_train, y_train, ids_train  = getPointsAndClasses(songs_dict,features, genres, "Train")

# #Scale
# scaler = MinMaxScaler()
# scaler.fit(X_train)

# # Find best model
# n_splits = 3
# er_best = 1
# cm_best = None
# ijk = None

# for i in range(5,61,5):
#     for j in range(0,i,5):
#         for k in range(0,j,5):
#             print(i,j,k)
#             X_train, y_train, ids_train  = getPointsAndClasses(songs_dict,features, genres, "Train")
#             X_train = scaler.transform(X_train)
#             skf = StratifiedKFold(n_splits=n_splits)
#             skf.get_n_splits(X_train, y_train)
#             errors = np.zeros(n_splits)
#             ei = 0
#             for train_index, test_index in skf.split(X_train, y_train):
#                 if j == 0:
#                     clf = MLPClassifier(solver='lbfgs', alpha=1, activation="logistic", max_iter=100000,verbose=False,
#                                 hidden_layer_sizes=(i,), random_state=1)
#                 elif k == 0:
#                     clf = MLPClassifier(solver='lbfgs', alpha=1, activation="logistic", max_iter=100000,verbose=False,
#                                 hidden_layer_sizes=(i,j), random_state=1)
#                 else:
#                     clf = MLPClassifier(solver='lbfgs', alpha=1, activation="logistic", max_iter=100000,verbose=False,
#                                 hidden_layer_sizes=(i,j,k), random_state=1)
#                 clf.fit(X_train[train_index,:], y_train[train_index])
#                 y_pred = clf.predict(X_train[test_index,:].copy())
#                 cm = confusion_matrix(y_train[test_index].copy(), y_pred)
#                 errors[ei] = error_rate(cm)
#                 ei = ei + 1

#             avg_error = np.average(errors)
#             if avg_error < er_best:
#                 er_best = avg_error
#                 cm_best = cm
#                 ijk = [i,j,k]
#             if j==0:
#                 break

# # Print final result
# print("Chosen ijk: ", ijk)
# print("Error rate: ", er_best)

# Evaluate final NN
# Extract training
X_train, y_train, ids_train  = getPointsAndClasses(songs_dict,features, genres, "Train")

#Scale
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)

#Create classifier
clf = MLPClassifier(solver='lbfgs', alpha=1,activation="logistic", max_iter=100000,verbose=True,
                                hidden_layer_sizes=(60,50,40))
clf.fit(X_train, y_train)

X_test, y_test, ids_test  = getPointsAndClasses(songs_dict,features, genres, "Test")
X_test = scaler.transform(X_test)

y_pred = clf.predict(X_test)
cm = confusion_matrix(y_test, y_pred)

print("Error rate: ", error_rate(cm))

disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=genres)
disp.plot(cmap=plt.cm.Blues,xticks_rotation=45)
plt.show()