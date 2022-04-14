
from data_extraction import getPointsAndClasses
from plotter import plot_histogram, plot_3D_feature_space
from song_features import readGenreClassData
from KNN import KNNClassifier
from KNN_scikit import KNNSciKitClassifier
from itertools import combinations
from sklearn.metrics import ConfusionMatrixDisplay

import matplotlib.pyplot as plt

# Import song feature
songs_dict = readGenreClassData("Data/GenreClassData_30s.txt")

# Choose features and genres:
# features    = [ "zero_cross_rate_mean","zero_cross_rate_std","rmse_mean","rmse_var","spectral_centroid_mean","spectral_centroid_var","spectral_bandwidth_mean",
#                 "spectral_bandwidth_var","spectral_rolloff_mean","spectral_rolloff_var","spectral_contrast_mean","spectral_contrast_var","spectral_flatness_mean",
#                 "spectral_flatness_var","chroma_stft_1_mean","chroma_stft_2_mean","chroma_stft_3_mean","chroma_stft_4_mean","chroma_stft_5_mean","chroma_stft_6_mean",
#                 "chroma_stft_7_mean","chroma_stft_8_mean","chroma_stft_9_mean","chroma_stft_10_mean","chroma_stft_11_mean","chroma_stft_12_mean","chroma_stft_1_std",
#                 "chroma_stft_2_std","chroma_stft_3_std","chroma_stft_4_std","chroma_stft_5_std","chroma_stft_6_std","chroma_stft_7_std","chroma_stft_8_std","chroma_stft_9_std",
#                 "chroma_stft_10_std","chroma_stft_11_std","chroma_stft_12_std","tempo","mfcc_1_mean","mfcc_2_mean","mfcc_3_mean","mfcc_4_mean","mfcc_5_mean","mfcc_6_mean",
#                 "mfcc_7_mean","mfcc_8_mean","mfcc_9_mean","mfcc_10_mean","mfcc_11_mean","mfcc_12_mean","mfcc_1_std","mfcc_2_std","mfcc_3_std","mfcc_4_std","mfcc_5_std",
#                 "mfcc_6_std","mfcc_7_std","mfcc_8_std","mfcc_9_std","mfcc_10_std","mfcc_11_std","mfcc_12_std"]
# features    = ["spectral_rolloff_mean","spectral_centroid_mean","mfcc_1_mean","tempo"]
features    = ["spectral_rolloff_mean","spectral_centroid_mean","mfcc_1_mean","tempo"]
genres      = ["pop","metal", "disco", "blues", "reggae", "classical", "rock", "hiphop", "country", "jazz"]
# genres      = ["pop", "disco", "metal", "classical"] # Genres to plot in excercise 2

#Extract training and test set
X_train, y_train, ids_train  = getPointsAndClasses(songs_dict,features, genres, "Train")
X_test, y_test, ids_test  = getPointsAndClasses(songs_dict,features, genres, "Test")

# Plotting histogram of features
# axs_list = plot_histogram(X_train, y_train, features, mode = "overlayed")
# plt.show()

# Create KNN object
knn = KNNClassifier(X_train, y_train, ids_train, features, 5 ,"min_max")
# knn_scikit = KNNSciKitClassifier(X_train, y_train, ids_train, features, 5,"min_max")


# Do PCA on knn
# knn.doPCA(3)

# Plot 3D space
# ax = plot_3D_feature_space(X_train,y_train,features)
# plt.show()


# Get confusion matrices
print("Our KNN implementation:")
confusion_matrix_our, confusion_matrix_list_our, error_rate = knn.evaluate(X_test.copy(),y_test.copy(),ids_test.copy())   
# print("\n Sci-kit KNN implementation:")
# confusion_matrix_scikit, confusion_matrix_list_scikit, error_rate = knn_scikit.evaluate(X_test.copy(),y_test.copy(),ids_test.copy())

# Plot confusion matrices
disp = ConfusionMatrixDisplay(confusion_matrix=confusion_matrix_our, display_labels=knn.classes)
disp.plot(cmap=plt.cm.Blues,xticks_rotation=45)
plt.show()

# disp = ConfusionMatrixDisplay(confusion_matrix=confusion_matrix_scikit, display_labels=genres)
# disp.plot(cmap=plt.cm.Blues,xticks_rotation=45)
# plt.show()