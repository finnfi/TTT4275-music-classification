
from data_extraction import GenreSet, ReducedSet
from plot_histogram import plot_histogram
from song_features import readGenreClassData
from KNN import KNNClassifier
from KNN_scikit import KNNSciKitClassifier
from itertools import combinations
from sklearn.metrics import ConfusionMatrixDisplay

import matplotlib.pyplot as plt

# Import song feature
songs_dict = readGenreClassData("Data/GenreClassData_30s.txt")

#Extract training and test set
training_set = GenreSet(songs_dict, "Train")
reduced_training_set = ReducedSet(training_set.pop, training_set.metal, training_set.disco, training_set.classical)

test_set = GenreSet(songs_dict,"Test")
reduced_test_set = ReducedSet(test_set.pop, test_set.metal, test_set.disco, test_set.classical)

# Plotting of features
genres_to_plot = ["pop","disco", "metal", "classical"]
axs1 = plot_histogram(training_set,genres_to_plot,"spectral_rolloff_mean",mode = "overlayed")
axs2 = plot_histogram(training_set,genres_to_plot,"mfcc_1_mean",mode = "overlayed")
axs3 = plot_histogram(training_set,genres_to_plot,"spectral_centroid_mean",mode = "overlayed")
axs4 = plot_histogram(training_set,genres_to_plot,"tempo",mode = "overlayed")
plt.show()

# Create KNN object
# Features: "spectral_centroid_mean","mfcc_1_mean","spectral_rolloff_mean","tempo"

# knn = KNNClassifier(training_set, ["spectral_centroid_mean","mfcc_1_mean","spectral_rolloff_mean","tempo"], 5 ,"min_max")
# knn_scikit = KNNSciKitClassifier(training_set, ["spectral_rolloff_mean","spectral_centroid_mean","tempo","mfcc_1_mean"], 5,"min_max")

# Do PCA on knn
# knn.doPCA(3)

# Plot 3D space
# ax = knn.plot_3D_feature_space()
# plt.show()


# Get confusion matrices
# print("Our KNN implementation:")
# confusion_matrix_our, confusion_matrix_list_our, error_rate = knn.evaluate(test_set)   


# print("\n Sci-kit KNN implementation:")
# confusion_matrix_scikit, confusion_matrix_list_scikit = knn_scikit.evaluate(test_set)

# Plot confusion matrices
# genres = ["pop","metal", "disco", "blues", "reggae", "classical", "rock", "hiphop", "country", "jazz"]
# disp = ConfusionMatrixDisplay(confusion_matrix=confusion_matrix_our, display_labels=genres)
# disp.plot(cmap=plt.cm.Blues,xticks_rotation=45)
# plt.show()

# disp = ConfusionMatrixDisplay(confusion_matrix=confusion_matrix_scikit, display_labels=genres)
# disp.plot(cmap=plt.cm.Blues,xticks_rotation=45)
# plt.show()