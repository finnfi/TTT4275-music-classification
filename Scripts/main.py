from data_extraction import TrainingSet, TestSet, ReducedSet
from plot_histogram import plot_histogram
from song_features import readGenreClassData
from KNN import KNNClassifier
from KNN_scikit import KNNSciKitClassifier

import matplotlib.pyplot as plt

# Import song feature
songs_dict = readGenreClassData("Data/GenreClassData_30s.txt")


#Extract training and test set
training_set = TrainingSet(songs_dict)
reduced_training_set = ReducedSet(training_set.pop, training_set.metal, training_set.disco, training_set.classical)
test_set = TestSet(songs_dict)
reduced_test_set = ReducedSet(test_set.pop, test_set.metal, test_set.disco, test_set.classical)

# #Plotting of features
# genres_to_plot = ["pop","disco", "metal", "classical"]
# axs1 = plot_histogram(training_set,genres_to_plot,"spectral_rolloff_mean")
# axs2 = plot_histogram(training_set,genres_to_plot,"mfcc_1_mean")
# axs3 = plot_histogram(training_set,genres_to_plot,"spectral_centroid_mean")
# axs4 = plot_histogram(training_set,genres_to_plot,"tempo")
# plt.show()

#Create KNN object
# Features: "spectral_centroid_mean","mfcc_1_mean","spectral_rolloff_mean","tempo"
knn = KNNClassifier(reduced_training_set, ["tempo","spectral_rolloff_mean","spectral_centroid_mean"], 5 ,"min_max")
knn_scikit = KNNSciKitClassifier(reduced_training_set, ["tempo","spectral_rolloff_mean","spectral_centroid_mean"], 5,"min_max")

print("Our KNN implementation:")
confusion_matrix, confusion_matrix_list = knn.evaluate(reduced_test_set)
print("\n Sci-kit KNN implementation:")
confusion_matrix, confusion_matrix_list = knn_scikit.evaluate(reduced_test_set)