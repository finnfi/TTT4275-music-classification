from data_extraction import TrainingSet, TestSet
from plot_histogram import plot_histogram
from song_features import readGenreClassData
from KNN import KNNClassifier
from KNN_scikit import KNNSciKitClassifier

import matplotlib.pyplot as plt

# Import song feature
songs_dict = readGenreClassData("Data/GenreClassData_30s.txt")


#Extract training and test set
training_set = TrainingSet(songs_dict)
test_set = TestSet(songs_dict)

#Plotting of features
# axs1 = plot_histogram(training_set,"spectral_rolloff_mean")
# axs2 = plot_histogram(training_set,"mfcc_1_mean")
# axs3 = plot_histogram(training_set,"spectral_centroid_mean")
# axs4 = plot_histogram(training_set,"tempo")
plt.show()

#Create KNN object

knn = KNNClassifier(training_set, ["spectral_rolloff_mean","mfcc_1_mean","spectral_centroid_mean","tempo"], 5, "z_score")
knn_scikit = KNNSciKitClassifier(training_set, ["spectral_rolloff_mean","mfcc_1_mean","spectral_centroid_mean","tempo"], 5, "z_score")

confusion_matrix, confusion_matrix_list = knn.evaluate(test_set)
confusion_matrix, confusion_matrix_list = knn_scikit.evaluate(test_set)