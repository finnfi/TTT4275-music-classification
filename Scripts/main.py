from data_extraction import TrainingSet, TestSet
from plot_histogram import plot_histogram
from song_features import readGenreClassData
from KNN import KNNClassifier

import matplotlib.pyplot as plt

# Import song feature
songs_dict = readGenreClassData("Data/GenreClassData_30s.txt")


#Extract training and test set
training_set = TrainingSet(songs_dict)
test_set = TestSet(songs_dict)

#Plotting of features
axs1 = plot_histogram(training_set,"spectral_rolloff_mean")
axs2 = plot_histogram(training_set,"mfcc_1_mean")
axs3 = plot_histogram(training_set,"spectral_centroid_mean")
axs4 = plot_histogram(training_set,"tempo")
plt.show()

#Create KNN object

nn_5 = KNNClassifier(training_set, ["spectral_rolloff_mean","mfcc_1_mean","spectral_centroid_mean","tempo"], 5, True)

confusion_matrix, confusion_matrix_list = nn_5.evaluate(test_set)