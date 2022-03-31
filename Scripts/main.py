from data_extraction import TrainingSet, TestSet
from plot_histogram import plot_histogram
from song_features import readGenreClassData

import matplotlib.pyplot as plt

# Import song feature
songs_dict = readGenreClassData("Data\GenreClassData_30s.txt")

#Extract training and test set
training_set = TrainingSet(songs_dict)
test_set = TestSet(songs_dict)


axs1 = plot_histogram(training_set,"spectral_rolloff_mean")
axs2 =plot_histogram(training_set,"mfcc_1_mean")
axs3 =plot_histogram(training_set,"spectral_centroid_mean")
axs4 =plot_histogram(training_set,"tempo")

plt.show()


