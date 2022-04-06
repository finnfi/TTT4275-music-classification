from data_extraction import GenreSet, ReducedSet
from plot_histogram import plot_histogram
from song_features import readGenreClassData
from KNN import KNNClassifier
from KNN_scikit import KNNSciKitClassifier
from itertools import combinations

import matplotlib.pyplot as plt

# Import song feature
songs_dict = readGenreClassData("Data/GenreClassData_30s.txt")


#Extract training and test set
training_set = GenreSet(songs_dict, "Train")
reduced_training_set = ReducedSet(training_set.pop, training_set.metal, training_set.disco, training_set.classical)

test_set = GenreSet(songs_dict,"Test")
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
# knn = KNNClassifier(training_set, ["tempo","spectral_rolloff_mean","spectral_centroid_mean"], 5 ,"min_max")
# knn_scikit = KNNSciKitClassifier(training_set, ["tempo","spectral_rolloff_mean","spectral_centroid_mean"], 5,"min_max")

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
confusion_matrix = []
chosen_features = []
for comb in feature_pool_1_combinations:
    for feature in feature_pool_2:
        features = list(comb) + [feature]
        knn = KNNClassifier(training_set, features, 5 ,"min_max")
        cm, cm_list, er = knn.evaluate(test_set)
        if er < error_rate:
            error_rate = er
            confusion_matrix = cm
            chosen_features = features


print(chosen_features)
print(confusion_matrix)
print(error_rate)


# print("Our KNN implementation:")
# confusion_matrix, confusion_matrix_list, error_rate = knn.evaluate(test_set)
# print("\n Sci-kit KNN implementation:")
# confusion_matrix, confusion_matrix_list = knn_scikit.evaluate(reduced_test_set)