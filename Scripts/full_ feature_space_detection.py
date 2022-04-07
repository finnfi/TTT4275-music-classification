# This script goes through all possible combinations of features and finds the one with lowest error rate
from data_extraction import GenreSet
from song_features import readGenreClassData
from KNN import KNNClassifier
from itertools import combinations

# Import song feature
songs_dict = readGenreClassData("Data/GenreClassData_30s.txt")

#Extract training and test set
training_set = GenreSet(songs_dict, "Train")
test_set = GenreSet(songs_dict,"Test")

# Create feature combinations
feature_pool = ["zero_cross_rate_mean","zero_cross_rate_std","rmse_mean","rmse_var","spectral_centroid_mean","spectral_centroid_var","spectral_bandwidth_mean",
"spectral_bandwidth_var","spectral_rolloff_mean","spectral_rolloff_var","spectral_contrast_mean","spectral_contrast_var","spectral_flatness_mean",
"spectral_flatness_var","chroma_stft_1_mean","chroma_stft_2_mean","chroma_stft_3_mean","chroma_stft_4_mean","chroma_stft_5_mean","chroma_stft_6_mean",
"chroma_stft_7_mean","chroma_stft_8_mean","chroma_stft_9_mean","chroma_stft_10_mean","chroma_stft_11_mean","chroma_stft_12_mean","chroma_stft_1_std",
"chroma_stft_2_std","chroma_stft_3_std","chroma_stft_4_std","chroma_stft_5_std","chroma_stft_6_std","chroma_stft_7_std","chroma_stft_8_std","chroma_stft_9_std",
"chroma_stft_10_std","chroma_stft_11_std","chroma_stft_12_std","tempo","mfcc_1_mean","mfcc_2_mean","mfcc_3_mean","mfcc_4_mean","mfcc_5_mean","mfcc_6_mean",
"mfcc_7_mean","mfcc_8_mean","mfcc_9_mean","mfcc_10_mean","mfcc_11_mean","mfcc_12_mean","mfcc_1_std","mfcc_2_std","mfcc_3_std","mfcc_4_std","mfcc_5_std",
"mfcc_6_std","mfcc_7_std","mfcc_8_std","mfcc_9_std","mfcc_10_std","mfcc_11_std","mfcc_12_std"]

error_rate = 1
confusion_matrix = []
chosen_features = []

n_features = len(feature_pool)
for i in range(1,n_features+1):
    feature_combinations = list(combinations(feature_pool,i))

    for comb in feature_combinations:
        features = list(comb)
        knn = KNNClassifier(training_set, features, 5 ,"min_max")
        cm, cm_list, er = knn.evaluate(test_set)
        if er < error_rate:
            error_rate = er
            confusion_matrix = cm
            chosen_features = features
            print(chosen_features,"\n", confusion_matrix,"\n",error_rate,file=open('output2.txt', 'w'))


print(chosen_features)
print(confusion_matrix)
print(error_rate)