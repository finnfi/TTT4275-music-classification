from soupsieve import select
from song_features import genre_id_to_string, genre_string_to_id
from song_features import readGenreClassData, getPointsAndClasses
from KNN import error_rate
from itertools import cycle
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import SelectKBest
import numpy as np
import copy


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
features    = ["spectral_rolloff_mean","spectral_centroid_mean","mfcc_1_mean", "tempo"]
# features    = ["spectral_rolloff_mean","spectral_centroid_mean","mfcc_1_mean","tempo"]
genres          = ["pop","metal", "disco", "blues", "reggae", "classical", "rock", "hiphop", "country", "jazz"]

best_gmms = dict() # Dict from genre to best gmm for that genre

X_train, y_train, ids_train  = getPointsAndClasses(songs_dict,features, genres, "Train")
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)

k_best= SelectKBest(k=4)
k_best.fit(X_train, y_train)

for genre in genres: 
    X_train, y_train, ids_train  = getPointsAndClasses(songs_dict,features, [genre], "Train")
    X_train = scaler.transform(X_train)
    X_train = k_best.transform(X_train)
    n_samples = len(ids_train)
    lowest_bic = np.infty
    bic = [] # ddescription: https://onlinelibrary.wiley.com/doi/pdf/10.1002/9781118856406.app5
    n_components_range = range(1,10)
    cv_types = ["spherical", "tied", "diag", "full"] # description: https://scikit-learn.org/stable/modules/generated/sklearn.mixture.GaussianMixture.html
    for cv_type in cv_types:
        for n_components in n_components_range:
            # Here we fit a GM
            gmm = GaussianMixture(n_components=n_components, covariance_type=cv_type)
            gmm.fit(X_train)
            bic.append(gmm.bic(X_train))
            if bic[-1] < lowest_bic:
                lowest_bic = bic[-1]
                best_gmm = gmm
    bic = np.array(bic)
    color_iter = cycle(["navy", "turquoise", "cornflowerblue", "darkorange"])
    clf = best_gmm
    bars = []

    best_gmms[genre] = copy.deepcopy(best_gmm)

    # Plot the BIC scores
    # plt.figure(figsize=(8, 6))
    # spl = plt.subplot(2, 1, 1)
    # for i, (cv_type, color) in enumerate(zip(cv_types, color_iter)):
    #     xpos = np.array(n_components_range) + 0.2 * (i - 2)
    #     bars.append(
    #         plt.bar(
    #             xpos,
    #             bic[i * len(n_components_range) : (i + 1) * len(n_components_range)],
    #             width=0.2,
    #             color=color,
    #         )
    #     )
    # plt.xticks(n_components_range)
    # plt.ylim([bic.min() * 1.01 - 0.01 * bic.max(), bic.max()])
    # plt.title("BIC score per model")
    # xpos = (
    #     np.mod(bic.argmin(), len(n_components_range))
    #     + 0.65
    #     + 0.2 * np.floor(bic.argmin() / len(n_components_range))
    # )
    # plt.text(xpos, bic.min() * 0.97 + 0.03 * bic.max(), "*", fontsize=14)
    # spl.set_xlabel("Number of components")
    # spl.legend([b[0] for b in bars], cv_types)
    #  plt.show()

# Test data:
X_test, y_test, ids_test  = getPointsAndClasses(songs_dict,features, genres, "Test")
X_test = scaler.transform(X_test)
X_test = k_best.transform(X_test)
# Compute log likelihood for every function
log_likelihood = np.zeros((len(y_test),len(genres)))
i = 0
for genre in genres: 
    model = best_gmms[genre]
    log_likelihood[:,i]  = model.score_samples(X_test)
    i += 1

maxVal_ind_rowise = np.argmax(log_likelihood, axis=1)

cm = np.zeros((len(genres), len(genres)))
for i in range(len(y_test)):
    cm[genres.index(genre_id_to_string(y_test[i])), maxVal_ind_rowise[i]] += 1

print(error_rate(cm))

disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=genres)
disp.plot(cmap=plt.cm.Blues,xticks_rotation=45)
plt.show()

        

