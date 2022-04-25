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
from sklearn.model_selection import train_test_split

# Choose features and genres:
features    = [ "zero_cross_rate_mean","zero_cross_rate_std","rmse_mean","rmse_var","spectral_centroid_mean","spectral_centroid_var","spectral_bandwidth_mean",
                "spectral_bandwidth_var","spectral_rolloff_mean","spectral_rolloff_var","spectral_contrast_mean","spectral_contrast_var","spectral_flatness_mean",
                "spectral_flatness_var","chroma_stft_1_mean","chroma_stft_2_mean","chroma_stft_3_mean","chroma_stft_4_mean","chroma_stft_5_mean","chroma_stft_6_mean",
                "chroma_stft_7_mean","chroma_stft_8_mean","chroma_stft_9_mean","chroma_stft_10_mean","chroma_stft_11_mean","chroma_stft_12_mean","chroma_stft_1_std",
                "chroma_stft_2_std","chroma_stft_3_std","chroma_stft_4_std","chroma_stft_5_std","chroma_stft_6_std","chroma_stft_7_std","chroma_stft_8_std","chroma_stft_9_std",
                "chroma_stft_10_std","chroma_stft_11_std","chroma_stft_12_std","tempo","mfcc_1_mean","mfcc_2_mean","mfcc_3_mean","mfcc_4_mean","mfcc_5_mean","mfcc_6_mean",
                "mfcc_7_mean","mfcc_8_mean","mfcc_9_mean","mfcc_10_mean","mfcc_11_mean","mfcc_12_mean","mfcc_1_std","mfcc_2_std","mfcc_3_std","mfcc_4_std","mfcc_5_std",
                "mfcc_6_std","mfcc_7_std","mfcc_8_std","mfcc_9_std","mfcc_10_std","mfcc_11_std","mfcc_12_std"]
genres          = ["pop","metal", "disco", "blues", "reggae", "classical", "rock", "hiphop", "country", "jazz"]

best_error_rate = 1
best_idx = 0
best_models = dict()
list_of_error_rates = []

# for k in range(1,len(features)+1):
for k in range(10,11):
    # Import song feature
    songs_dict = readGenreClassData("Data/GenreClassData_30s.txt")

    # Init dict for best models for each genre
    best_gmms = dict() # Dict from genre to best gmm for that genre

    # Create scalers and kbest transform
    X_train, y_train, ids_train  = getPointsAndClasses(songs_dict,features, genres, "Train")
    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X_train)
    k_best= SelectKBest(k=k)
    k_best.fit(X_train, y_train)

    # Print names of feature chosen
    feature_map = k_best.get_feature_names_out()
    for f in feature_map:
        idx = int(f[1:])
        print(features[idx])

    # Find appropriate size of GM using BIC
    for genre in genres:
        # Get training set of only choosen feature and create a test-validation split
        X, y, ids  = getPointsAndClasses(songs_dict,features, [genre], "Train")
        # Scale and extract k best features
        X = scaler.transform(X)
        X = k_best.transform(X)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1, stratify=y)
        
        # Initialise lowest bic
        lowest_bic = np.infty
        bic = [] # description: https://onlinelibrary.wiley.com/doi/pdf/10.1002/9781118856406.app5

        # Initialise best gmm
        best_gmm = None

        # Choose how many components that shall be tested. 
        n_components_range = range(1,10)
        # Loop through components
        for n_components in n_components_range:
            # Here we fit a GM
            gmm = GaussianMixture(n_components=n_components, covariance_type="full",init_params='kmeans',random_state=1)
            gmm.fit(X_train)
            # Calculate BIC of model and append to array
            bic.append(copy.deepcopy(gmm.bic(X_train)))
            # If bic is lowest, set it as best gmm
            if bic[-1] < lowest_bic:
                lowest_bic = bic[-1]
                best_gmm = copy.deepcopy(gmm)
        
        best_gmms[genre] = copy.deepcopy(best_gmm)

    # Get training set of only choosen feature and create a test-validation split
    X, y, ids  = getPointsAndClasses(songs_dict,features, genres, "Train")
    # Scale and extract k best features
    X = scaler.transform(X)
    X = k_best.transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1, stratify=y)
    
    # Compute log likelihood for every function
    log_likelihood = np.zeros((len(y_test),len(genres)))
    for genre, model in best_gmms.items():
        log_likelihood[:,genres.index(genre)]  = model.score_samples(X_test)

    # Find index of best log likelihood
    maxVal_ind_rowise = np.argmax(log_likelihood, axis=1)

    # Create confusion matrix
    cm = np.zeros((len(genres), len(genres)))
    for i in range(len(y_test)):
        cm[genres.index(genre_id_to_string(y_test[i])), maxVal_ind_rowise[i]] += 1

    # Display confusion matrices
    # disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=genres)
    # disp.plot(cmap=plt.cm.Blues,xticks_rotation=45)
    # plt.show()

    # Compute error rate
    error = error_rate(cm)
    list_of_error_rates.append(error)
    print(error)
    if error < best_error_rate:
        best_error_rate = error_rate(cm)
        best_idx = k
        best_models = copy.deepcopy(best_gmms)

for genre, model in best_models.items():
    print(genre, " has a model with ", model.n_components, " components")

print(best_error_rate)
print(best_idx)

plt.title("Error rates for different # of selected features")
plt.plot(np.arange(1,len(list_of_error_rates)+1,1), np.array(list_of_error_rates), color="red")
plt.xlabel("# of selected features")
plt.ylabel("Error rate")
plt.show()


# disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=genres)
# disp.plot(cmap=plt.cm.Blues,xticks_rotation=45)
# plt.show()

