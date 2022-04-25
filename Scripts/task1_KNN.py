from plotter import plot_histogram, plot_3D_feature_space
from song_features import readGenreClassData, getPointsAndClasses
from KNN import KNNClassifier
from KNN_scikit import KNNSciKitClassifier
from itertools import combinations
from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# Import song feature
songs_dict = readGenreClassData("Data/GenreClassData_30s.txt")

# Choose features and genres:
features    = ["spectral_rolloff_mean","spectral_centroid_mean","mfcc_1_mean","tempo"]
genres      = ["pop","metal", "disco", "blues", "reggae", "classical", "rock", "hiphop", "country", "jazz"]

#Extract training and test set
X_train, y_train, ids_train  = getPointsAndClasses(songs_dict,features, genres, "Train")
X_test, y_test, ids_test  = getPointsAndClasses(songs_dict,features, genres, "Test")

# Create KNN object
knn_scikit = KNNSciKitClassifier(X_train, y_train, ids_train, features, 5,"min_max")
knn = KNNClassifier(X_train, y_train, ids_train, features, 5 ,"min_max")


# Get confusion matrices
# Our own implementation
confusion_matrix_our, confusion_matrix_list_our, error_rate = knn.evaluate(X_test.copy(),y_test.copy(),ids_test.copy()) 
print("Our KNN implementation:")  
print(error_rate)

#Scikit implementation
print("\n Sci-kit KNN implementation:")
confusion_matrix_scikit, confusion_matrix_list_scikit, error_rate = knn_scikit.evaluate(X_test.copy(),y_test.copy(),ids_test.copy())
print(error_rate)

# Plot confusion matrice
disp = ConfusionMatrixDisplay(confusion_matrix=confusion_matrix_our, display_labels=knn.classes)
disp.plot(cmap=plt.cm.Blues,xticks_rotation=45)
plt.show()