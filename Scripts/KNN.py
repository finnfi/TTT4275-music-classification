import numpy as np
from sklearn.decomposition import PCA
from song_features import genre_id_to_string

class KNNClassifier:
    def __init__(self, X, y, ids_list, features, k, input_normalisation_type = ""):
        '''
        X               : point nd.array of dimension (N_POINTS, N_FEATURES)
        y               : nd.array of class labels
        ids_list        : array of correspongin track ids 
        features        : array of feature strings
        k               : int 
        should_normalise: "" -> no normalisation, "z_score", "min_max"
        '''
        self.X = X.copy()
        self.y = y.copy()
        self.ids_list = ids_list.copy()
        self.classes_id = list(np.unique(y))
        self.classes = [genre_id_to_string(i) for i in self.classes_id]
        self.num_classes = len(self.classes)
        self.k = k
        self.features = features
        self.dim = len(features)
        self.num_points = np.size(X,0)
        
        #PCA variables:
        self.isPCAused = False
        self.pca_n_components = 0
        
        #Normalisation
        # Keep track of mean, sd, min and max for each feature to normalise
        self.features_mean_sd = [] # array of tuple (mean, sd)
        self.features_min_max = [] # array of tuple (min, max)
        
        self.input_normalisation_type = input_normalisation_type
        if input_normalisation_type == "z_score":
            self.z_score_normalise()
        elif input_normalisation_type == "min_max":
            self.min_max_normalise()
            
    def z_score_normalise(self):
        '''
        Normalises features using z-score normalisation
        '''
        for i in range(self.dim):
            mean = np.mean(self.X[:,i])
            var = np.var(self.X[:,i])
            sd = np.sqrt(var)
            self.X[:,i] = (self.X[:,i]-mean)/sd
            self.features_mean_sd.append((mean,sd))

    def min_max_normalise(self):
        '''
        Normalises features using min-max normalisation
        '''
        for i in range(self.dim):
            min = np.min(self.X[:,i])
            max = np.max(self.X[:,i])
            diff = max - min

            self.X[:,i] = (self.X[:,i]-min)/diff
            self.features_min_max.append((min,max))
        
    def doPCA(self, n_components):
        '''
        Transform points using PCA analysis
        Changes self.points
        '''
        self.isPCAused = True
        self.pca_n_components = n_components

        self.pca = PCA(n_components=n_components)
        self.X = self.pca.fit_transform(self.X)

    def classify(self, x):
        '''
        input: x of type np.array with dim = k (NOT normalised)
        output: genre ID
        '''
        #normalise input if normalisation is enabled
        if self.input_normalisation_type == "z_score":
            for i in range(self.dim):
                x[i] = (x[i]-self.features_mean_sd[i][0])/self.features_mean_sd[i][1]
        elif self.input_normalisation_type == "min_max":
            for i in range(self.dim):
                min = (self.features_min_max[i][0])
                max = self.features_min_max[i][1]
                diff = max-min
                x[i] = (x[i]-min)/diff
        
        #Do PCA transform if PCA is anabled
        if self.isPCAused:
            x = self.pca.transform(x.reshape(1, -1))

        # Calculate distances
        difference = self.X-x
        distances = np.sum(difference*difference,axis=1)
        # k smallest distances: 
        #function gives array of indexes
        index_k_nearest_points = np.argpartition(distances, self.k)[:self.k] 
        distance_k_nearest_points = distances[index_k_nearest_points]

        # Find genres 
        genres_k_nearest_points = []
        for i in index_k_nearest_points:
            genres_k_nearest_points.append(self.y[i])
        
        # Combine data [[index, distance,genre], [index,distance,genre], ...]
        k_nearest_points = [[index_k_nearest_points[i],distance_k_nearest_points[i],
                            genres_k_nearest_points[i]] for i in range(self.k)]
        
        #Count genres
        genres_count = dict()
        for point in k_nearest_points:
            genres_count[point[2]] = genres_count.get(point[2],0) + 1
        genres_count = list(genres_count.items())
        genres_count = sorted(genres_count, key=lambda tup: tup[1], reverse=True)

        #Find genre(s) with most entries, delete the others
        for i in range(1,len(genres_count)):
            if genres_count[i][1] < genres_count[i-1][1]:
                genres_count = genres_count[:i]
                break
        
        genres = [genre[0] for genre in genres_count]

        #Find all points belonging to these genres
        points_to_consider = []
        for point in k_nearest_points:
            if point[2] in genres:
                points_to_consider.append(point)

        #Sort points to consider
        points_to_consider = sorted(points_to_consider, key=lambda x:x[1])

        #Return genre ID with smallest distance
        return points_to_consider[0][2]
    
    def evaluate(self, X_test, y_test, ids_list_test):
        '''
        input: 
        X_test                  : point nd.array of dimension (N_POINTS, N_FEATURES)
        y_test                  : nd.array of class labels
        ids_list_test           : array of correspongin track ids 

        output:
        confusion_matrix        : 2D np array of ints with size (N_CLASSES, N_CLASSES)
        confusion_matrix_list   : 2D array containg lists of classified song ids
        er                      : Error rate          
        '''
        confusion_matrix_list = [[[] for j in range(self.num_classes)] for i in range(self.num_classes)]
        confusion_matrix = np.zeros([self.num_classes,self.num_classes])

        for i in range(len(y_test)):
            genre_id = y_test[i]
            classified_id = self.classify(X_test[i,:])
            correct_index = self.classes_id.index(genre_id)
            predicted_index = self.classes_id.index(classified_id)
            confusion_matrix_list[correct_index][predicted_index].append(ids_list_test[i])
            confusion_matrix[correct_index][predicted_index] +=  1
        er = error_rate(confusion_matrix)
        return confusion_matrix, confusion_matrix_list, er
    
    
# Error rate helper function
def error_rate(confusion_matrix):
    '''
    input: confusion matrix as a 2D array
    output: error rate (# of falsely classified points)/( # total points)
    '''
    total = np.sum(confusion_matrix)
    correct = np.sum(np.diagonal(confusion_matrix))
    
    error_rate = (total-correct)/total

    return error_rate
    