import numpy as np
from sklearn import neighbors
from song_features import genre_id_to_string
from sklearn.decomposition import PCA



class KNNSciKitClassifier:
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
        
        #Scikit classifier
        self.scikit_clf = neighbors.KNeighborsClassifier(self.k, weights="uniform", algorithm="auto")
        self.scikit_clf.fit(self.X, self.y)

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

    def classify(self, x):
        '''
        input: x of type np.array with dim = k (NOT normalised)
        output: genre string
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
        
        return self.scikit_clf.predict(x.reshape(1,-1))
        
    
    def classify_song(self, song):
        '''
        input: song of type SongFeatures
        output: genre string
        '''
        x = np.zeros(self.dim)
        i = 0
        for feature in self.features:
            x[i] = song.__dict__[feature]
            i += 1
        genre = self.classify(x)
        return genre
    
    def evaluate(self, X_test, y_test, ids_list_test):
        '''
        input: training_set of type TrainingSet 
        output: confusion_matrices (one with numbers, one with lists of ids)
        '''
        confusion_matrix_list = [[[]]*self.num_classes for i in range(self.num_classes)]
        confusion_matrix = np.zeros([self.num_classes,self.num_classes])

        for i in range(len(y_test)):
            genre_id = y_test[i]
            classified_id = self.classify(X_test[i,:])
            confusion_matrix_list[self.classes_id.index(genre_id)][self.classes_id.index(classified_id)].append(ids_list_test[i])
            confusion_matrix[self.classes_id.index(genre_id)][self.classes_id.index(classified_id)] +=  1
        er = error_rate(confusion_matrix)
        return confusion_matrix, confusion_matrix_list, er
    
    def doPCA(self, n_components):
        '''
        Transform points using PCA analysis
        Changes self.points
        '''
        self.isPCAused = True
        self.pca_n_components = n_components

        self.pca = PCA(n_components=n_components)
        self.X = self.pca.fit_transform(self.X)
        

def error_rate(confusion_matrix):
    '''
    input: confusion matrix as a 2-D np array
    output: error rate (# of falsely classified points)/( # total points)
    '''
    total = np.sum(confusion_matrix)
    correct = np.sum(np.diagonal(confusion_matrix))
    
    error_rate = (total-correct)/total

    return error_rate