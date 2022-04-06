import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn import neighbors, datasets
from re import M
from collections import Counter
from song_features import genre_id_to_string, genre_string_to_id


class KNNSciKitClassifier:
    def __init__(self,training_set, features, k, input_normalisation_type = ""):
        '''
        training_set: variable of TrainingSet
        features: array of feature strings
        k: int 
        should_normalise: "" -> no normalisation, "z_score", "min_max"
        '''
        self.training_set = training_set
        
        self.classes = list(training_set.__dict__)
        self.num_classes = len(self.classes)
        self.k = k
        self.features = features
        self.dim = len(features)

        # Calculate number of points in traning set data
        self.num_points = 0
        for cls in training_set.__dict__.values():
            self.num_points += len(cls)

        # Initialise matrix of point and index-to-song array
        self.points = np.zeros([self.num_points,self.dim])
        self.index_to_song = [None]*self.num_points
        
        #Extract feature values
        i = 0
        for cls in training_set.__dict__.values():
            for song in cls:
                self.index_to_song[i] = song
                j = 0
                for feature in features:
                    self.points[i,j] = song.__dict__[feature]
                    j += 1
                i += 1

        # Generate target list (actual classes) for training set
        self.target_classes = [None]*self.num_points
        for i in range(self.num_points):
            song = self.index_to_song[i]
            self.target_classes[i]= song.Genre

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
        self.scikit_clf = neighbors.KNeighborsClassifier(self.k, weights="uniform")
        self.scikit_clf.fit(self.points, self.target_classes)

    def z_score_normalise(self):
        '''
        Normalises features using z-score normalisation
        '''
        for i in range(self.dim):
            mean = np.mean(self.points[:,i])
            var = np.var(self.points[:,i])
            sd = np.sqrt(var)
            self.points[:,i] = (self.points[:,i]-mean)/sd
            self.features_mean_sd.append((mean,sd))

    def min_max_normalise(self):
        '''
        Normalises features using min-max normalisation
        '''
        for i in range(self.dim):
            min = np.min(self.points[:,i])
            max = np.max(self.points[:,i])
            diff = max - min

            self.points[:,i] = (self.points[:,i]-min)/diff
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
    
    def evaluate(self, train_set):
        '''
        input: training_set of type TrainingSet 
        output: confusion_matrices (one with numbers, one with lists of ids)
        '''
        confusion_matrix_list = [[[]]*self.num_classes for i in range(self.num_classes)]
        confusion_matrix = np.zeros([self.num_classes,self.num_classes])
        for genre, song_list in train_set.__dict__.items():
            genre_id = self.classes.index(genre)
            for song in song_list:
                classified_id  = self.classes.index(self.classify_song(song))
                confusion_matrix_list[genre_id-1][classified_id-1].append(song.Track_ID)
                confusion_matrix[genre_id-1,classified_id-1] +=  1
        print("Confusion matrix: \n", confusion_matrix)
        print("Error rate: ", error_rate(confusion_matrix))
        return confusion_matrix, confusion_matrix_list
        

def error_rate(confusion_matrix):
    '''
    input: confusion matrix as a 2-D np array
    output: error rate (# of falsely classified points)/( # total points)
    '''
    total = np.sum(confusion_matrix)
    correct = np.sum(np.diagonal(confusion_matrix))
    
    error_rate = (total-correct)/total

    return error_rate