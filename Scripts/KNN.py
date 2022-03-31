
import numpy as np
from song_features import genre_id_to_string, genre_string_to_id

class KNNClassifier:
    def __init__(self,training_set, features, k, should_normalise):
        '''
        training_set: variable of TrainingSet
        features: array of feature strings
        k: int 
        '''
        self.k = k
        self.features = features
        self.dim = len(features)
        self.features_mean_and_sd = [] # array of tuple (mean, sd)
        self.training_set = training_set
        self.num_classes = len(training_set.__dict__)
        self.num_points = 0
        for cls in training_set.__dict__.values():
            self.num_points += len(cls)
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
        
        self.is_normalised = False
        if should_normalise:
            self.normalise()
            

    def normalise(self):
        '''
        Normalises features using z-score normalisation
        '''
        self.is_normalised = True
        for i in range(self.dim):
            mean = np.mean(self.points[:,i])
            var = np.var(self.points[:,i])
            sd = np.sqrt(var)
            self.points[:,i] = (self.points[:,i]-mean)/sd
            self.features_mean_and_sd.append((mean,sd))
        

    def classify(self, x):
        '''
        input: x of type np.array with dim = k (NOT normalised)
        output: genre string
        '''
        #normalise input if normalised
        if self.is_normalised:
            for i in range(self.dim):
                x[i] = (x[i]-self.features_mean_and_sd[i][0])/self.features_mean_and_sd[i][1]
        # Calculate distances
        difference = self.points-x
        distances = np.sum(difference*difference,axis=1)
        # k smallest distances: 
        ind = np.argpartition(distances, self.k)[:self.k] #function gives array of indexes

        # Find genres 
        genres = []
        for i in ind:
            song = self.index_to_song[i]
            genres.append(song.Genre)
        k_nearest_distances = distances[ind]
        # Return the most common or the closest one if we got 5 different genres
        if len(set(genres)) == len(genres):
            nearest_index = k_nearest_distances.argmin()
            return genres[nearest_index]
        else: 
            #HAVE TO DO MORE HERE f.ex. when: ['pop','pop','reggea','reggea','jazz']?????
            return max(set(genres), key = genres.count)

    
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
        Returns a confusion matrix
        
        input: training_set of type TrainingSet 
        output confusion_matrix containing list of ids
        '''
        confusion_matrix_list = [[[]]*self.num_classes for i in range(self.num_classes)]
        confusion_matrix = np.zeros([self.num_classes,self.num_classes])
        for genre, song_list in train_set.__dict__.items():
            genre_id = genre_string_to_id(genre)
            for song in song_list:
                classified_id = genre_string_to_id(self.classify_song(song))
                confusion_matrix_list[genre_id-1][classified_id-1].append(song.Track_ID)
                confusion_matrix[genre_id-1,classified_id-1] +=  1
        print(confusion_matrix)
        return confusion_matrix, confusion_matrix_list
        





        