
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

class KNNClassifier:
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
        
        #Do PCA transform if PCA is anabled
        if self.isPCAused:
            x = self.pca.transform(x.reshape(1, -1))

        # Calculate distances
        difference = self.points-x
        distances = np.sum(difference*difference,axis=1)
        # k smallest distances: 
        index_k_nearest_points = np.argpartition(distances, self.k)[:self.k] #function gives array of indexes
        distance_k_nearest_points = distances[index_k_nearest_points]

        # Find genres 
        genres_k_nearest_points = []
        for i in index_k_nearest_points:
            song = self.index_to_song[i]
            genres_k_nearest_points.append(song.Genre)
        
        # Combine data [[index, distance,genre], [index,distance,genre], ...]
        k_nearest_points = [[index_k_nearest_points[i],distance_k_nearest_points[i],genres_k_nearest_points[i]] for i in range(self.k)]
        
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

        #Return genre with smallest distance
        return points_to_consider[0][2]
        
    
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
                classified_id = self.classes.index(self.classify_song(song))
                confusion_matrix_list[genre_id][classified_id].append(song.Track_ID)
                confusion_matrix[genre_id,classified_id] +=  1
        er = error_rate(confusion_matrix)
        print("Confusion matrix: \n", confusion_matrix)
        print("Error rate: ", er)
        return confusion_matrix, confusion_matrix_list, er
    
    def plot_3D_feature_space(self):
        '''
        input
        training_set: a GenreSet
        genres: list of genres to plot
        features: a list of size 3 with features to use

        output
        returns an ac object if dim==3, else nothing
        '''
        if (self.isPCAused and self.pca_n_components != 3) or (not self.isPCAused and self.dim != 3):
            return
        #Init figure
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')

        xs = []
        ys = []
        zs = []

        for i in range(self.num_points):
            #Add point to figure 
            xs.append(self.points[i][0])
            ys.append(self.points[i][1])
            zs.append(self.points[i][2])

            if i+1 >= self.num_points or self.index_to_song[i].Genre != self.index_to_song[i+1].Genre:
                ax.scatter(xs,ys,zs,label=self.index_to_song[i].Genre)
                xs = []
                ys = []
                zs = []

        #Set labels
        if not self.isPCAused:
            ax.set_xlabel(self.features[0])
            ax.set_ylabel(self.features[1])
            ax.set_zlabel(self.features[2])
        elif self.isPCAused:
            ax.set_xlabel("pca1")
            ax.set_ylabel("pca2")
            ax.set_zlabel("pca3")

        #Add legends
        ax.legend(loc='best', frameon=False)

        return ax
    
    def doPCA(self, n_components):
        '''
        Transform points using PCA analysis
        Changes self.points
        '''
        self.isPCAused = True
        self.pca_n_components = n_components

        self.pca = PCA(n_components=n_components)
        self.points = self.pca.fit_transform(self.points)


        

def error_rate(confusion_matrix):
    '''
    input: confusion matrix as a 2-D np array
    output: error rate (# of falsely classified points)/( # total points)
    '''
    total = np.sum(confusion_matrix)
    correct = np.sum(np.diagonal(confusion_matrix))
    
    error_rate = (total-correct)/total

    return error_rate
    