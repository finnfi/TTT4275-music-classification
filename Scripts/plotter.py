import matplotlib.pyplot as plt
import numpy as np

from song_features import genre_id_to_string

def plot_histogram(X , y, features, mode = "separat"):
    '''
    input: 
    X       : Points np.array() with size (N_POINTS,N_FEATURES)
    y       : Class IDs np.array() with size (N_POINTS)
    features: array of features
    mode    : "separat" or "overlayed"

    output:axs_list
    '''
    # Define parameters:
    n_bins = 14
    n_features = len(features)
    n_points = np.size(X,0)
    # Get max and min for each feature
    min_max_features = [] # Array with tuple for min and max for each feature
    for i in range(n_features):
        mm = [0,0]
        mm[0] = np.min(X[:,i])
        mm[1] = np.max(X[:,i])
        min_max_features.append(mm)
    
    # Extract genres from dataset: 
    genre_dict = dict() # Dict from genre ID to np.array of points
    for i in range(n_points):
        genre_id = y[i]
        genre_dict[genre_id] = np.append(genre_dict.get(genre_id, np.empty([0,n_features])), X[i,:].reshape(1,n_features), axis=0)

    n_genres = len(genre_dict)

    axs_list = [None]*n_features
    if mode == "separat":
        for j in range(n_features):
            fig, axs = plt.subplots(n_genres,1,sharey=True, tight_layout=True)
            plt.setp(axs, xlim=(min_max_features[j][0],min_max_features[j][1]), ylim=(0,20))
            i = 0
            for genre_id in genre_dict:
                axs[i].hist(genre_dict[genre_id][:,j], bins = n_bins)
                axs[i].set_title(genre_id_to_string(genre_id))
                i = i + 1
            fig.suptitle(features[j])
            axs_list[j] = axs
        


    elif mode == "overlayed":
        colors = [  (1,0,0,0.5),(0,1,0,0.5),(0,0,1,0.5),(1,1,0,0.5),(1,0,1,0.5),(0,1,1,0.5),
                    (0.95,0.5,0.2,0.5), (0.25,0.75,0.3,0.5),(0.5,0,0,0.5),(0.25,0.4,0.8,0.5)]
        for j in range(n_features):
            fig, axs = plt.subplots(1,1,sharey=True, tight_layout=True)
            plt.setp(axs, xlim=(min_max_features[j][0],min_max_features[j][1]), ylim=(0,20))
            i = 0
            for genre_id in genre_dict:
                axs.hist(genre_dict[genre_id][:,j], bins = n_bins, label = genre_id_to_string(genre_id), fc = colors[i])
                i = i + 1
            fig.suptitle(features[j])
            axs_list[j] = axs

            #Add legends
            axs.legend(loc='best', frameon=False)
            fig.suptitle(features[j])

    return axs_list
    
def plot_3D_feature_space(X , y, features):
        '''
        input
        training_set: a GenreSet
        genres: list of genres to plot
        features: a list of size 3 with features to use

        output
        returns an ac object if dim==3, else nothing
        '''
        if (np.size(X,1) != 3):
            return

        # Define parameters:
        n_features = len(features)
        n_points = np.size(X,0)
        # Get max and min for each feature
        min_max_features = [] # Array with tuple for min and max for each feature
        for i in range(n_features):
            mm = [0,0]
            mm[0] = np.min(X[:,i])
            mm[1] = np.max(X[:,i])
            min_max_features.append(mm)
        
        # Extract genres from dataset: 
        genre_dict = dict() # Dict from genre ID to np.array of points
        for i in range(n_points):
            genre_id = y[i]
            genre_dict[genre_id] = np.append(genre_dict.get(genre_id, np.empty([0,n_features])), X[i,:].reshape(1,n_features), axis=0)
        
        #Init figure
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')

        for genre in genre_dict:
            ax.scatter(genre_dict[genre][:,0],genre_dict[genre][:,1],genre_dict[genre][:,2],label=genre_id_to_string(genre))

        #Set labels
        ax.set_xlabel(features[0])
        ax.set_ylabel(features[1])
        ax.set_zlabel(features[2])

        #Add legends
        ax.legend(loc='best', frameon=False)

        return ax