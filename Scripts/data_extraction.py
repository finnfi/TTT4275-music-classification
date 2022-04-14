import numpy as np

def getPointsAndClasses(songs_dict, features, classes, type):
    '''
    input:  songs_dict  - a dict from id to SongFeature
            features    - array of features to be used
            classes     - array of classes to be used
            type        - "Train" or "Test"
    output: X, y, id_list - X is a np.array of sice (N_POINTS, N_FEATURES), 
                            y is a np.array of class ids with size N_POINTS, 
                            id_list is id to corresponding song
    '''
    n_features  = len(features)

    X = np.empty([0,n_features])
    y = np.empty(0,dtype=np.int16)
    id_list = []
    for song in songs_dict.values():
        if song.Type == type and song.Genre in classes:
            x = np.zeros([1,n_features])
            for i in range(n_features):
                x[0,i] = song.__dict__[features[i]]
            X = np.append(X,x,axis = 0)
            y = np.append(y,song.GenreID) 
            id_list.append(song.Track_ID)
    
    return X, y, id_list


            



