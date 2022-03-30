import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colors
from matplotlib.ticker import PercentFormatter

from song_features import readGenreClassData

def plot_histogram(song_features, feature):
    '''
    song_features: dictionary of SongFeatures
    feature: string of feature to plot
    '''
    # Define parameters:
    n_bins = 20
    # Extract feature: 
    feature_values = np.zeros(len(song_features))
    index = 0
    for song in song_features.values():
        feature_values[index] = song.__dict__[feature]
        index = index + 1
    # Plot:
    fig, ax = plt.subplots(sharey=True, tight_layout=True)
    hist = ax.hist(feature_values, bins = n_bins)
    plt.show()
    


# Import song feature
songs_dict = readGenreClassData("Data\GenreClassData_30s.txt")
plot_histogram(songs_dict, "zero_cross_rate_mean")