import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colors
from matplotlib.ticker import PercentFormatter

from song_features import readGenreClassData
import math

def plot_histogram(data_set, genres, feature):
    '''
    data_set: training or test set
    genres: array of genres
    feature: string of feature to plot
    '''
    # Define parameters:
    n_bins = 20
    # Extract feature: 
    feature_values = dict() #from class to [feature_value_song1, feature_value_song2, ...]
    min_xlim = 10000000
    max_xlim = -10000000
    for cls, songs in data_set.__dict__.items():
        feature_values[cls] = []
        for song in songs:
            value = song.__dict__[feature]
            if value < min_xlim:
                min_xlim = value
            elif value > max_xlim:
                max_xlim = value
            feature_values[cls].append(value)
    # Plot:
    n_genres = len(genres)

    fig, axs = plt.subplots(n_genres,1,sharey=True, tight_layout=True)
    plt.setp(axs, xlim=(min_xlim,max_xlim), ylim=(0,15))
    for i in range(n_genres):
        axs[i].hist(feature_values[genres[i]], bins = n_bins)
        axs[i].set_title(genres[i])
    
    fig.suptitle(feature)

    return axs
    
