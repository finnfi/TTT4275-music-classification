import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colors
from matplotlib.ticker import PercentFormatter

from song_features import readGenreClassData


def plot_histogram(data_set, feature):
    '''
    data_set: training or test set
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
    fig, axs = plt.subplots(5,2,sharey=True, tight_layout=True)
    plt.setp(axs, xlim=(min_xlim,max_xlim), ylim=(0,15))
    axs[0][0].hist(feature_values["pop"], bins = n_bins)
    axs[0][0].set_title("pop")
    axs[1][0].hist(feature_values["metal"], bins = n_bins)
    axs[1][0].set_title("metal")
    axs[2][0].hist(feature_values["disco"], bins = n_bins)
    axs[2][0].set_title("disco")
    axs[3][0].hist(feature_values["blues"], bins = n_bins)
    axs[3][0].set_title("blues")
    axs[4][0].hist(feature_values["reggae"], bins = n_bins)
    axs[4][0].set_title("reggae")
    axs[0][1].hist(feature_values["classical"], bins = n_bins)
    axs[0][1].set_title("classicalis superioris fuckus other genrus")
    axs[1][1].hist(feature_values["rock"], bins = n_bins)
    axs[1][1].set_title("rock")
    axs[2][1].hist(feature_values["hiphop"], bins = n_bins)
    axs[2][1].set_title("hiphop")
    axs[3][1].hist(feature_values["country"], bins = n_bins)
    axs[3][1].set_title("køntri")
    axs[4][1].hist(feature_values["jazz"], bins = n_bins)
    axs[4][1].set_title("jazz")
    fig.suptitle(feature)

    return axs
    
