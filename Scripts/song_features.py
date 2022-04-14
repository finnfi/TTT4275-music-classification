
import re
class SongFeatures:
    def __init__(self, arr):
        self.Track_ID,self.File,self.zero_cross_rate_mean,self.zero_cross_rate_std,self.rmse_mean,self.rmse_var,self.spectral_centroid_mean,self.spectral_centroid_var,\
        self.spectral_bandwidth_mean,self.spectral_bandwidth_var,self.spectral_rolloff_mean,self.spectral_rolloff_var,self.spectral_contrast_mean,self.spectral_contrast_var,self.spectral_flatness_mean,\
        self.spectral_flatness_var,self.chroma_stft_1_mean,self.chroma_stft_2_mean,self.chroma_stft_3_mean,self.chroma_stft_4_mean,self.chroma_stft_5_mean,self.chroma_stft_6_mean,self.chroma_stft_7_mean,\
        self.chroma_stft_8_mean,self.chroma_stft_9_mean,self.chroma_stft_10_mean,self.chroma_stft_11_mean,self.chroma_stft_12_mean,self.chroma_stft_1_std,self.chroma_stft_2_std,self.chroma_stft_3_std,\
        self.chroma_stft_4_std,self.chroma_stft_5_std,self.chroma_stft_6_std,self.chroma_stft_7_std,self.chroma_stft_8_std,self.chroma_stft_9_std,self.chroma_stft_10_std,self.chroma_stft_11_std,self.chroma_stft_12_std,\
        self.tempo,self.mfcc_1_mean,self.mfcc_2_mean,self.mfcc_3_mean,self.mfcc_4_mean,self.mfcc_5_mean,self.mfcc_6_mean,self.mfcc_7_mean,self.mfcc_8_mean,self.mfcc_9_mean,self.mfcc_10_mean,self.mfcc_11_mean,\
        self.mfcc_12_mean,self.mfcc_1_std,self.mfcc_2_std,self.mfcc_3_std,self.mfcc_4_std,self.mfcc_5_std,self.mfcc_6_std,self.mfcc_7_std,self.mfcc_8_std,self.mfcc_9_std,self.mfcc_10_std,self.mfcc_11_std,self.mfcc_12_std,\
        self.GenreID,self.Genre,self.Type = arr


def readGenreClassData(file_path):
    '''
    Read genre class data from path_to_file and returns a dict of SongFeatures [from TrackID to SongFeature]
    '''
    with open(file_path) as f:
        f.readline() #Remove description
        lines = f.readlines()
    dict_of_SF = dict()
    for str in lines:
        arr = re.split(r'\t+', str.rstrip('\n'))
        arr[0] = int(arr[0])
        arr[-3] = int(arr[-3])
        for i in range(2,len(arr)-3):
            arr[i] = float(arr[i])
        dict_of_SF[arr[0]] = SongFeatures(arr)
    return dict_of_SF

def genre_string_to_id(genre):
    '''
    input: genre in string format
    output: genre id
    '''
    genres = ["pop","metal", "disco", "blues", "reggae", "classical", "rock", "hiphop", "country", "jazz"]
    return genres.index(genre)

def genre_id_to_string(genre_id):
    '''
    input: genre_id
    output: genre in string format
    '''
    genres = ["pop","metal", "disco", "blues", "reggae", "classical", "rock", "hiphop", "country", "jazz"]
    return genres[genre_id]
