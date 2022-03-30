
import re
class SongFeatures:
    def __init__(self, arr):
        Track_ID, File, zero_cross_rate_mean, zero_cross_rate_std, rmse_mean, rmse_var, spectral_centroid_mean, spectral_centroid_var,\
        spectral_bandwidth_mean,spectral_bandwidth_var,spectral_rolloff_mean,spectral_rolloff_var,spectral_contrast_mean,spectral_contrast_var,spectral_flatness_mean,\
        spectral_flatness_var,chroma_stft_1_mean,chroma_stft_2_mean,chroma_stft_3_mean,chroma_stft_4_mean,chroma_stft_5_mean,chroma_stft_6_mean,chroma_stft_7_mean,\
        chroma_stft_8_mean,chroma_stft_9_mean,chroma_stft_10_mean,chroma_stft_11_mean,chroma_stft_12_mean,chroma_stft_1_std,chroma_stft_2_std,chroma_stft_3_std,\
        chroma_stft_4_std,chroma_stft_5_std,chroma_stft_6_std,chroma_stft_7_std,chroma_stft_8_std,chroma_stft_9_std,chroma_stft_10_std,chroma_stft_11_std,chroma_stft_12_std,\
        tempo, mfcc_1_mean, mfcc_2_mean, mfcc_3_mean, mfcc_4_mean, mfcc_5_mean, mfcc_6_mean, mfcc_7_mean, mfcc_8_mean,mfcc_9_mean,mfcc_10_mean,mfcc_11_mean,\
        mfcc_12_mean,mfcc_1_std,mfcc_2_std,mfcc_3_std,mfcc_4_std,mfcc_5_std,mfcc_6_std,mfcc_7_std,mfcc_8_std,mfcc_9_std,mfcc_10_std,mfcc_11_std,mfcc_12_std,\
        GenreID,Genre,Type = arr
        self.Track_ID,self.File,self.zero_cross_rate_mean,self.zero_cross_rate_std,self.rmse_mean,self.rmse_var,self.spectral_centroid_mean,self.spectral_centroid_var,\
        self.spectral_bandwidth_mean,self.spectral_bandwidth_var,self.spectral_rolloff_mean,self.spectral_rolloff_var,self.spectral_contrast_mean,self.spectral_contrast_var,self.spectral_flatness_mean,\
        self.spectral_flatness_var,self.chroma_stft_1_mean,self.chroma_stft_2_mean,self.chroma_stft_3_mean,self.chroma_stft_4_mean,self.chroma_stft_5_mean,self.chroma_stft_6_mean,self.chroma_stft_7_mean,\
        self.chroma_stft_8_mean,self.chroma_stft_9_mean,self.chroma_stft_10_mean,self.chroma_stft_11_mean,self.chroma_stft_12_mean,self.chroma_stft_1_std,self.chroma_stft_2_std,self.chroma_stft_3_std,\
        self.chroma_stft_4_std,self.chroma_stft_5_std,self.chroma_stft_6_std,self.chroma_stft_7_std,self.chroma_stft_8_std,self.chroma_stft_9_std,self.chroma_stft_10_std,self.chroma_stft_11_std,self.chroma_stft_12_std,\
        self.tempo,self.mfcc_1_mean,self.mfcc_2_mean,self.mfcc_3_mean,self.mfcc_4_mean,self.mfcc_5_mean,self.mfcc_6_mean,self.mfcc_7_mean,self.mfcc_8_mean,self.mfcc_9_mean,self.mfcc_10_mean,self.mfcc_11_mean,\
        self.mfcc_12_mean,self.mfcc_1_std,self.mfcc_2_std,self.mfcc_3_std,self.mfcc_4_std,self.mfcc_5_std,self.mfcc_6_std,self.mfcc_7_std,self.mfcc_8_std,self.mfcc_9_std,self.mfcc_10_std,self.mfcc_11_std,self.mfcc_12_std,\
        self.GenreID,self.Genre,self.Type = Track_ID, File, zero_cross_rate_mean, zero_cross_rate_std, rmse_mean, rmse_var, spectral_centroid_mean, spectral_centroid_var,\
        spectral_bandwidth_mean,spectral_bandwidth_var,spectral_rolloff_mean,spectral_rolloff_var,spectral_contrast_mean,spectral_contrast_var,spectral_flatness_mean,\
        spectral_flatness_var,chroma_stft_1_mean,chroma_stft_2_mean,chroma_stft_3_mean,chroma_stft_4_mean,chroma_stft_5_mean,chroma_stft_6_mean,chroma_stft_7_mean,\
        chroma_stft_8_mean,chroma_stft_9_mean,chroma_stft_10_mean,chroma_stft_11_mean,chroma_stft_12_mean,chroma_stft_1_std,chroma_stft_2_std,chroma_stft_3_std,\
        chroma_stft_4_std,chroma_stft_5_std,chroma_stft_6_std,chroma_stft_7_std,chroma_stft_8_std,chroma_stft_9_std,chroma_stft_10_std,chroma_stft_11_std,chroma_stft_12_std,\
        tempo, mfcc_1_mean, mfcc_2_mean, mfcc_3_mean, mfcc_4_mean, mfcc_5_mean, mfcc_6_mean, mfcc_7_mean, mfcc_8_mean,mfcc_9_mean,mfcc_10_mean,mfcc_11_mean,\
        mfcc_12_mean,mfcc_1_std,mfcc_2_std,mfcc_3_std,mfcc_4_std,mfcc_5_std,mfcc_6_std,mfcc_7_std,mfcc_8_std,mfcc_9_std,mfcc_10_std,mfcc_11_std,mfcc_12_std,\
        GenreID,Genre,Type


def readGenreClassData(path_to_file):
    '''
    Read genre class data from path_to_file and returns a dict of SongFeatures [from TrackID to SongFeature]
    '''
    with open(path_to_file) as f:
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
