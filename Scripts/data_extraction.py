from array import array
from song_features import SongFeatures

class TrainingSet:
    def __init__(self, song_features):
        self.pop = []
        self.metal =  []
        self.disco =  []
        self.blues =  []
        self.reggae =  []
        self.classical =  []
        self.rock =  []
        self.hiphop =  []
        self.country =  []
        self.jazz =  []
        for song in song_features.values():
            if song.Type == "Train":
                self.__dict__[song.Genre].append(song)



class TestSet:
    def __init__(self, song_features):
        self.pop = []
        self.metal =  []
        self.disco =  []
        self.blues =  []
        self.reggae =  []
        self.classical =  []
        self.rock =  []
        self.hiphop =  []
        self.country =  []
        self.jazz =  []
        for song in song_features.values():
            if song.Type == "Test":
                self.__dict__[song.Genre].append(song)