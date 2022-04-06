from array import array
from song_features import SongFeatures

class GenreSet:
    def __init__(self, song_features, type):
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
            if song.Type == type:
                self.__dict__[song.Genre].append(song)

class ReducedSet:
    def __init__(self, pop,metal,disco, classical):
        self.pop = pop
        self.metal =  metal
        self.disco =  disco
        self.classical = classical
