import pandas as pd
from collections import defaultdict
import re


class UserCF:
    def __init__(self):
        self.genreFile = "./ml-100k/u.genre"
        self.itemFile = "./ml-100k/u.item"
        self.userFile = "./ml-100k/u.user"
        self.dataFile = "./ml-100k/u.data"
        self.infoFile = "./ml-100k/u.info"
        self.OcptFile = "./ml-100k/u.occupation"

    def getRating(self):
        ratings = pd.read_csv(
            self.dataFile,
            header=None,
            sep='\s',
            names=['UserID', 'MovieID', 'Rating', 'TimeStamp'],
            engine='python')
        return ratings

    def getGenre(self):
        genres = pd.read_csv(self.genreFile,
                             header=None,
                             sep='|',
                             names=['Genre', 'GenreID'],
                             engine='python')
        return genres

    def getInfo(self):
        info = pd.read_csv(self.infoFile,
                           header=None,
                           sep='\s',
                           names=['Number', 'Info'],
                           engine='python')
        return info

    def getMovies(self):
        movies_title = [
            'MovieID', 'MovieTitle', 'Date', 'Url', 'unknown', 'Action',
            'Adventure', 'Animation', 'Children\'s\'', 'Comedy', 'Crime',
            'Documentary', 'Drama', 'Fantasy', 'Film-Noir', 'Horror',
            'Musical', 'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'War',
            'Western'
        ]

        lattr1 = [0, 1, 2, 4]
        lattr2 = [i for i in range(5, 24)]
        movies = pd.read_csv(self.itemFile,
                             header=None,
                             sep='|',
                             usecols=lattr1 + lattr2,
                             names=movies_title,
                             engine='python')
        movies["index"] = movies.index
        title = ['index', 'MovieID', 'MovieTitle']
        movies = movies[title]

        return movies

    def getUser(self):
        occupation = pd.read_csv(self.OcptFile,
                                 header=None,
                                 sep='\s',
                                 names=['Occupation'],
                                 engine='python')

        users = pd.read_csv(
            self.userFile,
            header=None,
            sep='|',
            names=['UserID', 'Age', 'Gender', 'Occupation', 'ZipCode'],
            engine='python')
        # 将age值转化为连续的数字
        age_map = {val: ii for ii, val in enumerate(set(users['Age']))}
        users['Age'] = users['Age'].map(age_map)

        # 将zipcode值转化为连续的数字
        zip_map = {val: ii for ii, val in enumerate(set(users['ZipCode']))}
        users['ZipCode'] = users['ZipCode'].map(zip_map)

        # 将F转化为0，M转化为1
        if users['Gender'].dtype != 'int64':
            gender_map = {'F': 0, 'M': 1}
            users['Gender'] = users['Gender'].map(gender_map)

        # 将各个Occupation转化为对应的ID
        if users['Occupation'].dtype != 'int64':
            occupation_map = {}
            for index, row in occupation.iterrows():
                occupation_map[row[0].lower()] = index
            users['Occupation'] = users['Occupation'].map(occupation_map)

        return users

    def merge_rating_movies(self):
        rating = self.getRating()
        movie = self.getMovies()
        ratings = pd.merge(rating, movie, on='MovieID')
        new_csv = ratings[['UserID', 'index', 'Rating']]
        return new_csv


if __name__ == "__main__":
    user_cf = UserCF()
    user_cf.merge_rating_movies()