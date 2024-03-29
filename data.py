'''
@Description: Process the data file
@Author: Peng LIU, Zhihao LI, Kaiwen LUO, Jingjing WANG
@Date: 2019-08-08 18:43:02
@LastEditors: Peng LIU
@LastEditTime: 2019-08-11 11:19:21
'''
import pandas as pd
from collections import defaultdict
import re


class DataProcess:
    def __init__(self, dataFile):
        self.genreFile = "./ml-100k/u.genre"
        self.itemFile = "./ml-100k/u.item"
        self.userFile = "./ml-100k/u.user"
        self.infoFile = "./ml-100k/u.info"
        self.OcptFile = "./ml-100k/u.occupation"
        self.dataFile = dataFile

    def getRating(self):
        ratings = pd.read_csv(
            self.dataFile,
            header=None,
            sep='\s',
            names=['UserID', 'MovieID', 'Rating', 'TimeStamp'],
            engine='python')
        ratings = ratings[['UserID', 'MovieID', 'Rating']]
        return ratings

    def getLrRating(self):
        ratings = pd.read_csv(
            self.dataFile,
            header=None,
            sep='\s',
            names=['UserID', 'MovieID', 'Rating', 'TimeStamp'],
            engine='python')
        ratings = ratings[['UserID', 'MovieID', 'Rating']]
        ratings['Rating'] = ratings['Rating'].map(
            {5: 1, 4: 1, 3: 0, 2: 0, 1: 0})
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
        movies_title = [
            'MovieID', 'MovieTitle', 'unknown', 'Action',
            'Adventure', 'Animation', 'Children\'s\'', 'Comedy', 'Crime',
            'Documentary', 'Drama', 'Fantasy', 'Film-Noir', 'Horror',
            'Musical', 'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'War',
            'Western'
        ]
        movies = movies[movies_title]
        return movies

    def getMoviesInfo(self):
        movies = self.merge_rating_movies()
        movies = movies.drop(["Rating"], axis=1)
        movieDict = {}
        for index, row in movies.iterrows():
            if row[1] not in movieDict.keys():
                movieDict[row[1]] = [row.values]
            else:
                movieDict[row[1]].append(row.values)
        return movieDict

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
        movie = movie.drop('MovieTitle', 1)
        user = self.getUser()
        temp = pd.merge(rating, user, on='UserID')
        new = pd.merge(temp, movie, on='MovieID')
        return new

    def merge_lrRating_movies(self):
        rating = self.getLrRating()
        movie = self.getMovies()
        movie = movie.drop('MovieTitle', 1)
        user = self.getUser()
        temp = pd.merge(rating, user, on='UserID')
        new = pd.merge(temp, movie, on='MovieID')
        return new

# if __name__ == "__main__":
#     user_cf = DataProcess('./ml-100k/u.data')
#     print(user_cf.merge_rating_movies())
