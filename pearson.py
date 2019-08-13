'''
@Description: pearson algorithm
@Author: Peng LIU
@Date: 2019-08-09 14:59:00
@LastEditors: Peng LIU
@LastEditTime: 2019-08-11 23:14:14
'''
import pandas as pd
from collections import defaultdict
import re
import math
from data import DataProcess


class UserCFPearson:
    def __init__(self, data, trainData, testData):
        self.data = data
        self.MovieDict = self.classifyMovie(data)
        self.trainMovieDict = self.classifyMovie(trainData)
        self.testMovieDict = self.classifyMovie(testData)

    def classifyMovie(self, data):
        movieDict = {}
        df_rate = data.getRating()
        df_movie = data.getMovies()
        rating_movies = pd.merge(df_rate, df_movie,
                                 on='MovieID').sort_values('UserID')

        for index, row in rating_movies.iterrows():
            if not row["UserID"] in movieDict.keys():
                movieDict[row["UserID"]] = {
                    row["MovieID"]: (row["Rating"], row["MovieTitle"])
                }
            else:
                movieDict[row["UserID"]][row["MovieID"]] = (row["Rating"],
                                                            row["MovieTitle"])
        return movieDict

    # Pearson similarity
    def pearson(self, dataDict, user1, user2):
        user1_data = dataDict[user1]
        user2_data = dataDict[user2]
        common = {}

        for key in user1_data.keys():
            if key in user2_data.keys():
                common[key] = 1

        # Number of Common Movies
        commonNum = len(common)

        # if no common movies，return 0
        if commonNum == 0:
            return 0

        # Calculate the sum
        sum1 = sum([float(user1_data[movie][0]) for movie in common])
        sum2 = sum([float(user2_data[movie][0]) for movie in common])

        # Calculate the sum of squares
        sum1Sq = sum([pow(float(user1_data[movie][0]), 2) for movie in common])
        sum2Sq = sum([pow(float(user2_data[movie][0]), 2) for movie in common])

        # calculate multiple
        PSum = sum([
            float(user1_data[i][0]) * float(user2_data[i][0]) for i in common
        ])

        num = PSum - (sum1 * sum2 / commonNum)
        den = math.sqrt((sum1Sq - pow(sum1, 2) / commonNum) *
                        (sum2Sq - pow(sum2, 2) / commonNum))
        if den == 0:
            return 0
        return num / den

    def topKMatches(self, dataDict, userID, movieId, k):
        userSet = []
        scores = []
        users = []
        # Find out all users who have evaluated movie in dataDict and save them in userSet
        for user in dataDict:
            if movieId in dataDict[user]:
                userSet.append(user)

        # similarity scores = [(sim, userID)...]
        scores = [(self.pearson(dataDict, userID, restUser), restUser)
                  for restUser in userSet if restUser != userID]

        # order by similarity
        scores.sort(reverse=True)

        # chose K neighboor
        if len(scores) <= k:  # <k，only recommend these
            for item in scores:
                users.append(item[1])  # get userId
            return users
        else:  # if >k, cut k users
            for item in scores[:k]:
                users.append(item[1])  # get userId

            return users  # return K likely ID

    # Calculate the average user rating for a movie
    def getAverage(self, dataDict, userId, movieId):
        count = 0
        sum = 0.0
        for mvid in dataDict[userId]:
            if mvid == movieId:
                sum += dataDict[userId][mvid][0]
                count = count + 1
        if count == 0:
            return 0
        else:
            return sum / count

    # Average Weighting Strategy，predict userId's rating to itemId
    def getRating(self, dataDict, userId, movieId, knumber):
        sim = 0.0
        avgOther = 0.0
        weightedAverage = 0.0
        simSums = 0.0

        # Get K-Nearest Neighbor Users (Overrated User Set)
        users = self.topKMatches(dataDict, userId, movieId, k=knumber)

        # Obtain the average value of userId's evaluation of an item
        avgOfUser = self.getAverage(dataDict, userId, movieId)

        # Calculate the weights of each user, predict
        for restUser in users:
            # Computing and comparing the similarities of other users
            sim = self.pearson(dataDict, userId, restUser)
            # Average score of the user
            avgOther = self.getAverage(dataDict, restUser, movieId)
            # Cumulative similarity
            simSums += abs(sim)
            weightedAverage += avgOther * abs(sim)

        # simSums is 0，That is, the project has not been scored by other users. The method here is to return the average score of users.
        if simSums == 0:
            return avgOfUser
        else:
            return weightedAverage / simSums

    # Get all users' predictive ratings and store them in resultFile
    def setAllUserRating(self, predictedResult, K):
        inAllnum = 0

        # file = open(fileResult, 'w')
    def loadUserBias(self, userID, trainDict):
        itemid = []
        userBias = {}
        movieData = self.data.getMovies()

        for index, row in movieData.iterrows():
            itemid.append(row['MovieID'])

        itemid = set(itemid)  # Deduplicate and sort

        userBias.setdefault(userID, {})
        rating = 0
        for movieid in itemid:
            userBias[userID][movieid] = float(rating)

        return userBias

    def setUserBiasRating(self, userID, K):
        # Load user data item data and preferences with a preference value of 0
        userBias = self.loadUserBias(userID, self.trainMovieDict)
        bias = {}
        # 用户全部训练集物品 ID
        for userid in userBias.keys():
            bias.setdefault(userid, {})
            # 取出某用户的物品ID
            for movieID in userBias[userid].keys():
                # Predicting User Score Based on Training Set (Number of Users <=K)
                rating = self.getRating(
                    self.trainMovieDict, userid, movieID, K)
                userid = int(userid)
                bias[userid][movieID] = float(rating)

        return bias

    # =======================================================
    # recommendation(): Film Recommendation Function
    # userID It's a recommended user.
    # userBiasData It's the user's preference for all movies.
    # =======================================================
    def recommendation(self, userID, N, K):

        bias = self.setUserBiasRating(userID, K)
        recommend_list =[]
        # print(bias)
        # Identify the item ID that the user has evaluated in the training set
        if self.trainMovieDict[userID]:
            for mid in self.trainMovieDict[userID].keys():
                bias[userID][mid] = 0.0

        # Preference values of the first N items in order
        result = []
        for userid in bias:
            # order
            movie_rating = sorted(bias[userid].items(),
                                  key=lambda x: x[1],
                                  reverse=True)
            # output preferences of each user
            if N < len(movie_rating):
                for i in range(0, N):
                    mvName = self.IdToTitle(movie_rating[i][0])
                    result.append((userid, mvName, movie_rating[i][1]))
                    recommend_list.append(movie_rating[i][0])
            else:
                for i in range(0, len(movie_rating)):
                    mvName = self.IdToTitle(movie_rating[i][0])
                    result.append((userid, mvName, movie_rating[i][1]))
                    recommend_list.append(movie_rating[i][0])
        return result,recommend_list

    # by movieId get movie title
    def IdToTitle(self, movieID):
        for uid in self.MovieDict.keys():
            for mid in self.MovieDict[uid].keys():
                a = int(movieID)
                b = int(mid)
                if a == b:
                    return self.MovieDict[uid][mid][1]

    # ==================================================================
    # getRmseAndMae(): According to the evaluation results of the test set, the score prediction is made.
    # ==================================================================

    # Get all users'predictive ratings and return predicted users' rating dictionary for all movies
    def setAllUserRating(self, K):
        predictedDict = {}

        for uid in self.testMovieDict.keys():  # each user in test
            predictedDict.setdefault(uid, {})
            for movieid in self.testMovieDict[uid].keys():
                # 基于训练集预测用户评分(用户数目<=K)
                rating = self.getRating(self.trainMovieDict, uid, movieid, K)
                predictedDict[uid][movieid] = float(rating)

        return predictedDict

    # 计算 rmse 和 mae
    def calRmseAndMae(self, K):
        testData = self.testMovieDict
        resultData = self.setAllUserRating(K)  # Loading Prediction Result Set

        rmse = 0.0
        mae = 0.0
        for userid in testData.keys():  # test集中每个用户
            for mvid in testData[userid].keys(
            ):  # For each item in the test set with base data set, CF predictive score
                r1 = testData[userid][mvid][0]
                r2 = resultData[userid][mvid]
                rmse += (r1 - r2)**2
                mae += abs(r1 - r2)
        l = float(len(testData))

        return math.sqrt(rmse) / l, mae / l

    def recallAndPrecision(self, K, N):
        hit = 0
        recall = 0
        precision = 0
        for user in self.trainMovieDict.keys():
            tu = self.testMovieDict.get(user,{})
            _,rank = self.recommendation(user,K,N)
            for item in rank:
                if item in tu:
                    hit += 1
            recall += len(tu)
            precision += N
            # print((hit / (recall * 1.0),hit / (precision * 1.0)))
        return (hit / (recall * 1.0),hit / (precision * 1.0))

if __name__ == "__main__":
    N = 10

    # 输入的数据集
    totalData = './ml-100k/u.data'  # dataset
    trainFile = './ml-100k/u1.base'  # train_set
    testFile = './ml-100k/u1.test'  # test_set
    
    data = DataProcess(totalData)
    trainData = DataProcess(trainFile)
    testData = DataProcess(testFile)
    userCF = UserCFPearson(data, trainData, testData)
    # According to the training set and test set, the test 
    # result set of predicting test result is obtained, 
    # which is the same as the number of rows of test result set.

    # Calculate model accuracy based on test set and prediction result set
    # rmse, mae = userCF.calRmseAndMae(K)
    # print('rmse: %1.5f\t mae: %1.5f' % (rmse, mae))
    TopN, _ = userCF.recommendation(userID=123, N=10, K=10)

    i = 1
    print('userID = 123, N = 10, K = 10')
    
    for line in TopN:
        print("top", i, ": ", line[1],line[2])
        i += 1

    # # 测算recall 和 precision
    # print("%5s%5s%20s%20s" % ('K', 'N', "recall", 'precision'))
    # for k in [5, 10, 20, 50, 100]:
    #     recall, precision = userCF.recallAndPrecision(k, N)
    #     print("%5d%5d%19.3f%%%19.3f%%" % (k,n,recall * 100,precision * 100))
