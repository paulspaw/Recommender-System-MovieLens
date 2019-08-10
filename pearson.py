'''
@Description: pearson algorithm
@Author: Peng LIU
@Date: 2019-08-09 14:59:00
@LastEditors: Peng LIU
@LastEditTime: 2019-08-10 19:30:43
'''
import pandas as pd
from collections import defaultdict
import re
import math
from data import DataProcess


class UserCF:
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

    # 皮尔逊算法
    def pearson(self, dataDict, user1, user2):
        user1_data = dataDict[user1]
        user2_data = dataDict[user2]
        common = {}

        for key in user1_data.keys():
            if key in user2_data.keys():
                common[key] = 1

        #共同电影数目
        commonNum = len(common)

        #如果没有共同评论过的电影，则返回0
        if commonNum == 0:
            return 0

        #计算评分和
        sum1 = sum([float(user1_data[movie][0]) for movie in common])
        sum2 = sum([float(user2_data[movie][0]) for movie in common])

        #计算评分平方和
        sum1Sq = sum([pow(float(user1_data[movie][0]), 2) for movie in common])
        sum2Sq = sum([pow(float(user2_data[movie][0]), 2) for movie in common])

        #计算乘积和
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
        #找出所有 dataDict 中评价过 movie 的用户,存入userSet
        for user in dataDict:
            if movieId in dataDict[user]:
                userSet.append(user)

        #计算相似性 scores = [(sim, userID)...]
        scores = [(self.pearson(dataDict, userID, restUser), restUser)
                  for restUser in userSet if restUser != userID]

        #按相似度排序
        scores.sort(reverse = True)

        # 选取临近K个用户
        if len(scores) <= k:  #如果小于k，只选择这些做推荐。
            for item in scores:
                users.append(item[1])  #提取每项的userId
            return users
        else:  #如果 >k,截取k个用户
            for item in scores[:k]:
                users.append(item[1])  #提取每项的userId

            return users  #返回K个最相似用户的ID

    # 计算用户对某电影的平均评分
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

    # 平均加权策略，预测userId对itemId的评分
    def getRating(self, dataDict, userId, movieId, knumber):
        sim = 0.0
        avgOther = 0.0
        weightedAverage = 0.0
        simSums = 0.0

        #获取K近邻用户(评过分的用户集)
        users = self.topKMatches(dataDict, userId, movieId, k=knumber)

        #获取userId 对某个物品评价的平均值
        avgOfUser = self.getAverage(dataDict, userId, movieId)

        #计算每个用户的加权，预测
        for restUser in users:
            #计算比较其他用户的相似度
            sim = self.pearson(dataDict, userId, restUser)
            #该用户的平均分
            avgOther = self.getAverage(dataDict, restUser, movieId)
            # 累加相似度
            simSums += abs(sim)
            weightedAverage += avgOther * abs(sim)

        # simSums为0，即该项目尚未被其他用户评分，这里的处理方法：返回用户平均分
        if simSums == 0:
            return avgOfUser
        else:
            return weightedAverage / simSums

    def loadUserBias(self, userID, trainDict):
        itemid = []
        userBias = {}
        movieData = self.data.getMovies()

        for index, row in movieData.iterrows():
            itemid.append(row['MovieID'])

        itemid = set(itemid)  #去重，并排序

        userBias.setdefault(userID, {})
        rating = 0
        for movieid in itemid:
            userBias[userID][movieid] = float(rating)

        return userBias

    def setUserBiasRating(self, userID, K):
        # 加载用户数据 物品数据和偏好，其中偏好值为0
        userBias = self.loadUserBias(userID, self.trainMovieDict)
        bias = {}
        #用户全部训练集物品 ID
        for userid in userBias.keys():
            bias.setdefault(userid, {})
            #取出某用户的物品ID
            for movieID in userBias[userid].keys():
                #基于训练集预测用户评分(用户数目<=K)
                rating = self.getRating(self.trainMovieDict, userid, movieID, K)
                userid = int(userid)
                bias[userid][movieID] = float(rating)

        return bias

    ##=======================================================
    # recommendation(): 电影推荐函数
    # userID 是需要推荐的用户
    # userBiasData 是用户对所有电影的偏好度
    # N 是推荐物品个数
    # K 是临近的用户数
    ##=======================================================
    def recommendation(self, userID, N, K):

        bias = self.setUserBiasRating(userID, K)
        # print(bias)
        #找出用户在训练集中已经评价过的物品ID
        if self.trainMovieDict[userID]:
            for mid in self.trainMovieDict[userID].keys():
                bias[userID][mid] = 0.0

        #排序取前N个物品的偏好值
        result = []
        for userid in bias:
            #排序
            movie_rating = sorted(bias[userid].items(),
                                  key=lambda x: x[1],
                                  reverse=True)
            #每个用户要输出多少个偏好物品
            if N < len(movie_rating):
                for i in range(0, N):
                    mvName = self.IdToTitle(movie_rating[i][0])
                    result.append((userid, mvName, movie_rating[i][1]))
            else:
                for i in range(0, len(movie_rating)):
                    mvName = self.IdToTitle(movie_rating[i][0])
                    result.append((userid, mvName, movie_rating[i][1]))
        return result

    # 根据 movieId 获取 movie title
    def IdToTitle(self, movieID):
        for uid in self.MovieDict.keys():
            for mid in self.MovieDict[uid].keys():
                a = int(movieID)
                b = int(mid)
                if a == b:
                    return self.MovieDict[uid][mid][1]

    ##==================================================================
    ## getRmseAndMae(): 根据对测试集的评测结果，进行评分预测
    ##==================================================================

    # 获取所有用户的预测评分，返回 predicted users' rating dictionary for all movies
    def setAllUserRating(self, K):
        predictedDict = {}

        for uid in self.testMovieDict.keys():  #test集中每个用户
            predictedDict.setdefault(uid, {})
            for movieid in self.testMovieDict[uid].keys():
                #基于训练集预测用户评分(用户数目<=K)
                rating = self.getRating(self.trainMovieDict, uid, movieid, K)
                predictedDict[uid][movieid] = float(rating)

        return predictedDict

    #计算 rmse 和 mae
    def calRmseAndMae(self, K):
        testData = self.testMovieDict
        resultData = self.setAllUserRating(K)  # 加载预测结果集

        rmse = 0.0
        mae = 0.0
        for userid in testData.keys():  #test集中每个用户
            for mvid in testData[userid].keys(
            ):  #对于test集合中每一个项目用base数据集,CF预测评分
                r1 = testData[userid][mvid][0]
                r2 = resultData[userid][mvid]
                rmse += (r1 - r2)**2
                mae += abs(r1 - r2)
        l = float(len(testData))

        return math.sqrt(rmse) / l, mae / l
