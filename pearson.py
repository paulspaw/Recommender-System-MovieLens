'''
@Description: pearson algorithm
@Author: Peng LIU
@Date: 2019-08-09 14:59:00
@LastEditors: Peng LIU
@LastEditTime: 2019-08-09 18:56:06
'''
import pandas as pd
from collections import defaultdict
import re
import math
from data import DataProcess


class UserCF:
    def __init__(self, data, trainData, testData):
        self.trainData = trainData
        self.testData = testData
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
    def pearson(self,dataDict, user1, user2):
        user1_data = dataDict[user1]
        user2_data = dataDict[user2]
        common = {}

        for key in user1_data.keys():
            if key in user2_data.keys():
                common[key] = 1
        #如果没有共同评论过的电影，则返回0
        if len(common) == 0:
            return 0
        #共同电影数目
        commonNum = len(common)

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
        result = num / den
        return result

    def topKMatches(self,dataDict, userID, itemId, k):
        userSet = []
        scores = []
        users = []
        #找出所有MovieDict中评价过Item的用户,存入userSet
        for user in dataDict:
            if itemId in dataDict[user]:
                userSet.append(user)
        # print(userSet)
        #计算相似性
        scores = [(self.pearson(dataDict,userID, restUser), restUser)
                  for restUser in userSet if restUser != userID]
        # print (scores)

        #按相似度排序
        scores.sort()
        scores.reverse()

        if len(scores) <= k:  #如果小于k，只选择这些做推荐。
            for item in scores:
                users.append(item[1])  #提取每项的userId
            return users
        else:  #如果>k,截取k个用户
            kscore = scores[:k]
            for item in kscore:
                users.append(item[1])  #提取每项的userId
            #print users
            return users  #返回K个最相似用户的ID

    # 计算用户对某物品的平均评分
    def getAverage(self, dataDict, userId, itemId):
        count = 0
        sum = 0.0
        for item in dataDict[userId]:
            if item == itemId:
                sum += dataDict[userId][item][0]
                count = count + 1
        if count == 0:
            return 0
        else:
            return sum / count

    # 平均加权策略，预测userId对itemId的评分
    def getRating(self, dataDict,userId, itemId, knumber):
        sim = 0.0
        averageOther = 0.0
        weightedAverage = 0.0
        simSums = 0.0
        #获取K近邻用户(评过分的用户集)
        users = self.topKMatches(dataDict,userId, itemId, k=knumber)

        #获取userId 对某个物品评价的平均值
        averageOfUser = self.getAverage(dataDict,userId, itemId)

        #计算每个用户的加权，预测
        for restUser in users:
            sim = self.pearson(dataDict,userId, restUser)  #计算比较其他用户的相似度
            averageOther = self.getAverage(dataDict,restUser, itemId)  #该用户的平均分
            # 累加
            simSums += abs(sim)  #取绝对值
            weightedAverage += averageOther * abs(sim)  #累加，一些值为负

        # simSums为0，即该项目尚未被其他用户评分，这里的处理方法：返回用户平均分
        if simSums == 0:
            return averageOfUser
        else:
            return weightedAverage / simSums

    # 获取所有用户的预测评分，存放到resultFile中
    def setAllUserRating(self, avgRating, K):
        inAllnum = 0

        # file = open(fileResult, 'w')
        with open(avgRating,"w") as file:
            file.write('%s,%s,%s\n' % ('uid', 'movieid', 'rating'))
            for uid in self.testMovieDict:  #test集中每个用户
                for movieid in self.testMovieDict[uid]:
                    #基于训练集预测用户评分(用户数目<=K)
                    rating = self.getRating(self.trainMovieDict,uid, movieid, K)
                    file.write('%s,%s,%s\n' % (uid, movieid, rating))
                    inAllnum = inAllnum + 1
        file.close()


if __name__ == "__main__":
    data = DataProcess('./ml-100k/u.data')
    trainData = DataProcess('./ml-100k/u1.base')
    testData = DataProcess('./ml-100k/u1.test')
    userCF = UserCF(data, trainData, testData)
    userCF.setAllUserRating('./avgRating.data',5)
