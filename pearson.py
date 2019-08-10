'''
@Description: pearson algorithm
@Author: Peng LIU
@Date: 2019-08-09 14:59:00
@LastEditors: Peng LIU
@LastEditTime: 2019-08-09 23:31:06
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
        # 如果没有共同评论过的电影，则返回0
        if len(common) == 0:
            return 0
        # 共同电影数目
        commonNum = len(common)

        # 计算评分和
        sum1 = sum([float(user1_data[movie][0]) for movie in common])
        sum2 = sum([float(user2_data[movie][0]) for movie in common])

        # 计算评分平方和
        sum1Sq = sum([pow(float(user1_data[movie][0]), 2) for movie in common])
        sum2Sq = sum([pow(float(user2_data[movie][0]), 2) for movie in common])

        # 计算乘积和
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

    def topKMatches(self, dataDict, userID, itemId, k):
        userSet = []
        scores = []
        users = []
        # 找出所有MovieDict中评价过Item的用户,存入userSet
        for user in dataDict:
            if itemId in dataDict[user]:
                userSet.append(user)
        # print(userSet)
        #计算相似性
        scores = [(self.pearson(dataDict, userID, restUser), restUser)
                  for restUser in userSet if restUser != userID]
        # print (scores)

        # 按相似度排序
        scores.sort()
        scores.reverse()

        if len(scores) <= k:  # 如果小于k，只选择这些做推荐。
            for item in scores:
                users.append(item[1])  # 提取每项的userId
            return users
        else:  # 如果>k,截取k个用户
            kScore = scores[:k]
            for item in kScore:
                users.append(item[1])  # 提取每项的userId
            # print users
            return users  # 返回K个最相似用户的ID

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
    def getRating(self, dataDict, userId, itemId, knumber):
        sim = 0.0
        averageOther = 0.0
        weightedAverage = 0.0
        simSums = 0.0
        #获取K近邻用户(评过分的用户集)
        users = self.topKMatches(dataDict, userId, itemId, k=knumber)

        #获取userId 对某个物品评价的平均值
        averageOfUser = self.getAverage(dataDict, userId, itemId)

        # 计算每个用户的加权，预测
        for restUser in users:
            sim = self.pearson(dataDict, userId, restUser)  #计算比较其他用户的相似度
            averageOther = self.getAverage(dataDict, restUser,
                                           itemId)  #该用户的平均分
            # 累加
            simSums += abs(sim)  # 取绝对值
            weightedAverage += averageOther * abs(sim)  # 累加，一些值为负

        # simSums为0，即该项目尚未被其他用户评分，这里的处理方法：返回用户平均分
        if simSums == 0:
            return averageOfUser
        else:
            return weightedAverage / simSums

    # 获取所有用户的预测评分，存放到resultFile中
    def setAllUserRating(self, predictedResult, K):
        inAllnum = 0

        # file = open(fileResult, 'w')
<<<<<<< HEAD
        with open(avgRating, "w") as file:
            file.write('%s,%s,%s\n' % ('uid', 'movieid', 'rating'))
            for uid in self.testMovieDict:  # test集中每个用户
                for movieid in self.testMovieDict[uid]:
                    # 基于训练集预测用户评分(用户数目<=K)
                    rating = self.getRating(
                        self.trainMovieDict, uid, movieid, K)
                    file.write('%s,%s,%s\n' % (uid, movieid, rating))
        file.close()

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

    def setSomeUserRating(self, userID, userBiasData, K):

        userBias = self.loadUserBias(
            userID, self.trainMovieDict)  # 加载用户数据 物品数据和偏好，其中偏好值为0
        inAllnum = 0

        with open(userBiasData, "w") as file:
            for userid in userBias:  #用户全部训练集物品 ID
                for movieID in userBias[userid]:  #取出某用户的物品ID
                    rating = self.getRating(self.trainMovieDict, userid,
                                            movieID, K)  #基于训练集预测用户评分(用户数目<=K)
                    file.write('%d,%d,%s\n' % (userid, movieID, rating))
                    inAllnum = inAllnum + 1
        file.close()

    ## 参数:fileTrain是训练文件
    ## fileSomeUser 是需要推荐的用户列表，一行为一个用户
    ## fileResult1为结果文件
    ## N是推荐物品个数。
    def recommendation(self, userID, userBiasData, N, K):

        self.setSomeUserRating(userID, userBiasData, K)
        #从结果中取得数据
        bias = {}
        for line in open(userBiasData, 'r'):
            (uid, movieid, rating) = line.split(',')
            uid = int(uid)
            bias.setdefault(uid, {})
            # print (type(uid))
            bias[uid][movieid] = float(rating)
        # print(bias)

        #找出用户在训练集中已经评价过的物品ID
        if self.trainMovieDict[userID]:
            for mid in self.trainMovieDict[userID].keys():
                # print(type(mid))
                bias[userID][mid] = 0.0

        #排序取前N个物品的偏好值
        result = []
        for userid in bias:
            movie_rating = sorted(bias[userid].items(), key=lambda x: x[1],
                       reverse=True)  #排序
            if N < len(movie_rating):  #每个用户要输出多少个偏好物品
                for i in range(0, N):
                    mvName = self.IdToTitle(movie_rating[i][0])
                    result.append((userid, mvName, movie_rating[i][1]))
            else:
                for i in range(0, len(movie_rating)):
                    mvName = self.IdToTitle(movie_rating[i][0])
                    result.append((userid, mvName, movie_rating[i][1]))
        return result

    def IdToTitle(self,movieID):
        for uid in self.MovieDict.keys():
            for mid in self.MovieDict[uid].keys():        
                a = int(movieID)
                b = int(mid)
                if a == b:
                    return self.MovieDict[uid][mid][1]
                 
        




    ##==================================================================
    ## getRmseAndMae(): 根据对测试集的评测结果，进行评分预测
    ##
    ## 参数:testFile 是测试集文件，包括了用户的评分
    ##     resultFile 为对测试集预测的结果文件，包括了对用户预测的评分
    ##==================================================================
    def ResultDict(self,predictedResult):

        result = {}
        for line in open(predictedResult, 'r'):
            (userid, movieid, rating) = line.split(',')  #数据集中每行有3项
            uid = int(userid)
            movieid = int(movieid)
            result.setdefault(uid, {})
            result[uid][movieid] = float(rating)
        return result

    def calRmseAndMae(self, predictedResult):
        testData = self.testMovieDict
        resultData = self.ResultDict(predictedResult)  # 加载预测结果集
        # print(resultData)
        rmse = 0.0
        mae = 0.0
        for userid in testData.keys():  #test集中每个用户
            for mvid in testData[userid].keys():  #对于test集合中每一个项目用base数据集,CF预测评分
                # print("usrid: ",userid,"mvid: ", mvid)
                r1 = testData[userid][mvid][0]
                r2 = resultData[userid][mvid]
                # print (r1,r2)
                rmse += (r1 - r2)**2
                mae += abs(r1 - r2)
        l = float(len(testData))
        return math.sqrt(rmse) / l, mae / l

# if __name__ == "__main__":
#     data = DataProcess('./ml-100k/u.data')
#     trainData = DataProcess('./ml-100k/u1.base')
#     testData = DataProcess('./ml-100k/u1.test')
#     userCF = UserCF(data, trainData, testData)
#     userCF.setAllUserRating('./avgRating.data', 5)

