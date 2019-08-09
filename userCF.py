'''
@Description: user-based CF
@Author: Peng LIU, Zhihao LI, Kaiwen LUO, Jingjing WANG
@Date: 2019-08-08 18:43:02
@LastEditors: Peng LIU
@LastEditTime: 2019-08-09 22:57:33
'''
import pandas as pd
from collections import defaultdict
import re
import math
from data import DataProcess

class UserBasedCF:
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
    
    # 欧式算法
    def euclidean(self, user1, user2):
        # movieDict = self.classifyMovie()
        # pull out two users from movieDict
        user1_data = self.trainMovieDict[user1]
        user2_data = self.trainMovieDict[user2]
        distance = 0
        # cal euclidean distance
        for key in user1_data.keys():
            if key in user2_data.keys():
                # the smaller, the more simularity
                distance += pow(
                    float(user1_data[key][0]) - float(user2_data[key][0]), 2)
        return 1 / (1 + math.sqrt(distance))

    # 这里应该用一张 维度是(userNum, userNum)的矩阵去记录每个用户的相似度(未完成)
    def topSim(self, userID):
        res = []
        for uid in self.MovieDict.keys():
            if not uid == userID:
                similarity = self.euclidean(userID, uid)
                res.append((uid, similarity))
        res.sort(key=lambda val: val[1])
        return res[:10]

    # 预测可以根据年份去优先预测比较新的高分电影（并未实现）
    def predict(self, user, N):
        top_sim_user = self.topSim(user,algorithm)[0][0]
        items = self.trainMovieDict[top_sim_user]
        rec = []
        for item in items.keys():
            if item not in self.trainMovieDict[user].keys():
                rec.append((item, items[item]))
        rec.sort(key=lambda val: val[1], reverse=True)
        result = rec[:N]
        #for i in range(len(result)):
        #    print("Top",i+1," MovieTitle: ",result[i][1][1]," Rating: ",result[i][1][0])
        return result

    # 评估正确率 precision = R(u) 和 T(u) 重合个数 / R(U)
    # R(u): 在训练集上对用户u推荐N个物品, T(u): 用户u在测试集上评价过的物品集合
    # N是推荐电影数量, N = R(U)
    # 需要分测试集和训练集去计算, 因为推荐系统不会推荐用户评过分的电影
    def evaluation(self, N):
        count = 0
        total = 0
        trainMovieDict = self.trainMovieDict
        testMovieDict = self.testMovieDict
        for uid in trainMovieDict.keys():
            if uid not in testMovieDict.keys():
                continue
            t = 0
            pred = self.predict(uid, N, algorithm)
            for info in pred:
                if info[0] in testMovieDict[uid].keys():
                    t += 1
            p = t / N
            total += p
            count += 1
        return total / count

# if __name__ == "__main__":
#     data = DataProcess('./ml-100k/u.data')
#     trainData = DataProcess('./ml-100k/u1.base')
#     testData = DataProcess('./ml-100k/u1.test')
#     userCF = UserBasedCF(data, trainData, testData)
#     print(userCF.evaluation(10))
