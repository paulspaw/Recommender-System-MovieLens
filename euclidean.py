'''
@Description: user-based CF using Euclidean
@Author: Peng LIU, Zhihao LI, Kaiwen LUO, Jingjing WANG
@Date: 2019-08-08 18:43:02
@LastEditors: Peng LIU
@LastEditTime: 2019-08-11 16:34:19
'''
import pandas as pd
from collections import defaultdict
import re
import math
from data import DataProcess
from collections import defaultdict
from collections import Counter
import random
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_squared_error


class UserCFEuclidean:
    def __init__(self, data, trainData, testData, logReg=None):
        if logReg:
            self.lr = logReg
        else:
            self.lr = None
        self.Data = data
        self.trainData = trainData
        self.testData = testData
        self.MovieDict = self.classifyMovie(data)
        self.trainMovieDict = self.classifyMovie(trainData)
        self.testMovieDict = self.classifyMovie(testData)
        self.movieInfo = self.Data.getMoviesInfo()

    def coldStart(self, lr):
        ratings = self.Data.getRating()
        movies = self.Data.getMovies().drop(['MovieTitle'], axis=1)
        users = self.Data.getUser()

        userNum = users['UserID'].values.argmax() + 1
        movieNum = movies['MovieID'].values.argmax() + 1

        # 创建一个字典存 每个user看过的电影， 格式 {userID： movieList[]}
        # 被评论超过平均值次数的movie列为热门电影
        uDict = {}
        popular = []
        result = []  # 所有评论过的movie

        uTemp = pd.merge(ratings, users, on='UserID')
        for uid in range(1, userNum+1):
            temp = uTemp[(uTemp.UserID == uid)]
            uDict[uid] = temp['MovieID'].values
            result += temp['MovieID'].values.tolist()

        counter = Counter(result)
        maxCommented = max(counter.values())
        minCommented = min(counter.values())
        avgCommented = (maxCommented + minCommented) / 2
        for mid in counter.keys():
            if counter[mid] >= avgCommented:
                popular.append(mid)

        # new Ratings needs to insert
        nrLs = []  # new rating list
        for pMid in popular:
            for uid in uDict.keys():
                if pMid not in uDict[uid]:
                    nrLs.append([uid, pMid, 0])
        nRatings = pd.DataFrame(nrLs, columns=['UserID', 'MovieID', 'Rating'])
        dataSet = pd.merge(nRatings, users, on='UserID')
        dataSet = pd.merge(dataSet, movies, on='MovieID')

        predX = dataSet[['UserID', 'MovieID', 'Age',
                         'Gender', 'Occupation', 'ZipCode']]
        predY = lr.predict(predX)

        nRatings = nRatings.drop(['Rating'], axis=1)
        nRatings['Rating'] = predY
        return nRatings

    def classifyMovie(self, data):
        movieDict = {}
        df_rate = data.getRating()
        if self.lr:
            nRatings = self.coldStart(self.lr)
            df_rate = pd.concat([df_rate, nRatings])
        df_movie = data.getMovies()
        rating_movies = pd.merge(
            df_rate, df_movie, on='MovieID').sort_values('UserID')

        for index, row in rating_movies.iterrows():
            if not row["UserID"] in movieDict.keys():
                movieDict[row["UserID"]] = {
                    row["MovieID"]: (row["Rating"], row["MovieTitle"])}
            else:
                movieDict[row["UserID"]][row["MovieID"]] = (
                    row["Rating"], row["MovieTitle"])
        return movieDict

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
                distance += pow(float(user1_data[key][0]) -
                                float(user2_data[key][0]), 2)
        return 1 / (1 + math.sqrt(distance))

    # 这里应该用一张 维度是(userNum, userNum)的矩阵去记录每个用户的相似度(未完成)
    def topSim(self, userID):
        res = []
        for uid in self.MovieDict.keys():
            if not uid == userID:
                similarity = self.euclidean(userID, uid)
                res.append((uid, similarity))
        res.sort(key=lambda val: val[1])
        return res[0]

    # 预测可以根据年份去优先预测比较新的高分电影（并未实现）
    def predict(self, user, N, threshold):
        top_sim_user = self.topSim(user)[0]
        rec = []
        rec_list = []
        items = self.trainMovieDict[top_sim_user]
        for item in items.keys():
            if item not in self.trainMovieDict[user].keys():
                if items[item][0] >= threshold:
                    rec_list.append(item)
                    rec.append((item, items[item]))
        rec.sort(key=lambda val: val[1], reverse=True)
        return rec[:N], rec_list

    def lrPredict(self, lr, user, N):
        movies = self.Data.getMovies().drop(['MovieTitle'], axis=1)
        users = self.Data.getUser()
        top_sim_user = self.topSim(user)[0]
        items = items = self.trainMovieDict[top_sim_user]
        recMovies = []
        for item in items.keys():
            if item not in self.trainMovieDict[user].keys():
                recMovies.append([user, item])
        lrRatings = pd.DataFrame(recMovies, columns=['UserID', 'MovieID'])
        dataSet = pd.merge(lrRatings, users, on='UserID')
        dataSet = pd.merge(dataSet, movies, on='MovieID')
        predX = dataSet[['UserID', 'MovieID', 'Age',
                         'Gender', 'Occupation', 'ZipCode']]
        predY = lr.predict(predX)
        movieL = predX[['UserID', 'MovieID']].copy()
        movieL['Rating'] = predY
        movieL = movieL[(movieL.Rating >= 4)]['MovieID'].values

        # 得到推荐的 movie list result
        # 开始推荐

        result = []
        for movie in movieL:
            movie_title = self.trainMovieDict[top_sim_user][movie][1]
            result.append((movie, movie_title))
        return random.sample(result, 10)

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
            pred, _ = self.predict(uid, N, 4)
            for info in pred:
                if info[0] in testMovieDict[uid].keys():
                    t += 1
            p = t / N
            total += p
            count += 1
        return total / count

    def lrEvaluation(self, lr, N):
        count = 0
        total = 0
        trainMovieDict = self.trainMovieDict
        testMovieDict = self.testMovieDict
        for uid in trainMovieDict.keys():
            if uid not in testMovieDict.keys():
                continue
            t = 0
            pred = self.lrPredict(lr, uid, N)
            for info in pred:
                if info[0] in testMovieDict[uid].keys():
                    t += 1
            p = t / N
            total += p
            count += 1
        return total / count


if __name__ == "__main__":
    data = DataProcess('./ml-100k/u.data')
    trainData = DataProcess('./ml-100k/u1.base')
    testData = DataProcess('./ml-100k/u1.test')
    UserCFEuclidean = UserCFEuclidean(data, trainData, testData)
    print('Start -----------------------------------')
    topN,_ = UserCFEuclidean.predict(123,5,4)
    print(topN)
    score = UserCFEuclidean.evaluation(5)
    print(f'Precision {score}')
    print('Completed -------------------------------')
