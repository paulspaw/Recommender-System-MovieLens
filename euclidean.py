'''
@Description: user-based CF using Euclidean
@Author: Peng LIU, Zhihao LI, Kaiwen LUO, Jingjing WANG
@Date: 2019-08-08 18:43:02
@LastEditors: Peng LIU
@LastEditTime: 2019-08-11 16:34:19
'''
import pandas as pd
import numpy as np
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
    def __init__(self, data, trainData, testData, logReg=None, cs=0):
        if logReg:
            self.lr = logReg
        else:
            self.lr = None
        self.cs = cs
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
        if self.cs == 1:
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
                # the smaller, the more similarity
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
        return res

    # 预测可以根据年份去优先预测比较新的高分电影（并未实现）
    def predict(self, user, N, K, threshold):
        top_sim_users = self.topSim(user)[:K]
        rec = []
        rec_list = []
        for sUserInfo in top_sim_users:
            sUser = sUserInfo[0]
            items = self.trainMovieDict[sUser]
            for item in items.keys():
                if item not in self.trainMovieDict[user].keys():
                    if items[item][0] >= threshold:
                        rec_list.append(item)
                        rec.append((item, items[item]))
        rec_list = list(set(rec_list))
        rec = list(set(rec))
        rec.sort(key=lambda val: val[1], reverse=True)
        return rec[:N], rec_list

    def lrPredict(self, user, N, K, threshold):
        threshold = 0.6
        movies = self.Data.getMovies().drop(['MovieTitle'], axis=1)
        users = self.Data.getUser()
        top_sim_users = self.topSim(user)[:K]

        recMovies = []
        result = []
        rec_list = []
        for sUserInfo in top_sim_users:
            sUser = sUserInfo[0]
            items = self.trainMovieDict[sUser]
            for item in items.keys():
                if item not in self.trainMovieDict[user].keys():
                    if items[item][0] >= threshold:
                        recMovies.append([user, item])
            lrRatings = pd.DataFrame(recMovies, columns=['UserID', 'MovieID'])
            dataSet = pd.merge(lrRatings, users, on='UserID')
            dataSet = pd.merge(dataSet, movies, on='MovieID')
            predX = dataSet[['UserID', 'MovieID', 'Age',
                             'Gender', 'Occupation', 'ZipCode']]
            predY = self.lr.predict(predX)
            predY_proba = self.lr.predict_proba(predX)
            movieL = predX[['UserID', 'MovieID']].copy()
            movieL['Rating'] = predY_proba[:, 1]
            movieL = movieL[(movieL.Rating >= threshold)]['MovieID'].values
            superMovieL = self.trainMovieDict[sUser].keys()
            movieL = set(movieL).intersection(set(superMovieL))
            movieL = list(movieL)
            for movie in movieL:
                movie_title = self.trainMovieDict[sUser][movie][1]
                result.append((movie, movie_title))
                rec_list.append(movie)

        rec_list = list(set(rec_list))
        result = list(set(result))
        result.sort(key=lambda val: val[1], reverse=True)
        return result, rec_list

    # 评估正确率 precision = R(u) 和 T(u) 重合个数 / R(U)
    # R(u): 在训练集上对用户u推荐N个物品, T(u): 用户u在测试集上评价过的物品集合
    # N是推荐电影数量, N = R(U)
    # 需要分测试集和训练集去计算, 因为推荐系统不会推荐用户评过分的电影
    def evaluation(self, N, K, threshold):
        count = 0
        total = 0
        rAll = 0
        trainMovieDict = self.trainMovieDict
        testMovieDict = self.testMovieDict
        for uid in trainMovieDict.keys():
            if uid not in testMovieDict.keys():
                continue
            t = 0
            pred, _ = self.predict(uid, N, K, threshold)
            for info in pred:
                if info[0] in testMovieDict[uid].keys():
                    t += 1
            p = t / N
            total += p
            count += 1
            rAll += len(testMovieDict[uid])
        return total / count, t / rAll

    def lrEvaluation(self, N, K, threshold):
        count = 0
        total = 0
        rAll = 0
        trainMovieDict = self.trainMovieDict
        testMovieDict = self.testMovieDict
        for uid in trainMovieDict.keys():
            if uid not in testMovieDict.keys():
                continue
            t = 0
            pred = self.lrPredict(uid, N, K, threshold)
            for info in pred:
                if info[0] in testMovieDict[uid].keys():
                    t += 1
            p = t / N
            total += p
            count += 1
            rAll += len(testMovieDict[uid])
        return total / count, t / rAll

    def Recall(self, train, test, N, K, lr=0):
        hit = 0
        all = 0
        for user in train.keys():
            if user not in test.keys():
                continue
            tu = test[user]
            if lr == 1:
                _, recommend_list = self.lrPredict(user, N, K, 4)
            else:
                _, recommend_list = self.predict(user, N, K, 4)
            for item in recommend_list:
                if item in tu:
                    hit += 1
            all += len(tu)
        return hit/(all*1.0)

    def Precision(self, train, test, N, K, lr=0):
        hit = 0
        all = 0
        for user in train.keys():
            if user not in test.keys():
                continue
            tu = test[user]
            if lr == 1:
                _, recommend_list = self.lrPredict(user, N, K, 4)
            else:
                _, recommend_list = self.predict(user, N, K, 4)
            for item in recommend_list:
                if item in tu:
                    hit += 1
                all += N
        else:
            return hit/(all*1.0)

    def All_item(self, train):
        all_items = set()
        for user in train.keys():
            for item in train[user]:
                all_items.add(item)
        return all_items

    def Coverage(self, train, test, N, K, all_items, lr=0):
        recommend_items = set()
        for user in train.keys():
            if lr == 1:
                _, recommend_list = self.lrPredict(user, N, K, 4)
            else:
                _, recommend_list = self.predict(user, N, K, 4)
    #         recommend_list = get_recommendation(N, user, K, W, train)
            for item in recommend_list:
                recommend_items.add(item)
        return len(recommend_items)/(len(all_items)*1.0)

    def Item_popularity(self, train):
        item_popularity = {}
        for user, items in train.items():
            for item in items:
                if item not in item_popularity.keys():
                    item_popularity[item] = 0
                item_popularity[item] += 1
        return item_popularity

    def Popularity(self, train, test, N, K, item_popularity, lr=0):
        ret = 0
        n = 0
        for user in train.keys():
            if lr == 1:
                _, recommend_list = self.lrPredict(user, N, K, 4)
            else:
                _, recommend_list = self.predict(user, N, K, 4)
    #         recommend_list = get_recommendation(N, user, K, W, train)
            for item in recommend_list:
                ret += np.log(1+item_popularity[item])
                n += 1
        return ret/(n*1.0)

    def recallAndPrecision(self, K, N, lr=0):
        train_set = self.trainMovieDict
        test_set = self.testMovieDict

        all_item = self.All_item(train_set)
        item_popularity = self.Item_popularity(train_set)

        if lr == 1:
            recall = self.Recall(train_set, test_set, N, K, 1)
            precision = self.Precision(train_set, test_set, N, K, 1)
            cover = self.Coverage(train_set, test_set, N, K, all_item, 1)
            popular = self.Popularity(
                train_set, test_set, N, K, item_popularity, 1)
            return precision, recall, cover, popular
        else:
            recall = self.Recall(train_set, test_set, N, K)
            precision = self.Precision(train_set, test_set, N, K)
            cover = self.Coverage(train_set, test_set, N, K, all_item)
            popular = self.Popularity(
                train_set, test_set, N, K, item_popularity)
        return precision, recall, cover, popular

    def xxx(self):
        a = self.Precision(self.trainMovieDict, self.testMovieDict, 5, 10, 1)
        return a


if __name__ == "__main__":
    data = DataProcess('./ml-100k/u.data')
    trainData = DataProcess('./ml-100k/u1.base')
    testData = DataProcess('./ml-100k/u1.test')

    trainLr = trainData.merge_lrRating_movies()
    trainX = trainLr[['UserID', 'MovieID', 'Age',
                      'Gender', 'Occupation', 'ZipCode']]
    trainY = trainLr['Rating']

    testLr = testData.merge_lrRating_movies()
    testX = testLr[['UserID', 'MovieID', 'Age',
                    'Gender', 'Occupation', 'ZipCode']]
    testY = testLr['Rating']

    lr = LogisticRegression()
    lr.fit(trainX, trainY.ravel())
    predY = lr.predict(testX)
    # acc_log = round(lr.score(trainX, trainY) * 100, 2)
    # mse = mean_squared_error(testY, predY)
    UserCFEuclidean = UserCFEuclidean(data, trainData, testData, lr)
    # UserCFEuclidean.recallAndPrecision(5, 5)

    # print('(without logistic regression)')
    # print("%5s%5s%20s%20s%20s%20s" %
    #       ('K', 'N', 'precision(%)', "recall(%)", 'coverage(%)', 'popularity'))
    # K 选取临近的用户数量
    # N 输出推荐电影的数量
    N = 10
    # for K in [5, 10, 20, 40, 80, 160]:
    #     precision, recall, coverage, popularity = UserCFEuclidean.recallAndPrecision(
    #         K, N)
    #     print('%5d%5d%19.3f%19.3f%19.3f%19.3f' %
    #           (K, N, precision * 100, recall * 100, coverage * 100, popularity))

    # print()
    print('(with logistic regression)')
    print("%5s%5s%20s%20s%20s%20s" %
          ('K', 'N', 'precision(%)', "recall(%)", 'coverage(%)', 'popularity'))
    # K 选取临近的用户数量
    # N 输出推荐电影的数量
    N = 10
    for K in [5, 10, 20, 40, 80, 160]:
        precision, recall, coverage, popularity = UserCFEuclidean.recallAndPrecision(
            K, N, 1)
        print('%5d%5d%19.3f%19.3f%19.3f%19.3f' %
              (K, N, precision * 100, recall * 100, coverage * 100, popularity))
