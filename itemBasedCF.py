'''
@Description: item-based 
@Author: Peng LIU
@Date: 2019-08-10 14:59:07
@LastEditors: Peng LIU
@LastEditTime: 2019-08-11 19:36:29
'''
#coding=utf-8
import pandas as pd
import math
from data import DataProcess

class ItemBasedCF:

    def __init__(self,trainfile,testfile):
        self.trainMovieDict = self.classifyMovie(trainData)
        self.testMovieDict = self.classifyMovie(testData)
        self.itemSimBest = self.ItemSimilarity()

    def classifyMovie(self, data):
        movieDict = {}
        df_rate = data.getRating()
        df_movie = data.getMovies()
        rating_movies = pd.merge(df_rate, df_movie,
                                    on='MovieID').sort_values('UserID')

        for index, row in rating_movies.iterrows():
            if not row["UserID"] in movieDict.keys():
                movieDict[row["UserID"]] = {
                    row["MovieID"]: row["Rating"]
                    }
            else:
                movieDict[row["UserID"]][row["MovieID"]] = row["Rating"]
        return movieDict

    def ItemSimilarity(self):
        #存放最终的物品相似度矩阵
        item_similarity = dict()
        #存放每个电影的评分人数
        rating_num = dict()
        #物品与物品之间的相似度
        itemSimBest = dict()
        self.movie_popular = {}
        for user, movies in self.trainMovieDict.items():
            for movie in movies.keys():
                # count item popularity
                if movie not in self.movie_popular:
                    self.movie_popular[movie] = 0
                self.movie_popular[movie] += 1
        self.movie_count = len(self.movie_popular)
        
        for uid, items in self.trainMovieDict.items():
            for mid in items.keys():
                rating_num.setdefault(mid,0)
                rating_num[mid] += 1
                for mvid in items.keys():
                    if mid == mvid:
                        continue
                    item_similarity.setdefault(mid,{})
                    item_similarity[mid].setdefault(mvid,0)
                    # 两部电影的相似度
                    item_similarity[mid][mvid] += 1

        #存放最终的物品余弦相似度矩阵
        itemSimBest = {}
        for mid,related_items in item_similarity.items():
            itemSimBest.setdefault(mid,{})
            for mvid, sim in related_items.items():
                itemSimBest[mid][mvid] = sim / math.sqrt(rating_num[mid] * rating_num[mvid])
        return itemSimBest

    ##=============================================
    # Recommendation(): 推荐函数
    # userID 是需要推荐的用户
    # N 是推荐物品个数
    # K 是临近的用户数
    ##=============================================
    def Recommendation(self, userID, K, N):
        weight = {}
        count = {}
        #用户历史记录
        movie_rating = self.trainMovieDict.get(userID)

        for mid,rating in movie_rating.items():
            # itemSimBest - movieID, simularity
            for mvId, sim in sorted(self.itemSimBest[mid].items(),key=lambda x : x[1],reverse = True)[:K]:
                count.setdefault(mvId,0)
                #过滤掉推荐中看过的
                if mvId in movie_rating:
                    continue
                count[mvId] += 1
                weight.setdefault(mvId,0)
                #每一个电影推荐的分数是  电影用户打分 * 矩阵相似分数
                weight[mvId] += rating * sim

        # weight = dict(sorted(weight.items(),key = lambda x :x[1],reverse = True))
        # # print(dict(sorted(weight.items(),key = lambda x :x[1],reverse = True)[:N]))
        # for mid in weight.keys():
        #     weight[mid] = weight[mid]/count[mid]
        
        return dict(sorted(weight.items(),key = lambda x :x[1],reverse = True)[:N])

    def recallAndPrecision(self, K, N):
        hit = 0
        rec_count = 0
        test_count = 0
        # varables for coverage
        all_rec_movies = set()
        # varables for popularity
        popular_sum = 0

        for user in self.trainMovieDict.keys():
            tu = self.testMovieDict.get(user,{})
            rank = self.Recommendation(user,K,N) 
            for item,_ in rank.items():
                if item in tu:
                    hit += 1
                all_rec_movies.add(item)
                popular_sum += math.log(1 + self.movie_popular[item])
            test_count += len(tu)
            rec_count += N
            
        precision = hit / (1.0 * rec_count)
        recall = hit / (1.0 * test_count)
        coverage = len(all_rec_movies) / (1.0 * self.movie_count)
        popularity = popular_sum / (1.0 * rec_count)
        
        return precision, recall, coverage, popularity


if __name__=='__main__':
    # 输入的数据集
    totalData = './ml-100k/u.data'  #总数据集
    trainFile = './ml-100k/u1.base'  #训练集
    testFile = './ml-100k/u1.test'  #测试集

    data = DataProcess(totalData)
    trainData = DataProcess(trainFile)
    testData = DataProcess(testFile)
    ItemCF = ItemBasedCF(trainData, testData)
    # recd = ItemCF.Recommendation(1,8,10)
    N = 10

    print("%5s%5s%20s%20s%20s%20s" % ('K','N','precision(%)',"recall(%)",'coverage(%)','popularity'))
    # K 选取临近的用户数量
    # N 输出推荐电影的数量
    for K in [5,10,20,40,80,160]:
        precision,recall,coverage,popularity= ItemCF.recallAndPrecision(K,N)
        print('%5d%5d%19.3f%19.3f%19.3f%19.3f' % (K,N,precision * 100,recall * 100,coverage * 100,popularity))