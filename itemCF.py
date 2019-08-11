'''
@Description: 
@Author: Peng LIU
@Date: 2019-08-10 14:59:07
@LastEditors: Peng LIU
@LastEditTime: 2019-08-11 12:50:42
'''
#coding=utf-8
import pandas as pd
import math
from data import DataProcess
from recallAndPrecision import Evaluation

class ItemBasedCF:

    def __init__(self,data,trainData,testData):
        self.data = data
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
        result = []
        recommend_list = []
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
        weight = dict(sorted(weight.items(),key = lambda x :x[1],reverse = True)[:N])
        # print(sorted(weight.items(),key = lambda x :x[1],reverse = True)[:N])
        for key,values in  weight.items():
            recommend_list.append(key)
            # print(key,values)
            
        for mid in weight.keys():
            mvName = self.IdToTitle(mid)
            result.append((mvName,weight[mid]))
        return result,weight,recommend_list
    
    # 根据 movieId 获取 movie title
    def IdToTitle(self, movieID):
        movies = self.data.getMovies()
        for index, row in movies.iterrows():
            if movieID == row['MovieID']:
                return row['MovieTitle']
        

    def recallAndPrecision(self, K, N):
        hit = 0
        recall = 0
        precision = 0
        for user in self.trainMovieDict.keys():
            tu = self.testMovieDict.get(user,{})
            _,rank,_ = self.Recommendation(user,K,N)
            print(rank.items()) 
            for item,_ in rank.items():
                print(item)
                if item in tu:
                    hit += 1
            recall += len(tu)
            precision += N
        # for key,values in  rank.items():
        #     print(key)
        return (hit / (recall * 1.0),hit / (precision * 1.0))
    

# if __name__=='__main__':
#     # 输入的数据集
#     totalData = './ml-100k/u.data'  #总数据集
#     trainFile = './ml-100k/u1.base'  #训练集
#     testFile = './ml-100k/u1.test'  #测试集

#     data = DataProcess(totalData)
#     trainData = DataProcess(trainFile)
#     testData = DataProcess(testFile)
#     ItemCF = ItemBasedCF(data,trainData, testData)

#     trainDict = ItemCF.classifyMovie(trainData)
#     testDict = ItemCF.classifyMovie(testData)
#     _,_,recommend_list = ItemCF.Recommendation(555,5,5)
#     eva = Evaluation(trainDict,testDict,5,5,recommend_list)

#     print(eva.Precision())

    # 测算recall 和 precision
    # print("%5s%5s%20s%20s" % ('K','N',"recall",'precision'))
    # # K 选取临近的用户数量
    # # N 输出推荐电影的数量
    # for k in [5,10,20,40,80,160]:
    #     for n in [5,10,15,20]:
    #         recall,precision = ItemCF.recallAndPrecision(k,n)
    #         print("%5d%5d%19.3f%%%19.3f%%" % (k,n,recall * 100,precision * 100))