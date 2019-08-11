import pandas as pd
import numpy as np
from data import DataProcess
from evalMatrix import EvaluateMatrix
from model import normalizeMatrix


class RecommondMovies:
    def __init__(self,data,nomlMatrix):
        self.movies = data.getMovies()
        self.predicts = nomlMatrix.trainingModel()

    def recommond_movies(self):
        user_id = input('您要想哪位用户进行推荐？请输入用户编号：')
        sortedResult = self.predicts[:, int(user_id)].argsort()[::-1]
        # argsort()函数返回的是数组值从小到大的索引值; argsort()[::-1] 返回的是数组值从大到小的索引值
        idx = 0
        print('为该用户推荐的评分最高的10部电影是：'.center(80, '='))
        # center() 返回一个原字符串居中,并使用空格填充至长度 width 的新字符串。默认填充字符为空格。
        for i in sortedResult:
            print(
                '评分: %.2f, 电影名: %s' %
                (self.predicts[i, int(user_id)], self.movies.iloc[i]['MovieTitle']))
            # .iloc的用法：https://www.cnblogs.com/harvey888/p/6006200.html
            idx += 1
            if idx == 10: break


# if __name__ == "__main__":
#     data = DataProcess()
#     evalMatrix = EvaluateMatrix(data)
#     nomlMatrix = normalizeMatrix(data, evalMatrix)
#     recommond = RecommondMovies(data,nomlMatrix)
#     recommond.recommond_movies()