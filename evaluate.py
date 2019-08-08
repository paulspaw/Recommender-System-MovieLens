import pandas as pd
import numpy as np
import tensorflow as tf
from data import UserCF

class Evaluate:
    def __init__(self,data):
        self.ratings_df = data.merge_rating_movies()

    def ratingMatrix(self):
        #userNo的最大值
        userNo = self.ratings_df['UserID'].max() + 1
        #movieNo的最大值
        movieNo = self.ratings_df['index'].max() + 1

        #创建一个值都是0的数据
        rating = np.zeros((movieNo,userNo))
        # flag = 0
        #查看矩阵ratings_df的第一维度是多少
        ratings_df_length = np.shape(self.ratings_df)[0]
        #interrows（），对表格ratings_df进行遍历
        for index,row in self.ratings_df.iterrows():
            #将ratings_df表里的'movieRow'和'userId'列，填上row的‘评分’
            rating[int(row['index']),int(row['UserID'])] = row['Rating']
            # flag += 1
        record = rating > 0
        record
        # 更改数据类型，0表示用户没有对电影评分，1表示用户已经对电影评分
        record = np.array(record, dtype = int)
        return record
        

if __name__ == "__main__":
    mv = UserCF()
    evl = Evaluate(mv)
    record = evl.ratingMatrix()
