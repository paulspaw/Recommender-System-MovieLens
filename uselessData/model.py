import pandas as pd
import numpy as np
from data import DataProcess
from evalMatrix import EvaluateMatrix
import tensorflow as tf


class normalizeMatrix:
    def __init__(self, data, evalMatrix):
        self.ratings_df = data.merge_rating_movies()
        self.userNo,self.movieNo,self.rating, self.record = evalMatrix.ratingMatrix()
        self.ratingNorm, self.ratingMean = self.normalizeRatings()
        self.loss,self.train, self.X, self.theta = self.buildModel()

    def normalizeRatings(self):
        #m代表电影数量，n代表用户数量
        movieNum, userNum = self.rating.shape
        #每部电影的平均得分
        rating_mean = np.zeros((movieNum, 1))
        #处理过的评分
        rating_norm = np.zeros((movieNum, userNum))

        for i in range(movieNum):
            idx = self.record[i, :] != 0
            #每部电影的评分，[i，:]表示每一行的所有列
            rating_mean[i] = np.mean(self.rating[i, idx])
            #第i行，评过份idx的用户的平均得分；
            #np.mean() 对所有元素求均值
            rating_norm[i, idx] -= rating_mean[i]
            #rating_norm = 原始得分-平均得分

            #对值为NaNN进行处理，改成数值0
            rating_norm = np.nan_to_num(rating_norm)
            rating_mean = np.nan_to_num(rating_mean)

        return rating_norm, rating_mean

    def buildModel(self):
        num_features = 10
        X_parameters = tf.Variable(tf.random.normal([self.movieNo, num_features],stddev = 0.35))
        Theta_parameters = tf.Variable(tf.random.normal([self.userNo, num_features],stddev = 0.35))
        #tf.Variables()初始化变量
        #tf.random_normal()函数用于从服从指定正太分布的数值中取出指定个数的值，mean: 正态分布的均值。stddev: 正态分布的标准差。dtype: 输出的类型
        loss = 1/2 * tf.reduce_sum(((tf.matmul(X_parameters, Theta_parameters, transpose_b = True) - self.ratingNorm) * self.record) ** 2) + 1/2 * (tf.reduce_sum(X_parameters ** 2) + tf.reduce_sum(Theta_parameters ** 2))
        #基于内容的推荐算法模型
        optimizer = tf.compat.v1.train.AdamOptimizer(1e-4)
        # https://blog.csdn.net/lenbow/article/details/52218551
        train = optimizer.minimize(loss)
        # Optimizer.minimize对一个损失变量基本上做两件事
        # 它计算相对于模型参数的损失梯度。
        # 然后应用计算出的梯度来更新变量。
        return loss, train, X_parameters,Theta_parameters

    def trainingModel(self):
        # #用来显示标量信息
        tf.compat.v1.summary.scalar('loss',self.loss)
        summaryMerged = tf.compat.v1.summary.merge_all()
        # #merge_all 可以将所有summary全部保存到磁盘，以便tensorboard显示。
        # filename = './movie_tensorborad'
        # writer = tf.compat.v1.summary.FileWriter(filename)
        # #指定一个文件用来保存图。
        sess = tf.compat.v1.Session()
        # #https://www.cnblogs.com/wuzhitj/p/6648610.html
        init = tf.compat.v1.global_variables_initializer()
        sess.run(init)
        # #运行
        for i in range(5000):
            _, movie_summary = sess.run([self.train, summaryMerged])
            # 把训练的结果summaryMerged存在movie里
        #     writer.add_summary(movie_summary, i)
        #     # 把训练的结果保存下来
        
        Current_X_parameters, Current_Theta_parameters = sess.run([self.X, self.theta])
        # Current_X_parameters为用户内容矩阵，Current_Theta_parameters用户喜好矩阵
        predicts = np.dot(Current_X_parameters,Current_Theta_parameters.T) + self.ratingMean
        # dot函数是np中的矩阵乘法，np.dot(x,y) 等价于 x.dot(y)
        errors = np.sqrt(np.sum((predicts - self.rating)**2))
        # sqrt(arr) ,计算各元素的平方根
        # print(errors)
        return predicts


# if __name__ == "__main__":
#     data = DataProcess()
#     evalMatrix = EvaluateMatrix(data)
#     nomlMatrix = normalizeMatrix(data, evalMatrix)
#     nomlMatrix.trainingModel()