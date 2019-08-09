'''
@Description: run main.py
@Author: Peng LIU, Zhihao LI, Kaiwen LUO, Jingjing WANG
@Date: 2019-08-08 18:43:02
@LastEditors: Peng LIU
@LastEditTime: 2019-08-09 13:57:52
'''
import pandas as pd
import numpy as np
from data import DataProcess
from evalMatrix import EvaluateMatrix
from model import normalizeMatrix
from recommondation import RecommondMovies

def run():
    dataFile = './ml-100k/u.data'
    # data - 处理文件信息
    data = DataProcess(dataFile)
    # matrix - 建立评分矩阵
    evalMatrix = EvaluateMatrix(data)
    # 模型初始化及更新
    nomlMatrix = normalizeMatrix(data, evalMatrix)
    # 推荐
    recommond = RecommondMovies(data,nomlMatrix)
    recommond.recommond_movies()

if __name__ == '__main__':
    run()

