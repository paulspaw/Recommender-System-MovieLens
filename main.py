import pandas as pd
import numpy as np
from data import DataProcess
from evalMatrix import EvaluateMatrix
from model import normalizeMatrix
from recommondation import RecommondMovies

def run():
    # data - 处理文件信息
    data = DataProcess()
    # matrix - 建立评分矩阵
    evalMatrix = EvaluateMatrix(data)
    # 模型初始化及更新
    nomlMatrix = normalizeMatrix(data, evalMatrix)
    # 推荐
    recommond = RecommondMovies(data,nomlMatrix)
    recommond.recommond_movies()

if __name__ == '__main__':
    run()
