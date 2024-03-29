'''
@Description: run main.py
@Author: Peng LIU, Zhihao LI, Kaiwen LUO, Jingjing WANG
@Date: 2019-08-08 18:43:02
@LastEditors: Peng LIU
@LastEditTime: 2019-08-11 16:38:20
'''

from data import DataProcess
from pearson import UserCFPearson
from euclidean import UserCFEuclidean
from itemCF import ItemBasedCF
from recallAndPrecision import Evaluation


def run(userID, method):
    # 输入的数据集
    totalData = './ml-100k/u.data'  # 总数据集
    trainFile = './ml-100k/u1.base'  # 训练集
    testFile = './ml-100k/u1.test'  # 测试集

    # 参数  #用户ID
    K = 5  # K为选取相邻用户个数
    N = 5  # 推荐没有接触过的物品的个数
    threshold = 4  # 评分阀值

    data = DataProcess(totalData)
    trainData = DataProcess(trainFile)
    testData = DataProcess(testFile)

    if method == 'userbased-pearson':
        #userBased - pearson
        userCF = UserCFPearson(data, trainData, testData)

        # 根据训练集和测试集，得到预测试结果的测结果集，和测试集结果的行数一样
        # 根据测试集和预测结果集，计算模型精确度
        # rmse, mae = userCF.calRmseAndMae(K)
        # print('rmse: %1.5f\t mae: %1.5f' % (rmse, mae))
        TopN, recommend_list = userCF.recommendation(userID, N, K)
        trainDict = userCF.classifyMovie(trainData)
        testDict = userCF.classifyMovie(testData)

        i = 1
        for line in TopN:
            print("top", i, ": ", line)
            i += 1

        # 测算recall 和 precision
        # print("%5s%5s%20s%20s" % ('K', 'N', "recall", 'precision'))
        # K 选取临近的用户数量
        # N 输出推荐电影的数量
        for k in [5, 10, 20, 40, 80, 160]:
            for n in [5, 10, 15, 20]:
                recall, precision = userCF.recallAndPrecision(k, n)
                # print("%5d%5d%19.3f%%%19.3f%%" % (k,n,recall * 100,precision * 100))

        return 0

    elif method == 'userbased-euclidean':
        userCF = UserCFEuclidean(data, trainData, testData)
        TopN, _ = userCF.predict(userID, N, threshold)
        precision = userCF.evaluation(N)

        i = 1
        for line in TopN:
            print("top", i, ": ", line[1][1], "\t", line[1][0])
            i += 1

        return 0

    elif method == 'itembased':
        #itemBased - 余弦函数
        ItemCF = ItemBasedCF(data, trainData, testData)
        TopN, _, recommend_list = ItemCF.Recommendation(userID, K, N)

        i = 1
        for line in TopN:
            print("top", i, ": ", line)
            i += 1

        # 测算recall 和 precision
        print("%5s%5s%20s%20s" % ('K', 'N', "recall", 'precision'))
        # K 选取临近的用户数量
        # N 输出推荐电影的数量
        for k in [5, 10, 20, 40, 80, 160]:
            for n in [10]:
                recall, precision = ItemCF.recallAndPrecision(k, n)
                # print("%5d%5d%19.3f%%%19.3f%%" % (k,n,recall * 100,precision * 100))
        return 0

    elif method == 'exit':
        return 0
    else:
        return 1

    print("-------------Completed!!-----------")


if __name__ == '__main__':
    while True:
        print("input method name (userbased-pearson,userbased-euclidean,itembased):\n")
        method = input()
        print("input user id:\n")
        userID = input()
        print("\n--------------------- loading... --------------------\n")
        method = run(int(userID), method)
        if not method:
            break
        else:
            print('wrong,please try again!\n')
