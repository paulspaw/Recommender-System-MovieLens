'''
@Description: run main.py
@Author: Peng LIU, Zhihao LI, Kaiwen LUO, Jingjing WANG
@Date: 2019-08-08 18:43:02
@LastEditors: Peng LIU
@LastEditTime: 2019-08-10 20:38:02
'''

from data import DataProcess
from pearson import UserCFPearson
from euclidean import UserCFEuclidean
from itemCF import ItemBasedCF

def run(method):
    # 输入的数据集
    totalData = './ml-100k/u.data'  #总数据集
    trainFile = './ml-100k/u1.base'  #训练集
    testFile = './ml-100k/u1.test'  #测试集
    
    # 参数
    userID = 1  #用户ID
    K = 10  # K为选取相邻用户个数
    N = 10  #推荐没有接触过的物品的个数

    data = DataProcess(totalData)
    trainData = DataProcess(trainFile)
    testData = DataProcess(testFile)
    
    if method == 'userbased-pearson':
        #userBased - pearson
        userCF = UserCFPearson(data, trainData, testData)

        #根据训练集和测试集，得到预测试结果的测结果集，和测试集结果的行数一样
        #根据测试集和预测结果集，计算模型精确度
        rmse, mae = userCF.calRmseAndMae(K)
        print('rmse: %1.5f\t mae: %1.5f' % (rmse, mae))
        TopN = userCF.recommendation(userID, N, K)
        i = 1
        for line in TopN:
            print("top",i,": ",line)
            i += 1
        return 0
            
    elif method == 'userbase-euclidean':
        userCF = UserCFEuclidean(data, trainData, testData)
        TopN = userCF.predict(userID,N)
        return 0

    elif method == 'itembased':
        #itemBased - 余弦函数
        ItemCF = ItemBasedCF(data,trainData, testData)
        TopN = ItemCF.Recommendation(userID,K,N)
        i = 1
        for line in TopN:
            print("top",i,": ",line)
            i += 1
        return 0
    
    elif method == 'exit':
        return 0
    else:
        return 1

        # 测算recall 和 precision
        # print("%5s%5s%20s%20s" % ('K','N',"recall",'precision'))
        # # K 选取临近的用户数量
        # # N 输出推荐电影的数量
        # for K in [5,10,20,40,80,160]:
        #     for N in [5,10,15,20]:
        #         recall,precision = ItemCF.recallAndPrecision(K,N)
        #         print("%5d%5d%19.3f%%%19.3f%%" % (K,N,recall * 100,precision * 100))

    print("-------------Completed!!-----------")


if __name__ == '__main__':
    while True:
        print("input method name (userbased-pearson,userbased-euclidean,itembased):\n")
        method = input()
        print("\n--------------------- loading... --------------------\n")
        method = run(method)
        if not method:
            break
        else:
            print('wrong,please try again!\n')
    


