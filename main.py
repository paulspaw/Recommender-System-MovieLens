'''
@Description: run main.py
@Author: Peng LIU, Zhihao LI, Kaiwen LUO, Jingjing WANG
@Date: 2019-08-08 18:43:02
@LastEditors: Peng LIU
@LastEditTime: 2019-08-10 14:20:21
'''

from data import DataProcess
from pearson import UserCF
def run():
    print("\n-------------- user-based recommodation -------------\n")
    print("\n--------------------- loading... --------------------\n")

    # 输入的数据集
    totalData = './ml-100k/u.data'  #总数据集
    trainFile = './ml-100k/u1.base'  #训练集
    testFile = './ml-100k/u1.test'  #测试集
    # 参数
    userID = 1  #用户ID
    K = 10  # K为选取相邻用户个数

    data = DataProcess(totalData)
    trainData = DataProcess(trainFile)
    testData = DataProcess(testFile)
    userCF = UserCF(data, trainData, testData)

    #根据训练集和测试集，得到预测试结果的测结果集，和测试集结果的行数一样
    #根据测试集和预测结果集，计算模型精确度
    rmse, mae = userCF.calRmseAndMae(K)
    print('rmse: %1.5f\t mae: %1.5f' % (rmse, mae))
    N = 10  #推荐没有接触过的物品的个数，存放到fileResultN文件中
    TopN = userCF.recommendation(userID, userBiasFile, N, K)

    i = 1
    for line in TopN:
        print("top",i,": ",line)
        i += 1

    print("-------------Completed!!-----------")


if __name__ == '__main__':
    run()
