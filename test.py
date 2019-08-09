'''
@Description: 
@Author: Peng LIU
@Date: 2019-08-09 15:23:29
@LastEditors: Peng LIU
@LastEditTime: 2019-08-09 19:12:10
'''
#--------------------------------------------------------
# Purpose:  基于已知的训练集，"测试集"中的user的item进行评分预测，并进行用户没有接触过的物品进行推荐.
#--------------------------------------------------------
from math import sqrt
import operator
##==================================
#         加载指定的训练集文件
#  参数fileName 代表某个训练集文件
##==================================
def loadMovieLensTrain(fileName):

    prefer = {}
    for line in open(fileName,'r'):       # 打开指定文件
        (userid, movieid, rating,ts) = line.split('\t')     # 数据集中每行有4项
        prefer.setdefault(userid, {})      # 设置字典的默认格式,元素是user:{}字典
        prefer[userid][movieid] = float(rating)    

    return prefer      # 格式如{'user1':{itemid:rating, itemid2:rating, ,,}, {,,,}}


##==================================
#        加载对应的测试集文件
#  参数fileName 代表某个测试集文件
##==================================
def loadMovieLensTest(fileName):

    prefer = {}
    for line in open(fileName,'r'):    
        (userid, movieid, rating,ts) = line.split('\t')   #数据集中每行有4项
        userid = str(int(userid))
        movieid = str(int(movieid))
        prefer.setdefault(userid, {})    
        prefer[userid][movieid] = float(rating) 

        #print prefer[userid]
        #print '\n'

    return prefer                   


##==================================
#        加载对应的预测结果集文件
#  参数fileName 代表某个预测结果集文件
##==================================
def loadMovieLensResult(fileName):

    prefer = {}
    for line in open(fileName,'r'):    
        (userid, movieid, rating) = line.split('\t')   #数据集中每行有3项
        prefer.setdefault(userid, {})    
        prefer[userid][movieid] = float(rating) 
    return prefer   


##==================================
#  根据trainfilename文件中物品数据，
#  将userfilename中的用户列表和trainfilename文件中的物品ID组合成一个新的字典数据
#        字典数据的结构为userid movieid rating
#  其中rating为0
##==================================
def loadMovieLensTestSomeUser(userfileName,trainfilename):
    itemid = []
    for line in open(trainfilename,'r'):       # 打开指定文件
        (userid, movieid, rating,ts) = line.split('\t')     # 数据集中每行有4项
        itemid.append(movieid)

    itemid = set(itemid)#去重，并排序


    prefer = {}
    for line in open(userfileName,'r'):    
        userid = str(int(line))   #数据集仅一行
        prefer.setdefault(userid, {})
        rating = 0
        for movieid in itemid:
            prefer[userid][movieid] = float(rating)
    print(prefer)           
    return prefer  

def OSDistance (prefer, person1, person2):


    #查找双方都评价过的项
    v1 = []
    v2 = []

    for item in prefer[person1]:
        v1.append(prefer[person1][item])
        if item in prefer[person2]:
            v2.append(prefer[person2][item])
        else:
            v2.append(0)   

    for item1 in prefer[person2]:
        if item1 in prefer[person1]:
            print(v1)
        else:
            v2.append(prefer[person2][item1])
            v1.append(0) 

    sq = v1-v2
    sq = sq**2
    sdis = sum(sq)
    dist = sqrt(sdis)
    return dist



### 计算pearson相关度
def sim_pearson(prefer, person1, person2):
    sim = {}
    #查找双方都评价过的项
    for item in prefer[person1]:
        if item in prefer[person2]:
            sim[item] = 1           #将相同项添加到字典sim中
    #元素个数
    n = len(sim)
    if len(sim)==0:
        return -1

    # 所有偏好之和
    sum1 = sum([prefer[person1][item] for item in sim])  
    sum2 = sum([prefer[person2][item] for item in sim])  

    #求平方和
    sum1Sq = sum( [pow(prefer[person1][item] ,2) for item in sim] )
    sum2Sq = sum( [pow(prefer[person2][item] ,2) for item in sim] )

    #求乘积之和 ∑XiYi
    sumMulti = sum([prefer[person1][item]*prefer[person2][item] for item in sim])

    num1 = sumMulti - (sum1*sum2/n)
    num2 = sqrt( (sum1Sq-pow(sum1,2)/n)*(sum2Sq-pow(sum2,2)/n))  
    if num2==0:                                                ### 如果分母为0，本处将返回0.
        return 0  

    result = num1/num2
    return result


### 获取对item评分的K个最相似用户（K默认20）
def topKMatches(prefer, person, itemId, k, sim = sim_pearson):
    userSet = []
    scores = []
    users = []
    #找出所有prefer中评价过Item的用户,存入userSet
    for user in prefer:
        if itemId in prefer[user]:
            userSet.append(user)
    #计算相似性
    scores = [(sim(prefer, person, other),other) for other in userSet if other!=person]
    #print scores

    #按相似度排序
    scores.sort()
    scores.reverse()

    if len(scores)<=k:       #如果小于k，只选择这些做推荐。
        for item in scores:
            users.append(item[1])  #提取每项的userId
        return users
    else:                   #如果>k,截取k个用户
        kscore = scores[0:k]
        for item in kscore:
            users.append(item[1])  #提取每项的userId
        #print users
        return users               #返回K个最相似用户的ID



### 计算用户对某物品的平均评分
def getAverage(prefer, userId,itemId):
    count = 0
    sum = 0.0
    for item in prefer[userId]:
        if item == itemId:
            sum += prefer[userId][item]
            count = count+1
    if count == 0:
        return 0
    else:
        return sum/count


### 平均加权策略，预测userId对itemId的评分
def getRating(prefer1, userId, itemId, knumber,similarity=sim_pearson):
    sim = 0.0
    averageOther =0.0
    jiaquanAverage = 0.0
    simSums = 0.0
    #获取K近邻用户(评过分的用户集)
    users = topKMatches(prefer1, userId, itemId, k=knumber, sim = sim_pearson)

    #获取userId 对某个物品评价的平均值
    averageOfUser = getAverage(prefer1, userId,itemId)     

    #计算每个用户的加权，预测 
    for other in users:
        sim = similarity(prefer1, userId, other)    #计算比较其他用户的相似度
        averageOther = getAverage(prefer1, other,itemId)   #该用户的平均分
        # 累加
        simSums += abs(sim)    #取绝对值
        jiaquanAverage +=  averageOther*abs(sim)   #累加，一些值为负

    # simSums为0，即该项目尚未被其他用户评分，这里的处理方法：返回用户平均分
    if simSums==0:
        return averageOfUser
    else:
        return jiaquanAverage/simSums


##==================================================================
##     getAllUserRating(): 获取所有用户的预测评分，存放到fileResult中
##
## 参数:fileTrain,fileTest 是训练文件和对应的测试文件，fileResult为结果文件
##     similarity是相似度度量方法，默认是皮尔森。
##     K为选取相邻用户个数
##==================================================================
def setAllUserRating(fileTrain, fileTest, fileResult, similarity,K):
    prefer1 = loadMovieLensTrain(fileTrain)         # 加载训练集 
    prefer2 = loadMovieLensTest(fileTest)           # 加载测试集  
    inAllnum = 0

    file = open(fileResult, 'w')    

    for userid in prefer2:             #test集中每个用户
        for item in prefer2[userid]:   #对于test集合中每一个项目用base数据集,CF预测评分
            rating = getRating(prefer1, userid, item, K)   #基于训练集预测用户评分(用户数目<=K)
            file.write('%s\t%s\t%s\n'%(userid, item, rating))
            inAllnum = inAllnum +1
    file.close()


##==================================================================
##    根据部分用户的列表数据，和训练集中的物品ID进行组合
##    为部分用户进行全部物品的偏好计算
##    同时将生成的偏好文件存储到fileResult文件中
##     similarity是相似度度量方法，默认是皮尔森。
##     K为选取相邻用户个数
##==================================================================
def setSomeUserRating(fileTrain, fileSomeUser, fileResult,K,similarity):
    prefer1 = loadMovieLensTrain(fileTrain)         # 加载训练集 
    prefer2 = loadMovieLensTestSomeUser(fileSomeUser,fileTrain)           # 加载用户数据 物品数据和偏好，其中偏好值为0 
    inAllnum = 0

    file = open(fileResult, 'w')    

    for userid in prefer2:             #用户全部训练集物品 ID       
        for item in prefer2[userid]:   #取出某用户的物品ID
            rating = getRating(prefer1, userid, item, K)   #基于训练集预测用户评分(用户数目<=K)
            file.write('%s\t%s\t%s\n'%(userid, item, rating))
            inAllnum = inAllnum +1
    file.close()



##==================================================================
##    recommendation(): 获取某些用户的前n个推荐，存放到fileResult1中
##
## 参数:fileTrain是训练文件
##     fileSomeUser 是需要推荐的用户列表，一行为一个用户
##     fileResult1为结果文件
##     n是推荐物品个数。
##==================================================================


def recommendation(fileTrain, fileSomeUser, fileResult,fileResult1,n,K,similarity):


    setSomeUserRating(fileTrain, fileSomeUser, fileResult,K,similarity)
    #从结果中取得数据     
    prefer = {}
    for line in open(fileResult,'r'):    
        (userid, movieid, rating) = line.split('\t')        
        prefer.setdefault(userid, {})    
        prefer[userid][movieid] = float(rating)

    #找出用户在训练集中已经评价过的物品ID 
    for line1 in open(fileSomeUser,'r'):
        arrUserId = str(int(line1))
        for line3 in open(fileTrain,'r'): 
            (userid1, movieid1, rating,ts) = line3.split('\t')
            if userid1 == arrUserId:#找到了，将其爱好设置为0
                prefer[userid1][movieid1] = 0.0 

    #排序取前N个物品的偏好值    
    file = open(fileResult1, 'w')  
    for userid in prefer:
        a = sorted(prefer[userid].items(), key=lambda x:x[1], reverse=True)#排序              
        if n<len(a):#每个用户要输出多少个偏好物品
            for i in range(0,n):
                file.write('%s\t%s\t%s\n'%(userid, a[i][0],a[i][1]))
        else:
            for i in range(0,len(a)):
                file.write('%s\t%s\t%s\n'%(userid, a[i][0],a[i][1]))

    file.close()


##==================================================================
## getRmseAndMae(): 根据对测试集的评测结果，进行评分预测
##
## 参数:fileTest是测试集文件，包括了用户的评分
##      fileResult1为对测试集预测的结果文件，包括了对用户预测的评分
##==================================================================

def calRmseAndMae(fileTest,fileResult):
    prefer1 = loadMovieLensTest(fileTest)         # 加载测试集 
    prefer2 = loadMovieLensResult(fileResult)           # 加载预测结果集  
    rmse = 0.0
    mae = 0.0
    for userid in prefer1:             #test集中每个用户
        for item in prefer1[userid]:   #对于test集合中每一个项目用base数据集,CF预测评分
            r1 = prefer1[userid][item]
            r2 = prefer2[userid][item]
            #print r1,r2            
            rmse += (r1-r2)**2
            mae += abs(r1-r2)    
    l = float(len(prefer1))
    return sqrt(rmse)/l,mae/l




############    主程序   ##############
if __name__ == "__main__":
    print("\n--------------基于MovieLens的推荐系统 运行中... -----------\n")

    fileTrain = './ml-100k/u1.base'#训练集
    fileTest = './ml-100k/u1.test'#测试集
    fileResult = './result.data'#根据测试集得到的预测结果集
    fileResult2 = './result2.data'#根据user.txt的用户列表，进行相关物品推荐结果集
    fileResultN = './topN'#推荐的TOPN个结果集，去除了用户以前在训练集中评价过的物品
    fileUser = './user.data'#需要推荐的用户ID列表，一行数据为一个用户ID


    similarity = sim_pearson
    for  K in range(4,6):
        #根据训练集和测试集，得到预测试结果的测结果集，和测试集结果的行数一样    
        setAllUserRating(fileTrain, fileTest, fileResult,similarity,K)    
        #根据测试集和预测结果集，计算模型精确度
        rmse,mae = calRmseAndMae(fileTest,fileResult)
        print ('%1.5f\t%1.5f'%(rmse,mae))  
        N = 10#推荐没有接触过的物品的个数，存放到fileResultN文件中
        recommendation(fileTrain, fileUser, fileResult2,fileResultN+str(K)+'.txt',N,K,similarity)

    print("-------------Completed!!-----------")