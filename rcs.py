import pandas as pd
from math import *


def getDataInfo(fileName):
    # u.data: user id, movie id, rating, commend time stamp
    df = pd.DataFrame(
        pd.read_csv(fileName,
                    header=None,
                    sep='\s',
                    names=["uid", "mid", "rating", "tstamp"],
                    engine='python'))
    return df


def getItemInfo(genreFile, itemFile):
    # u.genre: movie type name, movie type id
    # u.item: movie id, movie name, movie released date, movie type id
    df_genre = pd.DataFrame(
        pd.read_csv(genreFile, header=None, sep='|', names=["type", "tid"]))
    df_item1 = pd.DataFrame(
        pd.read_csv(itemFile,
                    header=None,
                    sep='|',
                    encoding="ISO-8859-1",
                    usecols=[0, 1, 2],
                    names=["mid", "mname", "date"]))
    df_item2 = pd.DataFrame(
        pd.read_csv(itemFile, header=None, sep='|',
                    encoding="ISO-8859-1")).iloc[:, 5:24]
    df_item = pd.concat([df_item1, df_item2], axis=1)
    df_item = df_item.rename(
        columns={
            5: 'unknown',
            6: 'Action',
            7: 'Adventure',
            8: 'Animation',
            9: 'Children\'s\'',
            10: 'Comedy',
            11: 'Crime',
            12: 'Documentary',
            13: 'Drama',
            14: 'Fantasy',
            15: 'Film-Noir',
            16: 'Horror',
            17: 'Musical',
            18: 'Mystery',
            19: 'Romance',
            20: 'Sci-Fi',
            21: 'Thriller',
            22: 'War',
            23: 'Western'
        })
    return df_genre, df_item


def getUserInfo(fileName):
    # u.user: user id, user gender, user occupation, user zipcode
    df = pd.DataFrame(
        pd.read_csv(fileName,
                    header=None,
                    sep='|',
                    names=["uid", "age", "gender", "occupation", "zipcode"]))
    return df


def classifyMovie(fileName):
    # dict = {uid: {mid: (rating, mname) } }
    movieDict = {}
    for index, row in fileName.iterrows():
        #if not exit uid
        if not row["uid"] in movieDict.keys():
            movieDict[row["uid"]] = {row["mid"]: (row["rating"], row["mname"])}
        else:
            movieDict[row["uid"]][row["mid"]] = (row["rating"], row["mname"])
    return movieDict
    # 用csv文件的方式遍历所有行
    # with open (fileName, 'r', encoding='UTF-8') as file:
    #     for line in fileName.readlines():
    #         line = line.strip().split(',')
    #         #if not exit uid
    #         if not line[0] in movieDict.keys():
    #             movieDict[line[0]] = {line[1]:(line[2],line[4])}
    #         else:
    #             movieDict[line[0]][line[1]] = (line[2],line[4])
    # return movieDict

''' *******************************************
    User-based collaborative filtering (below)*
    *******************************************
'''
# euclidean algorithm - compare similarity between two users 
def euclidean(user1,user2,movieDict):
    #pull out two users from movieDict
    user1_data=movieDict[user1]
    user2_data=movieDict[user2]
    distance = 0
    #cal euclidean distance
    for key in user1_data.keys():
        if key in user2_data.keys():
            # the smaller, the more simularity
            distance += pow(float(user1_data[key][0])-float(user2_data[key][0]),2)
 
    return 1/(1+sqrt(distance))
 
#计算某个用户与其他用户的相似度
def top10_simliar(userID,movieDict):
    res = []
    for uid in movieDict.keys():
        #排除与自己计算相似度
        if not uid == userID:
            simliarty = euclidean(userID,uid,movieDict)
            res.append((uid,simliarty))
    res.sort(key=lambda val:val[1])
    return res[:10]


''' *******************************************
    User-based collaborative filtering (above)*
    *******************************************
'''

def recommend():
    return


def main():
    recommend()


if __name__ == "__main__":
    # movieDict = {}
    df_user = getUserInfo("./ml-100k/u.user")
    df_genre, df_item = getItemInfo("./ml-100k/u.genre", "./ml-100k/u.item")
    df_rank = getDataInfo("./ml-100k/u.data")

    # trans each dataframe to csv
    # df_user.to_csv("user.csv",index=False,sep=',')
    # df_genre.to_csv("genre.csv",index=False,sep=',')
    # df_item.to_csv("item.csv",index=False,sep=',')
    # df_rank.to_csv("rank.csv",index=False,sep=',')

    # merge rating file and item file
    data = pd.merge(df_rank, df_item, on='mid').sort_values('uid')
    # data.to_csv("./csv/movie.csv",index=False,sep=',')
    movieDict = classifyMovie(data)
    # print(movieDict["1"])
    RES = top10_simliar(1,movieDict)
    print(RES)