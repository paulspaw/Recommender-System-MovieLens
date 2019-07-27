import pandas as pd


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
        pd.read_csv(genreFile,
                    header=None,
                    sep='|',
                    names=["type", "tid"]))
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
                    names=[
                        "uid", "gender", 
                        "occupation", "zipcode"
                    ]))
    return df


def recommend():
    return


def main():
    recommend()


if __name__ == "__main__":
    df_user = getUserInfo("./ml-100k/u.user")
    df1,df2 = getItemInfo("./ml-100k/u.genre", "./ml-100k/u.item")
    df_rank = getDataInfo("./ml-100k/u.data")
