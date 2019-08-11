'''
@Description: 
@Author: Peng LIU
@Date: 2019-08-11 10:16:40
@LastEditors: Peng LIU
@LastEditTime: 2019-08-11 11:50:11
'''
# recommend_list 必须要在recommend函数里面另外返回一个like rec_list=[3,44,324,623,....]
class Evaluation:
    def __init__(self,train,test,N,K,recommendList):
        self.train = train
        self.test = test
        self.N = N
        self.K = K
        self.recommend_list = recommendList
    def Recall(self):
        hit = 0
        all = 0
        for user in self.train.keys():
            if user not in self.test.keys():
                continue
            tu = self.test[user]
            for item in self.recommend_list:
                if item in tu:
                    hit += 1
            all += len(tu)
        return hit/(all*1.0)

    def Precision(self):
        hit = 0
        all = 0
        for user in self.train.keys():
            # if user not in self.test.keys():
            #     continue
            tu = self.test.get(user,{})
    #         recommend_list = get_recommendation(N, user, K, W, train)
            for item in self.recommend_list:
                if item in tu:
                    hit += 1
                all += self.N
        return hit/(all*1.0)


    def All_item(self):
        all_items = set()
        for user in self.train.keys():
            for item in self.train[user]:
                all_items.add(item)
        return all_items
    
    def Coverage(self,all_items):
        recommend_items = set()
        for user in self.train.keys():
    #         recommend_list = get_recommendation(N, user, K, W, train)
            for item in self.recommend_list:
                recommend_items.add(item)
        return len(recommend_items)/(len(all_items)*1.0)

    def Item_popularity(self):
        item_popularity = {}
        for user, items in self.train.items():
            for item in items:
                if item not in item_popularity.keys():
                    item_popularity[item] = 0
                item_popularity[item] += 1
        return item_popularity
    
    def Popularity(self,item_popularity):  
        ret = 0
        n = 0
        for user in self.train.keys():
    #         recommend_list = get_recommendation(N, user, K, W, train)
            for item in self.recommend_list:
                ret += np.log(1+item_popularity[item])
                n += 1
        return ret/(n*1.0)