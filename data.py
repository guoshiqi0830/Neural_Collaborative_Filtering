import numpy as np
import pandas as pd
import config
import torch

np.random.seed(1)

class Data():
    def __init__(self):
        self.raw = self._read_data(config.DATA_PATH)
        self.user_rating_matrix = self._get_user_rating_matrix(self.raw)
        self.train, self.test = self._leave_one(self.raw)
        self.negative_samples = self._negative_sampling(self.user_rating_matrix)
        self._curr_batch_index = 0
        print('Finished loading data')

    def _read_data(self, data_path):
        print('Reading data...')
        data = np.loadtxt(data_path,delimiter='::')
        # np.random.shuffle(data)
        return data

    def _get_user_rating_matrix(self, raw):
        m = np.zeros([config.N_USER+1, config.N_ITEM+1])
        for rec in raw:
            m[int(rec[0])][int(rec[1])] = 1
        return m
    
    def _leave_one(self, raw):
        print('Spliting dataset...')
        # get records not rated
        df = pd.DataFrame(self.user_rating_matrix)
        df = df.stack().reset_index()
        df.columns = ['user','item','rating']
        df = df.drop(df[ (df['user'] == 0) | (df['item'] == 0) | (df['rating'] == 1)].index)
        train_not_rated = np.array(df)
        # get records rated
        df = pd.DataFrame(raw,columns=['user_id','item_id','rating','timestamp'])
        df['rating'] = 1.
        test_index = df.groupby('user_id')['timestamp'].idxmax()
        test = np.array(df.iloc[test_index][['user_id','item_id','rating']])
        train = np.array(df.drop(test_index)[['user_id','item_id','rating']])
        # union train dataset
        train = np.r_[train, train_not_rated]
        # np.random.shuffle(train)
        return train, test

    def _negative_sampling(self, user_rating_matrix, k = 99):
        print('Negative sampling...')
        m = np.zeros([config.N_USER + 1,k])
        for user_id in range(1, config.N_USER+1):
            not_rated = np.argwhere(user_rating_matrix[user_id] == 0).T[0]
            np.random.shuffle(not_rated)
            m[user_id] = not_rated[:k]
        return m

    def get_one_batch(self):
        if self._curr_batch_index >= len(self.train):
          return None
        train_batch = self.train[self._curr_batch_index:self._curr_batch_index + config.BATCH_SIZE]
        user_batch = torch.tensor(train_batch[:,0]).long()
        item_batch = torch.tensor(train_batch[:,1]).long()
        rating_batch = torch.tensor(train_batch[:,2]).float()
        self._curr_batch_index += config.BATCH_SIZE
        return user_batch, item_batch, rating_batch

    def init_batcher(self):
        np.random.shuffle(self.train)
        self._curr_batch_index = 0

