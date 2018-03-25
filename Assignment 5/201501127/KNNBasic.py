import numpy as np
import surprise  # run 'pip install scikit-surprise' to install surprise
import os
import time
from guppy import hpy
from surprise import BaselineOnly
from surprise import Dataset
from surprise import Reader
from surprise.model_selection import cross_validate
import pandas as pd

class MatrixFacto(surprise.AlgoBase):
    '''A basic rating prediction algorithm based on matrix factorization.'''
    
    skip_train = 0
    
    def __init__(self, learning_rate, n_epochs, n_factors):
        
        self.lr = learning_rate  # learning rate for SGD
        self.n_epochs = n_epochs  # number of iterations of SGD
        self.n_factors = n_factors  # number of factors
        
    def train(self, trainset):
        '''Learn the vectors p_u and q_i with SGD'''
        
        print('Fitting data with SGD...')
        
        # Randomly initialize the user and item factors.
        p = np.random.normal(0, .1, (trainset.n_users, self.n_factors))
        q = np.random.normal(0, .1, (trainset.n_items, self.n_factors))
        
        # SGD procedure
        for _ in range(self.n_epochs):
            for u, i, r_ui in trainset.all_ratings():
                err = r_ui - np.dot(p[u], q[i])
                # Update vectors p_u and q_i
                p[u] += self.lr * err * q[i]
                q[i] += self.lr * err * p[u]
                # Note: in the update of q_i, we should actually use the previous (non-updated) value of p_u.
                # In practice it makes almost no difference.
        
        self.p, self.q = p, q
        self.trainset = trainset

    def estimate(self, u, i):
        '''Return the estmimated rating of user u for item i.'''
        
        # return scalar product between p_u and q_i if user and item are known,
        # else return the average of all ratings
        if self.trainset.knows_user(u) and self.trainset.knows_item(i):
            return np.dot(self.p[u], self.q[i])
        else:
            return self.trainset.global_mean

start = time.time()

df1 = pd.read_csv('./data100.csv', dtype={'rating': float})
reader = Reader(rating_scale=(1, 5))
data = Dataset.load_from_df(df1[['user_id','song_id','rating']], reader)
data.split(2)  # split data for 2-folds cross validation

algo = surprise.KNNBasic()
surprise.evaluate(algo, data, measures=['RMSE'])
end = time.time()
print("t1",end - start)

mem = hpy()
print("memory",mem.heap())
######################################################################3
start = time.time()
df2 = pd.read_csv('./data1000.csv', dtype={'rating': float})
reader = Reader(rating_scale=(1, 5))
data = Dataset.load_from_df(df1[['user_id','song_id','rating']], reader)
data.split(2)  # split data for 2-folds cross validation

algo = surprise.KNNBasic()
surprise.evaluate(algo, data, measures=['RMSE'])
end = time.time()
print("t2",end - start)
mem = hpy()
print("memory",mem.heap())
#################################################################################
start = time.time()
df3 = pd.read_csv('./data10000.csv', dtype={'rating': float})
reader = Reader(rating_scale=(1, 5))
data = Dataset.load_from_df(df1[['user_id','song_id','rating']], reader)
data.split(2)  # split data for 2-folds cross validation

algo = surprise.KNNBasic()
surprise.evaluate(algo, data, measures=['RMSE'])
end = time.time()
print("t3",end - start)
mem = hpy()
print("memory",mem.heap())
############################################################################################
start = time.time()
df4 = pd.read_csv('./data100000.csv', dtype={'rating': float})
reader = Reader(rating_scale=(1, 5))
data = Dataset.load_from_df(df1[['user_id','song_id','rating']], reader)
data.split(2)  # split data for 2-folds cross validation

algo = surprise.KNNBasic()
surprise.evaluate(algo, data, measures=['RMSE'])
end = time.time()
print("t4",end - start)
mem = hpy()
print("memory",mem.heap())
######################################################################################


