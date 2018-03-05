import numpy as np
import pandas as pd

import sklearn.preprocessing as sk
import gensim
import gensim.models.lsimodel as ls

from sklearn.decomposition import TruncatedSVD
from sklearn.random_projection import sparse_random_matrix
from sklearn.utils.extmath import randomized_svd

ratings = pd.read_csv('data.csv', dtype={'rating': float})

ratings.loc[:,'rating'] = sk.minmax_scale(ratings.loc[:,'rating'] )
print(ratings.loc[:,'rating'])
print (ratings)
print (ratings.head())


Ratings = ratings.pivot(index = 'user_id', columns ='song_id', values = 'rating').fillna(0).to_sparse(fill_value=0)
print(Ratings.head())

R = Ratings.as_matrix()

Z=gensim.matutils.Dense2Corpus(R, documents_columns=True)
print(Z)

lsi=ls.LsiModel(Z, num_topics=3)
print("\n\n\n\nSigma\n\n")

print(lsi.projection.s)
print("\n\n\n\nU\n\n")

print(lsi.projection.u)
print("\n\n\n\nVT\n\n")
V = gensim.matutils.corpus2dense(lsi[Z], len(lsi.projection.s)).T / lsi.projection.s
print(V)
