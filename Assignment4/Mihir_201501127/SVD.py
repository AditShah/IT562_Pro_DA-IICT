import pandas as pd
import numpy as np

import sklearn.preprocessing as sk
from sklearn.decomposition import TruncatedSVD
from sklearn.utils.extmath import randomized_svd

ratings = pd.read_csv('data.csv', dtype={'rating': float})
ratings.loc[:,'rating'] = sk.minmax_scale(ratings.loc[:,'rating'] )
print(ratings.loc[:,'rating'])
print (ratings)
print (ratings.head())


Ratings = ratings.pivot(index = 'user_id', columns ='song_id', values = 'rating').fillna(0)
print(Ratings.head())

R = Ratings.as_matrix()
user_ratings_mean = np.mean(R, axis = 1)
#print(R.size)

R_demeaned = R - user_ratings_mean.reshape(-1, 1)
U, Sigma, VT = randomized_svd(R,n_components=2)

svd = TruncatedSVD(n_components=20, n_iter=7)
svd.fit(R)
print("\n\n\n\nU\n\n")
print(U)
print("\n\n\n\nSigma\n\n")
print(Sigma)
print("\n\n\n\nVT\n\n")
print(VT)


print(svd.explained_variance_ratio_)  

print(svd.components_)  

print(svd.singular_values_) 