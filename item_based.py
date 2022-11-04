from pandas.core.frame import DataFrame
from itertools import groupby
from sklearn.neighbors import NearestNeighbors
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from scipy import sparse

# open ratings.csv file and read all the lines to data list
r_data=pd.read_csv('ratings.csv',sep=',',header='infer',quotechar='\"')

# read movie data from movies.csv
m_data = pd.read_csv("movies.csv", sep=",", header='infer',quotechar='\"')
m_data = m_data.iloc[:,0:2]

data = r_data.merge(m_data,on="movieId")
data.drop(['timestamp'],inplace=True,axis=1)

data_table = pd.pivot_table(data,values='rating',columns='userId',index='title')

# item-based collaborative filtering using cosine similarity

# fill nan values with zeroes
nonan = data_table.fillna(0)

# get 30 nearest neighbours with NearestNeighbors cosine similarity
# n_neighbours = 30
# nbrs = NearestNeighbors(metric='cosine', algorithm='brute').fit(nonan.values)
# distances, indices = nbrs.kneighbors(nonan.values)

# print(indices)
# print(distances)

nonan_sparse = sparse.csr_matrix(nonan.values)
similarity_matrix = cosine_similarity(data_table.fillna(0))
print("similary matrix")
print(similarity_matrix)

similarity_sparse = cosine_similarity(nonan_sparse,dense_output=False)
print('pairwise sparse output:\n {}\n'.format(similarity_sparse))

# movies user 1 has watched
user = 1
movie = "(500) Days of Summer (2009)"
user_watched = pd.DataFrame(data_table[user].dropna(axis=0, how='all')\
                          .sort_values(ascending=False))\
                          .reset_index()\
                          .rename(columns={1:'rating'})
print(user_watched)

