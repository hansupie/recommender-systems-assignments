from pandas.core.frame import DataFrame
from itertools import groupby
import pandas as pd
import numpy as np

'''   This part was used to calculate corrcoef, but since it took so long, we wrote it to
      to calculated_corr.csv and now just load the csv :D
      
# open ratings.csv file and read all the lines to data list
r_data=pd.read_csv('ratings.csv',sep=',',header='infer',quotechar='\"')

# print first  rows
get_rows = r_data.head(3)
print(get_rows)

# print list length
print(r_data.count())

# calculate similarities with Pearson correlation

# read movie data from movies.csv
m_data = pd.read_csv("movies.csv", sep=",", header='infer',quotechar='\"')
m_data = m_data.iloc[:,0:2]

data = r_data.merge(m_data,on="movieId")
data.drop(['timestamp'],inplace=True,axis=1)

data_table = pd.pivot_table(data,values='rating',columns='userId',index='title')
#print('MOVIELENS')
# print(data_table.head(10))

# just testing with smaller data
#data_table = pd.read_csv('small_movie_ratings.csv', index_col=0)
#print(data_table)

# function to check if all values are equal
def all_equal(iterable):
  g = groupby(iterable)
  return next(g, True) and not next(g, False)

# function that finds the correlation between two users
def find_corr(data_table: pd.DataFrame, user1: str, user2: str):
  rated_by_both = data_table[[user1, user2]].dropna(axis=0).values

  ## at least 3 common ratings to compare
  min_degrees_of_freedom = 3
  if len(rated_by_both) < min_degrees_of_freedom:
    return -1
  
  user1_ratings = rated_by_both[:,0]
  user2_ratings = rated_by_both[:,1]

  # pearson correlatin returns nan if there is no variance in ratings
  if all_equal(user1_ratings) or all_equal(user2_ratings):
    return -1

  return np.corrcoef(user1_ratings, user2_ratings)[0,1]

users = list(data_table.columns)
movies = list(data_table.index)
similarity_matrix = np.array([[find_corr(data_table, user1, user2) for user1 in users] for user2 in users])
similarity_data = pd.DataFrame(similarity_matrix, columns=users, index=users)
print(similarity_data)

similarity_data.to_csv('calculated_corr.csv', index=False)
'''

# reading calculated data to dataframe
calc_data=pd.read_csv('calculated_corr.csv',sep=',',header='infer',quotechar='\"')
calc_data.index = np.arange(1, len(calc_data) + 1)
print(calc_data)


# nearest 10 correlations for user '1'

# sorts list to nearest ones to user '1'
nearest = calc_data.sort_values(axis = 'index', by='1', ascending=False)

# removes all other colums except '1'
nearest = nearest[['1']]
print(nearest.head(11))