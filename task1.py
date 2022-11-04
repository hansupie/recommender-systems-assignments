from pandas.core.frame import DataFrame
from itertools import groupby
import pandas as pd
import numpy as np
      
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

print('MOVIELENS')
print(data_table.head(10))

# just testing with smaller data
# data_table = pd.read_csv('small_movie_ratings.csv', index_col=0)
# print(data_table)

'''   This part was used to calculate corrcoef, but since it took so long, we wrote it to
      to calculated_corr.csv and now just load the csv :D

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


#nearest 10 correlations for user '1'

# sorts list to nearest ones to user '1'
nearest = calc_data.sort_values(axis = 'index', by='1', ascending=False)

# removes all other colums except '1'
nearest = nearest[['1']]
print(nearest.head(11))


# predict movie scores

# function to select only users who have rated the movie
def get_users_who_rated(data_table: pd.DataFrame, movie: str):
  return data_table.loc[movie, :].dropna().index.values

# get n top neighbours for user
# returns a dictionary of users and their similarity to user
def get_top_neighbours(similarity_data: pd.DataFrame, user: str, users_who_rated: str, n: int):
  return similarity_data[user][users_who_rated].nlargest(n).to_dict()

# adjust for bias

# subtract user's mean rating from a rating
# returns adjusted rating
def substract_bias(rating: float, mean_rating: float):
  return rating - mean_rating

def get_rating_without_bias(data_table: pd.DataFrame, user: str, movie: str):
  mean_rating = data_table[int(user)].mean()
  rating = data_table.loc[movie, int(user)]
  return substract_bias(rating, mean_rating)

def get_ratings_of_neighbours(data_table: pd.DataFrame, neighbours: list, movie: str):
  return [
    get_rating_without_bias(data_table, neighbour, movie)
    for neighbour in neighbours
  ]

# predicting user rating for a movie
# first get weighted average of neighbours' ratings
# then get the rating by adjusting for bias

# dividing by 0 :)
def get_weighted_average_rating_of_neighbours(ratings: list, neighbour_dist: list):
  weighted_sum = np.array(ratings).dot(np.array(neighbour_dist))
  abs_neighbour_dist = np.abs(neighbour_dist)
  # if np.sum(abs_neighbour_dist) == 0:
  #   print("neighbour_dist", neighbour_dist)
  #   print(np.sum(abs_neighbour_dist))
  return weighted_sum / np.sum(abs_neighbour_dist)

def get_user_rating(data_table: pd.DataFrame, user: str, avg_neighbour_rating: float):
  user_avg_rating = data_table[int(user)].mean()
  if(avg_neighbour_rating < 0):
    return round(user_avg_rating + avg_neighbour_rating, 2)
  else:
    return round(user_avg_rating - avg_neighbour_rating, 2)
  

def predict_rating(data: pd.DataFrame, similarity_data: pd.DataFrame, user: int, movie: str, neighbours: int = 30):
  data_table = data.copy()
  rated_users = get_users_who_rated(data_table, movie)
  top_neighbours_dist = get_top_neighbours(similarity_data, str(user), rated_users, neighbours)
  neighbours, distance = top_neighbours_dist.keys(), top_neighbours_dist.values()
  ratings = get_ratings_of_neighbours(data_table, neighbours, movie)
  avg_neighbour_rating = get_weighted_average_rating_of_neighbours(ratings, list(distance))
  return get_user_rating(data_table, user, avg_neighbour_rating)

all_ratings = data_table.copy()
selected_user = 1

user_ratings = []
for movie in data_table.index:
  if np.isnan(all_ratings.loc[movie, selected_user]):
    rating = predict_rating(data_table, calc_data, selected_user, movie)
    user_ratings.append(rating)

rating_list = list(zip(data_table.index, user_ratings))

rating_df = pd.DataFrame(rating_list, columns=['Movie', 'Rating'])
print(rating_df)

movie_recs = rating_df.sort_values('Rating', ascending=False)
print(movie_recs.head(20))