from pandas.core.frame import DataFrame
from itertools import groupby
import pandas as pd
import numpy as np
from numpy import mean
import itertools

#
# User-based collaborative filtering for individual users.
# (Group recommendation methods begin on line 92)
#
      
# open ratings.csv file and read all the lines to data list
r_data=pd.read_csv('ratings.csv',sep=',',header='infer',quotechar='\"')

# calculate similarities with Pearson correlation

# read movie data from movies.csv
m_data = pd.read_csv("movies.csv", sep=",", header='infer',quotechar='\"')
m_data = m_data.iloc[:,0:2]

data = r_data.merge(m_data,on="movieId")
data.drop(['timestamp'],inplace=True,axis=1)

data_table = pd.pivot_table(data,values='rating',columns='userId',index='title')

# reading calculated data to dataframe
calc_data=pd.read_csv('calculated_corr.csv',sep=',',header='infer',quotechar='\"')
calc_data.index = np.arange(1, len(calc_data) + 1)
print(calc_data)

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

#
# A. Average aggregation method
#

users = [1, 2, 3]
group_user_ratings = []

for movie in data_table.index:
  movie_ratings = []
  for user in users:
    if np.isnan(all_ratings.loc[movie, user]):
      rating = predict_rating(data_table, calc_data, user, movie)
      movie_ratings.append(rating)
  # only add movie if a prediction exists for all user
  if len(movie_ratings) == len(users):
    # get average of movie_ratings
    avg = mean(movie_ratings)
    group_user_ratings.append((movie, avg))

group_user_ratings.sort(key=lambda a: a[1], reverse=True)

print("Part A: Average aggregation method")
print(*group_user_ratings[:20], sep="\n")

#
# A. Least misery method
#



#
# B. Method that takes disagreements into account
#

def calculate_distance(rating1: float, rating2: float):
  return abs(rating1 - rating2)

users = [1, 2, 3]
group_user_ratings = []
all_movie_prediction = []
ratings_method_b = []

for movie in data_table.index:
  movie_ratings = []
  for user in users:
    if np.isnan(all_ratings.loc[movie, user]):
      rating = predict_rating(data_table, calc_data, user, movie)
      movie_ratings.append(rating)
  # only add movie if a prediction exists for all user
  if len(movie_ratings) == len(users):
    # get average of movie_ratings
    avg = mean(movie_ratings)
    group_user_ratings.append((movie, avg, movie_ratings))
    # calculate distance for each pair of ratings
    distances = []
    for pair in itertools.combinations(movie_ratings, r=2):
      distances.append(calculate_distance(*pair))
    # print(distances, sep="\n")
    if max(distances) < 2:
      ratings_method_b.append((movie, avg, movie_ratings))

group_user_ratings.sort(key=lambda a: a[1], reverse=True)
ratings_method_b.sort(key=lambda a: a[1], reverse=True)

print("Part B")
print(*ratings_method_b[:20], sep="\n")