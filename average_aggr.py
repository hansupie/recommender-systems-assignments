from task1 import *
from numpy import mean

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

print(*group_user_ratings[:20], sep="\n")

