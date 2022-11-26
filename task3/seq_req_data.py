import pandas as pd
import numpy as np
# from collaborative_filtering import get_similarity_data
# from collaborative_filtering import predict_rating
from borda_aggr import count_borda_scores
from borda_aggr import borda_aggregation
from satisfaction import calc_satisfaction

# open ratings.csv file and read all the lines to data list
data = pd.read_csv('ratings.csv',sep=',',header='infer',quotechar='\"')
m_data = pd.read_csv("movies.csv", sep=",", header='infer',quotechar='\"')
m_data = m_data.iloc[:,0:2]
merged = data.merge(m_data,on="movieId")
sorted_data = merged.sort_values(by=['timestamp'])
sorted_data.drop(['timestamp'],inplace=True,axis=1)
split_data = np.array_split(sorted_data, 5)

data_table1 = pd.pivot_table(split_data[0],values='rating',index="title",columns='userId')

# similarity_data = get_similarity_data(data_table1)
# print(similarity_data)
# similarity_data.to_csv('similarity_df.csv', index_label=False)

# similarity_data = pd.read_csv('similarity_df.csv')
# print(similarity_data)

# all_ratings = data_table1.copy()
selected_users = [6, 19, 604]
# group_ratings = []

# for movie in data_table1.index:
#   movie_ratings = []
#   for user in selected_users:
#     if np.isnan(all_ratings.loc[movie, user]):
#       rating = predict_rating(data_table1, similarity_data, user, movie)
#       movie_ratings.append(rating)
#       all_ratings[user][movie] = rating

# all_ratings.to_csv('all_ratings.csv', index_label=False)

# read predictions from file
all_ratings = pd.read_csv('all_ratings.csv')
borda_scores = []
unwatched_movies = []

for movie in data_table1.index:
  unwatched = True
  for user in selected_users:
    if not(np.isnan(data_table1.loc[movie, user])):
      unwatched = False
  if (unwatched):
    unwatched_movies.append(movie)

all_bordas = []
# Calculate Borda scores for users
for user in selected_users:
  borda_scores = count_borda_scores(data_table1, all_ratings, user, unwatched_movies)
  all_bordas.append(borda_scores)
  
# Create DataFrame  
scores_df = pd.DataFrame(all_bordas,index=['6', '19', '604'])  

group_scores = borda_aggregation(unwatched_movies, scores_df, selected_users)
group_scores.sort(key=lambda a:a[1],reverse=True)
print("\nGroup Borda Scores:")
print(*group_scores[:20], sep="\n")

# calculate satisfaction for users
user_sats = []
for i, user in enumerate(selected_users):
  user_sat = calc_satisfaction(scores_df, group_scores, i)
  user_sats.append((user, user_sat))

print("\nUser satisfaction scores:")
print(*user_sats, sep="\n")