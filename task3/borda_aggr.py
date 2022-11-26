import pandas as pd

def count_borda_scores(data: pd.DataFrame, ratings: pd.DataFrame, user: int, movies: list):

  predictions = []
  for movie in movies:
    predictions.append((movie, ratings.loc[movie, str(user)]))

  predictions.sort(key=lambda a: a[1],reverse=True)

  # print(*predictions[:20], sep="\n")
  
  max_score = len(predictions)
  borda_scores = {}

  for i, score in enumerate(predictions):
    borda = max_score - i
    borda_scores[score[0]] = borda

  # print("BORDA")
  # print(*borda_scores[:20], sep="\n")

  return borda_scores

def borda_aggregation(movies: list, all_bordas: pd.DataFrame, users: list):
  scores = []
  # only works for 3 users now
  for movie in movies:
    group_score = all_bordas.loc[str(users[0]), movie] + all_bordas.loc[str(users[1]), movie] + all_bordas.loc[str(users[2]), movie]
    scores.append((movie, group_score))

  return scores
