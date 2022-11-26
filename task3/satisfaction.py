import pandas as pd

# groupListSat
def calc_group_list_sat(user_scores: pd.Series, group_scores: list):
  sum = 0
  for score in group_scores:
    if score[0] in user_scores:
      sum = sum + score[1]
  return sum

def calc_user_list_sat(user_scores: pd.Series):
  return user_scores.sum()

def calc_satisfaction(scores: pd.DataFrame, group_scores: list, user: int):
  user_scores = scores.iloc[user]
  group_list_sat = calc_group_list_sat(user_scores[:20], group_scores[:20])
  user_list_sat = calc_user_list_sat(user_scores[:20])

  return group_list_sat / user_list_sat