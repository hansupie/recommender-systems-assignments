from pandas.core.frame import DataFrame
import pandas as pd

# open ratings.csv file and read all the lines to data list
data=pandas.read_csv('ratings.csv',sep=',',header='infer',quotechar='\"')

# print first  rows
get_rows = data.head(3)
print(get_rows)

# print list length
print(data.count())