from csv import DictReader

# open ratings.csv file and read all the lines to data list

with open("ratings.csv", 'r') as f:
    dict_reader = DictReader(f)
    list_of_dict = list(dict_reader)

# print first 4 rows (column names + 3 first value rows)
# print list length

for x in range(4):
    print(list_of_dict[x])
print(len(list_of_dict))

