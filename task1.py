import csv

# open ratings.csv file and read all the lines to data list

with open('ratings.csv', newline='') as f:
    reader = csv.reader(f)
    data = list(reader)

# print first 4 rows (column names + 3 first value rows)
# print list length

for x in range(4):
    print(data[x])
print(len(data))
