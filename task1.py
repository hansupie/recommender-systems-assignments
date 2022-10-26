# open text

import csv

with open('ratings.csv', newline='') as f:
    reader = csv.reader(f)
    data = list(reader)
    
print(data)

# print some text lines + line count

