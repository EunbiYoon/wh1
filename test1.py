import pandas as pd
import numpy as np
import csv


# open & read csv file
wdbc_file = csv.reader(open('wdbc.csv', mode='r', newline=''))

# first column 
for row in wdbc_file:
    print(row[0])

# normalize the dataset 
# normalize = (x-mean)/std
for i in range(len(wdbc_file.columns)):
    data=np.array(wdbc_file.iloc[i])
    print("data")
    print(data)
    print("/")
    mean=np.mean(data)
    std=np.std(data)
    normalized_datas=(data-mean)/std
    print(normalized_datas)