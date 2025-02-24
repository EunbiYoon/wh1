import pandas as pd
import numpy as np

wdbc_file=pd.read_csv("wdbc.csv")

print(wdbc_file.iloc[0])
# normalize the dataset 
# normalize = (x-mean)/std
count=0
for i in range(len(wdbc_file.columns)):
    data=np.array(wdbc_file.iloc[i])
    print("data")
    print(data)
    print("/")
    mean=np.mean(data)
    std=np.std(data)
    normalized_datas=(data-mean)/std
    print(normalized_datas)
    count+=1
    print(count)