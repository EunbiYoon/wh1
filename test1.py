import pandas as pd
import numpy as np
import sklearn as sk

# read CSV file
wdbc_file = pd.read_csv('wdbc.csv', header=None)

# convert DataFrame to NumPy array
wdbc_numpy = wdbc_file.to_numpy()

# min-max normalization using NumPy
# for last column, min = 0 & max =1 -> value is just itself
normalized_numpy = (wdbc_numpy - np.min(wdbc_numpy, axis=0)) / (np.max(wdbc_numpy, axis=0) - np.min(wdbc_numpy, axis=0))

# change normalized data to pandas dataframe
normalized_data = pd.DataFrame(normalized_numpy, columns=wdbc_file.columns)
normalized_data.to_excel("normalized_df.xlsx")

# shuffle the DataFrame
shuffled_data = sk.utils.shuffle(normalized_data)

# split data with training and testing
train_data, test_data= sk.model_selection.train_test_split(shuffled_data, test_size=0.2)
