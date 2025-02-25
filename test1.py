import pandas as pd
import numpy as np
import openpyxl

# read CSV file
wdbc_file = pd.read_csv('wdbc.csv', header=None)
wdbc_file.to_excel("original.xlsx")

# Convert DataFrame to NumPy array
wdbc_numpy = wdbc_file.to_numpy()

# Min-max normalization using NumPy
normalized_numpy = (wdbc_numpy - np.min(wdbc_numpy, axis=0)) / (np.max(wdbc_numpy, axis=0) - np.min(wdbc_numpy, axis=0))

# change normalized data to pandas dataframe
normalized_df = pd.DataFrame(normalized_numpy, columns=wdbc_file.columns)

# end of column is follow with original one
start_data = pd.concat([normalized_df, pd.DataFrame([wdbc_file.iloc[-1]], columns=wdbc_file.columns)], ignore_index=True)

start_data.to_excel("new.xlsx")