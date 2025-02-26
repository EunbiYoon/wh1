import pandas as pd
import numpy as np
import sklearn as sk


# Prepared train_data, test_data
def process_dataset():
    # read csv file
    car_data = pd.read_csv('car.csv')

    # shuffle the DataFrame
    ##### 나중에 지우기 랜덤하게 하지 않게 하기 위해서 고정함 random_state=42
    shuffled_data = sk.utils.shuffle(car_data, random_state=42)

    # split data with training and testing
    ##### 나중에 지우기 랜덤하게 하지 않게 하기 위해서 고정함 random_state=42
    train_data, test_data= sk.model_selection.train_test_split(shuffled_data, test_size=0.2, random_state=42)

    # reset index both train_data and test_data
    train_data=train_data.reset_index(drop=True)
    test_data=test_data.reset_index(drop=True)

    print(train_data)
    print(test_data)
    
    # message
    print("Shuffling are done. Split train_data and test_data. Please wait...")

    # return final train_data, test_data
    return train_data, test_data

def attribute_class(car_data):
    # reset column name as number
    car_data.columns=list(range(len(car_data.columns)))

    # check class in each attribute
    unique_attributes=pd.DataFrame()
    for i in range(len(car_data.columns)):
        remove_duplicate=car_data[i].drop_duplicates().reset_index(drop=True)
        unique_attributes[i]=remove_duplicate

# calculate entropy
# make decision tree
def main():
    process_dataset()

    

# ensures that the main function is executed only.
if __name__ == "__main__":
    main()
