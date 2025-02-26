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

# get class from each attribute
def attribute_class(car_data):
    # reset column name as number
    car_data.columns=list(range(len(car_data.columns)))

    # check class in each attribute
    unique_attributes=pd.DataFrame()
    for i in range(len(car_data.columns)):
        remove_duplicate=car_data[i].drop_duplicates().reset_index(drop=True)
        unique_attributes[i]=remove_duplicate

# calculate entropy
def entropy_formula():
    return 3

# make entropy_matrix using entropy_foumula
def entropy_matrix():
    return 3

# check majority
def majority_formula(list):
    return 3

# Accuracy in Training and Testing
def calculate_accuracy(actual_list, predicted_list):
    # compare two column. matched->1, mismatched->0
    compared_result = list((actual_list == predicted_list).astype(int))
    
    # calculate accuracy
    accuracy_list=(sum(compared_result)/len(compared_result))   

    # return accuracy value
    return accuracy_list

# make decision tree
def decision_tree():
    return 4

# draw graph
def draw_graph():
    return 4

# main function - united all function above
def main():
    process_dataset()

    

# ensures that the main function is executed only.
if __name__ == "__main__":
    main()
