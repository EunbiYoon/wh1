import pandas as pd
import numpy as np
import sklearn as sk
import matplotlib

# Prepared data before KNN
def process_dataset():
    # read CSV file
    wdbc_file = pd.read_csv('wdbc.csv', header=None)

    # convert DataFrame to NumPy array
    wdbc_numpy = wdbc_file.to_numpy()

    # min-max normalization using NumPy = (x-min)/(max-min)
    # for last column, min = 0 & max =1 -> value is just itself
    normalized_numpy = (wdbc_numpy - np.min(wdbc_numpy, axis=0)) / (np.max(wdbc_numpy, axis=0) - np.min(wdbc_numpy, axis=0))

    # change normalized data to pandas dataframe
    normalized_data = pd.DataFrame(normalized_numpy, columns=wdbc_file.columns)

    # shuffle the DataFrame
    ##### 나중에 지우기 랜덤하게 하지 않게 하기 위해서 고정함
    shuffled_data = sk.utils.shuffle(normalized_data, random_state=42)

    # split data with training and testing
    train_data, test_data= sk.model_selection.train_test_split(shuffled_data, test_size=0.2, random_state=42)

    # reset index both train_data and test_data
    train_data=train_data.reset_index(drop=True)
    test_data=test_data.reset_index(drop=True)
    
    # save result as excel
    train_data.to_excel('train_data.xlsx')
    test_data.to_excel('test_data.xlsx')

    # return final train_data, test_data
    return train_data, test_data

# caculate Euclidean Distance
def euclidean_formula(x1,x2):
    euclidean_distance = np.sqrt(np.sum(x1-x2)**2)
    return euclidean_distance

# check majority
def majority_formula(list):
    # count 1 or 0, if there is nothing value is 0
    count_1 = list.value_counts().get(1, 0)  
    count_0 = list.value_counts().get(0, 0) 
    #betwen 1 and 0 which one is more
    if count_1 > count_0:
        return 1
    else:
        return 0

# Calculate Euclidean Distane in train_data
def euclidean_matrix(train_data):
    train_euclidean=np.zeros((len(train_data),len(train_data)))
    for i in range(len(train_data)):
        for j in range(len(train_data)):
            train_euclidean[i,j]=euclidean_formula(train_data.iloc[i], train_data.iloc[j])
    train_euclidean=pd.DataFrame(train_euclidean)
    train_euclidean.to_excel("train_eculidean.xlsx")
    return train_euclidean

# Insert predicted list in the original dataset at the last column
def insert_predicted(predicted_list, train_data):
    train_data.loc[-1]=predicted_list
    return train_data

# KNN algorithm using Euclidian Matrix
def knn_algorithm(k, train_euclidean, train_data):
    # Iterate over k values : 1~51 odd number
    for i in range(1,k+1,2): 
        # j is for data instances 
        for j in range(len(train_euclidean)):
            # find smallest components count:k for xj
            k_smallest_column=train_euclidean.loc[j].nsmallest(k)

            # get the index list from smallest_column
            smallest_index_list=k_smallest_column.index.tolist()

            # depends on k, cut off list
            smallest_index_cutoff=smallest_index_list[:i+1]

            # get train_data which has index as smallest_index_list
            train_smallest=train_data.loc[smallest_index_cutoff]

            # get the last column from smallest_index_cutoff
            train_last=train_smallest[30]

            # check majority 
            final_predicted_value=majority_formula(train_last)

            # merge final_predicted_value to train_data
            train_data.at[j,"k="+str(i)]=final_predicted_value
            train_data.to_excel("merge_k.xlsx")

            print(f"make knn algorithm : k={i} , train_instance={j}")

    #Save the DataFrame once after all iterations
    train_data.to_excel("merge_k.xlsx")
    print("Saved to merge_k.xlsx")

# Accuracy in Training and Testing
def accuracy_matrix():
    return 0

# Graph created with k
def graph_knn():
    return 0

# main function - united all function ahead
def main():
    train_data, test_data=process_dataset()
    train_euclidean=euclidean_matrix(train_data)
    knn_algorithm(51, train_euclidean, train_data)


# ensures that the main function is executed only
if __name__ == "__main__":
    main()