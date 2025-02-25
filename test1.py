import pandas as pd
import numpy as np
import sklearn as sk
import matplotlib.pyplot as plt


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
    ##### 나중에 지우기 랜덤하게 하지 않게 하기 위해서 고정함 random_state=42
    shuffled_data = sk.utils.shuffle(normalized_data)

    # split data with training and testing
    ##### 나중에 지우기 랜덤하게 하지 않게 하기 위해서 고정함 random_state=42
    train_data, test_data= sk.model_selection.train_test_split(shuffled_data, test_size=0.2)

    # reset index both train_data and test_data
    train_data=train_data.reset_index(drop=True)
    test_data=test_data.reset_index(drop=True)
    
    # message
    print("Normalized and Shuffling are done. Split train_data and test_data. Please wait...")

    # return final train_data, test_data
    return train_data, test_data

# caculate Euclidean Distance
def euclidean_formula(x1,x2):
    # follow eucliean distance formula
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
def euclidean_matrix(train_data, test_data):
    # make numpy matrix
    train_euclidean=np.zeros((len(train_data),len(test_data)))

    # apply euclidean formula
    for i in range(len(train_data)):
        for j in range(len(test_data)):
            train_euclidean[i,j]=euclidean_formula(train_data.iloc[i], test_data.iloc[j])
    train_euclidean=pd.DataFrame(train_euclidean)

    # message
    print("Euclidean matrix has been created")
    return train_euclidean


# Accuracy in Training and Testing
def calculate_accuracy(actual_list, predicted_list):
    # compare two column. matched->1, mismatched->0
    compared_result = list((actual_list == predicted_list).astype(int))
    
    # calculate accuracy
    accuracy=(sum(compared_result)/len(compared_result))   

    # return accuracy value
    return accuracy

                                        
# KNN algorithm using Euclidian Matrix
def knn_algorithm(k, train_euclidean, data, data_info, try_count):
    # make empty dataframe
    knn_data=pd.DataFrame(data)
    knn_accuracy=pd.DataFrame()

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
            train_smallest=data.loc[smallest_index_cutoff]

            # get the last column from smallest_index_cutoff
            train_last=train_smallest[len(data.columns)-1]

            # check majority 
            final_predicted_value=majority_formula(train_last)

            # merge final_predicted_value to train_data
            knn_data.at[j,"k="+str(i)]=final_predicted_value

            # message
            print(f"knn algorithm : data_type={data_info} , try={try_count+1} , k={i} , instance_number={j}")
        
        # calcualte accuracy depending on k and make accuracy_list
        accuracy_value=calculate_accuracy(knn_data[len(data.columns)-1], knn_data["k="+str(i)])
        knn_accuracy.at[i,"accuracy"]=accuracy_value

    return knn_accuracy 


# calculate average and standard deviation of accuracy
def accuracy_avg_std(accuracy_table):
    accuracy_table["Mean"]=accuracy_table.mean(axis=1)
    accuracy_table["Std"]=accuracy_table.std(axis=1)
    print(accuracy_table)



# Graph created with k
def graph_accuracy(accuracy_table, title):
    # plot the data
    plt.figure(figsize=(10, 5))
    plt.plot(accuracy_table)

    # formatting 
    plt.xlabel("Value of k")
    plt.ylabel("Accuracy over "+title)
    plt.savefig(title+".png",dpi=300, bbox_inches='tight')

    # message
    print("save graph image file to the folder")

# main function - united all function ahead
def main():
    accuracy_train=pd.DataFrame()
    accuracy_test=pd.DataFrame()
    # iterate 20 times 
    for try_count in range(2):
        print("================================================================================")
        print(f"[[ try = {try_count+1} ]]")
        # preprocess dataset
        train_data, test_data=process_dataset()

        # train_data
        train_euclidean=euclidean_matrix(train_data, train_data)
        train_knn=knn_algorithm(51, train_euclidean, train_data,"train",try_count)
        accuracy_train["try="+str(try_count)]=train_knn

        # train_data -> excel
        train_data.to_excel('train_data.xlsx')
        train_euclidean.to_excel("train_eucliean.xlsx")
        train_knn.to_excel("train_knn.xlsx")
        accuracy_train.to_excel("accuracy_train.xlsx")

        # test_data
        test_euclidean=euclidean_matrix(train_data,test_data)
        test_knn=knn_algorithm(51, test_euclidean, test_data,"test",try_count)
        accuracy_test["try="+str(try_count)]=test_knn

        # test_data -> excel
        test_data.to_excel('test_data.xlsx')
        test_euclidean.to_excel("test_eucliean.xlsx")
        test_knn.to_excel("test_knn.xlsx")
        accuracy_test.to_excel("accuracy_test.xlsx")
    
    accuracy_train.to_excel("accuracy_train.xlsx")
    accuracy_train.to_excel("accuracy_test.xlsx")
    # make graph
    # graph_accuracy(accuracy_table, "training data")
    # message
    print("[[ Complete task! ]]\n")

# ensures that the main function is executed only
if __name__ == "__main__":
    main()