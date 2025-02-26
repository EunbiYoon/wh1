import pandas as pd
import numpy as np
import sklearn as sk
import matplotlib.pyplot as plt
import re


# Prepared train_data, test_data
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
    shuffled_data = sk.utils.shuffle(normalized_data)

    # split data with training and testing
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

# check majority
def majority_formula(list):
    # count 1 or 0, if there is nothing value is 0
    count_1 = list.value_counts().get(1, 0)  
    count_0 = list.value_counts().get(0, 0) 

    # betwen 1 and 0 which one is more
    if count_1 > count_0:
        return 1
    else:
        return 0


# Accuracy in Training and Testing
def calculate_accuracy(actual_list, predicted_list):
    # compare two column. matched->1, mismatched->0
    compared_result = list((actual_list == predicted_list).astype(int))
    
    # calculate accuracy
    accuracy_list=(sum(compared_result)/len(compared_result))   

    # return accuracy value
    return accuracy_list

                                        
# KNN algorithm using Euclidian Matrix
def knn_algorithm(k, train_euclidean, data, accuracy_table, data_info, try_count):
    # make empty dataframe
    knn_data=pd.DataFrame(data)

    # Iterate over k values : 1~51 odd number
    for i in range(1,k+1,2): 
        # j is for data instances 
        for j in range(len(train_euclidean)):
            # find smallest components count:k for xj
            k_smallest_column=train_euclidean.loc[j].nsmallest(i)

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
            print(f"knn algorithm : data_type={data_info} , try={try_count} , k={i} , instance_number={j}")
        
        # calcualte accuracy depending on k and make accuracy_list
        accuracy_value=calculate_accuracy(knn_data[len(data.columns)-1], knn_data["k="+str(i)])
        accuracy_table.at["try="+str(try_count),"accuracy (k="+str(i)+")"]=accuracy_value
        print("\n*** Accuracy table ==> "+str(data_info)+" dataset ***")
        print(accuracy_table)

    return knn_data, accuracy_table 


# calculate average and standard deviation of accuracy
def accuracy_avg_std(accuracy_table):
    # add mean and std bottom of the each column (same k, different try)
    meanstd = accuracy_table.agg(['mean', 'std'])
    
    # merge accuracy_table with meanstd table 
    graph_table=pd.concat([accuracy_table,meanstd])

    # message
    print("Calcuate mean and standard deviation of each k value")
    return graph_table



# Graph created with k
def draw_graph(accuracy_table, title):
    # Extract integer k values from the column names using regex
    k_values = []
    for col in accuracy_table.columns:
        match = re.search(r'\(k=(\d+)\)', col)
        if match:
            k_values.append(int(match.group(1)))
        else:
            k_values.append(col)  # if the pattern is not found, keep the original name

    # Get mean and std rows from the DataFrame
    mean_values = accuracy_table.loc['mean']
    std_values =accuracy_table.loc['std']

    print("error detect")
    print(mean_values)

    # plot the data
    plt.figure(figsize=(6, 4))
    plt.errorbar(
        k_values,            # x-axis: k=1,3,5,7
        mean_values,         # y-axis: mean values
        yerr=std_values,     # error bars: std values
        fmt='o',           # 'o' marker with a line
        capsize=5,           # size of the error bar caps
        label='Mean ± STD'
    )

    # formatting 
    plt.xlabel("Value of k")
    plt.ylabel("Accuracy over "+title)
    plt.title(title)
    plt.savefig(title+".png",dpi=300, bbox_inches='tight')

    # message
    print("saved graph image file...")

# main function - united all function above
def main():
    train_accuracy_table=pd.DataFrame()
    test_accuracy_table=pd.DataFrame()

    # iterate 20 times 
    for try_count in range(1,21):
        print("\n================================================================================")
        print(f"[[ try = {try_count} ]]")
        
        # preprocess dataset
        train_data, test_data=process_dataset()

        # make euclidean-knn algorithm using train_data and check accuracy between trian_data, test_data
        train_euclidean=euclidean_matrix(train_data, train_data)
        train_knn, train_accuracy=knn_algorithm(51, train_euclidean, train_data, train_accuracy_table, "train", try_count)

        test_euclidean=euclidean_matrix(train_data, test_data)
        test_knn, test_accuracy=knn_algorithm(51, test_euclidean, test_data, test_accuracy_table, "test", try_count)

    # draw graph
    train_graph_table=accuracy_avg_std(train_accuracy)
    draw_graph(train_graph_table,"train_dataset")

    test_graph_table=accuracy_avg_std(test_accuracy)
    draw_graph(test_graph_table,"test_dataset")

    # transfer to excel
    train_data.to_excel('train_data.xlsx') 
    train_euclidean.to_excel("train_euclidean.xlsx")
    train_knn.to_excel("train_knn.xlsx")
    train_accuracy.to_excel("train_accuracy.xlsx")
    train_graph_table.to_excel('train_graph_table.xlsx')

    test_data.to_excel('test_data.xlsx')
    test_euclidean.to_excel("test_euclidean.xlsx")
    test_knn.to_excel("test_knn.xlsx")
    test_accuracy.to_excel("test_accuracy.xlsx")
    test_graph_table.to_excel('test_graph_table.xlsx')

    # message
    print("[[ Complete task! ]]\n")

# ensures that the main function is executed only.
if __name__ == "__main__":
    main()