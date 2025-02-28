import pandas as pd
import numpy as np
import sklearn as sk

# separate attribute and class in data
def attribute_class(data):
    # separate attributes and class
    attribute_data=data.iloc[:,:-1]
    class_data=data.iloc[:, -1]
    return attribute_data, class_data

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
    train_data.to_excel('train_data.xlsx')
    test_data.to_excel('test_data.xlsx')
    # reset index both train_data and test_data
    train_data=train_data.reset_index(drop=True)
    test_data=test_data.reset_index(drop=True)

    # separate attribute <-> class
    train_attribute, train_class=attribute_class(train_data)
    test_attribute, test_class=attribute_class(test_data)

    # message
    print("Shuffling are done. Split train_data and test_data. Please wait...")

    # return final train_data, test_data
    return car_data, train_attribute, train_class, test_attribute, test_class


# get label from each attribute
def unique_attribute(train_attribute):
    # reset column name as number
    train_attribute.columns=list(range(len(train_attribute.columns)))

    # remove duplicate and put the column into unique_attribute_table
    unique_attributes_table=pd.DataFrame()
    for i in range(len(train_attribute.columns)):
        remove_duplicate=train_attribute[i].drop_duplicates().reset_index(drop=True)
        unique_attributes_table[i]=remove_duplicate
    return unique_attributes_table

# entropy formula
def entropy_formula(data_series):
    # count original_data label 
    class_count = data_series.value_counts()

    # calculate probability 
    total_count = len(data_series)
    prob_list = class_count / total_count

    # Filter out probabilities that are zero (since log2(0) is undefined)
    prob_valid = prob_list[prob_list > 0]  

    # calculate entropy
    entropy = -np.sum(prob_valid * np.log2(prob_valid))
    
    return entropy


# create entropy logic
def entropy_logic(train_class, train_attribute):
    # original entropy calculate
    original_entropy=entropy_formula(train_class)
        
    # unique attribute
    unique_attributes_table=unique_attribute(train_attribute)

    ##### other etropy #####
    other_entropy=pd.DataFrame()
    attribute_count=pd.DataFrame()
    for i in range(len(unique_attributes_table.columns)):
        for j in range(len(unique_attributes_table.index)):
            # find unique values in each attribute
            unique_value=unique_attributes_table.at[j,i]
            
            # filtered data same as unique_vlaue in each attribute
            filtered_attribute = train_attribute[train_attribute[i]==unique_value]

            # find sorted data index
            filtered_index=filtered_attribute.index
            
            # find class label usign sorted data index
            filtered_class=train_class[filtered_index]

            # calculate entropy
            other_entropy.at[j,i]=entropy_formula(filtered_class)
            
            # count entropy
            attribute_count.at[j,i]=len(filtered_index)
    
    # if there is all 0 entropy, stop the branching
    if (other_entropy==0).all().all():
        print("====> Stopping branches and writing decision tree...")
        return "zero_entropy"
    
    # change all the components as integer
    attribute_count=attribute_count.fillna(0).astype(int)

    # add sum at the botton of attribute_count
    attribute_count.loc['total_count_sum']=attribute_count.sum()

    # check each attribute sum are all the same
    sum_row = attribute_count.iloc[-1]
    if np.all(sum_row == sum_row[0]):  
        sum_count=sum_row[0]
    else:
        print("error - check attribute_count table")


    # divide each count by sum
    weighted_entropy=attribute_count.div(sum_count)

    # attribute_count_prob x other_entropy
    other_entropy_final=pd.DataFrame()
    for i in range(len(other_entropy)):
        for j in range(len(other_entropy.columns)):
            other_entropy_final.at[i,j]=other_entropy.at[i,j]*weighted_entropy.at[i,j]
    
    # add sum 
    other_entropy_final.loc['weight_entropy_sum']=other_entropy_final.sum()

    # calculate information gain
    other_entropy_final.loc['information_gain']=original_entropy-other_entropy_final.loc['weight_entropy_sum']

    # choose lowest information gain attribute for decision tree
    max_entropy_attribute=other_entropy_final.loc['information_gain'].idxmax()

    # print entropy_table
    print("[[ Entropy Table ==> Choosen Attribute = "+str(max_entropy_attribute)+"th Column ]]")
    print(other_entropy_final)
    print("\n")

    # return max_entropy_attribute
    return max_entropy_attribute



# draw graph
def draw_graph():
    return 4

# main function - united all function above
def main():
    # preprocess data
    # train_attribute = 1382 / test_attribute = 346 / total = 1728
    car_data, train_attribute, train_class, test_attribute, test_class=process_dataset()

    # build_tree
    decision_tree=pd.DataFrame()

    ######### training data ########
    # Select First Attribute! 
    print("\n--> Selecting 1st Attribute...")
    # unique attribute
    first_unique=unique_attribute(train_attribute)

    # choose first attribute to sort
    choosen_first_attribute=entropy_logic(train_class, train_attribute)

    # get the real column name
    first_column_name=car_data.columns[choosen_first_attribute]

    # input decision tree
    decision_tree.at[0,"level_0"]=first_column_name

    # separate train_attribute depends on choosen first attribute
    first_attribute_labels=first_unique[choosen_first_attribute].dropna().tolist()

    # Select Second Attribute! 
    print("\n-> Selecting 2th Attribute...")
    for i in range(len(first_attribute_labels)):
        print("====> 2th Attribute "+str(i+1)+"th Branch :: Data Sorting...")
        # divide attribute based on the first_attribute_column
        divided_attribute=train_attribute[train_attribute[choosen_first_attribute]==first_attribute_labels[i]]
        # remove choosen attribute column
        divided_attribute=divided_attribute.drop(columns=choosen_first_attribute)
        # sorted index list from divided_attribute
        index_class=divided_attribute.index.tolist()
        # get the following class
        divided_class = train_class.loc[index_class]

        print("====> 2th Attribute "+str(i+1)+"th Branch :: Entropy Calculating...")
        # choose first attribute to sort
        second_attribute_column=entropy_logic(divided_class, divided_attribute)
        # message

        # check able to stop or not
        if second_attribute_column=="zero_entropy":
            # input decision tree
            decision_tree.at[1,"level_1"]=first_attribute_labels[i]
            decision_tree.at[1,"level_2"]=divided_class.iloc[0]
        else:
            ee=3


    print("\n-> Selecting 3rd Attribute...")
    print("\n###### DECISION TREE #####")
    print(decision_tree)
    # message
    print("\n[[ Complete task! ]]\n")

# ensures that the main function is executed only.
if __name__ == "__main__":
    main()

