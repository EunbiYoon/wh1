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
def entropy_logic(train_class, train_attribute, branch_info):
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
        print("[[ Entropy Table ==> Zero Entropy ]]")
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

    # save entropy_table
    other_entropy_final.to_excel('entropy_table_'+str(branch_info)+'.xlsx')

    # return max_entropy_attribute
    return max_entropy_attribute

def branch_decision(count, train_attribute, choosen_attribute, attribute_label, train_class):
    print("====> "+str(count+1)+"th Branch :: Data Sorting...")
    # divide attribute based on the first_attribute_column
    divided_attribute=train_attribute[train_attribute[choosen_attribute]==attribute_label[count]]
    # remove choosen attribute column
    divided_attribute=divided_attribute.drop(columns=choosen_attribute)
    # sorted index list from divided_attribute
    index_class=divided_attribute.index.tolist()
    # get the following class
    divided_class = train_class.loc[index_class]

    # message
    print("====> "+str(count+1)+"th Branch :: Entropy Calculating...")
    # choose first attribute to sort
    branch_attribute=entropy_logic(divided_class, divided_attribute,2)
    
    return divided_class, branch_attribute



# draw graph
def draw_graph():
    return 4



def append_to_decision_tree(decision_tree, column_names, values):
    """This function appends a row to the decision tree DataFrame."""
    new_row = {column_names[i]: values[i] for i in range(len(column_names))}
    return pd.concat([decision_tree, pd.DataFrame([new_row])], ignore_index=True)

def process_branch(decision_tree, column_name_1, attribute_label_1, i, choosen_attribute_2, divided_class_2):
    """This function handles the branch decision and appends data to the decision tree."""
    # Append the result to decision tree
    decision_tree = append_to_decision_tree(decision_tree, 
                                            ['level_0', 'level_1', 'level_2'], 
                                            [column_name_1, attribute_label_1[i], divided_class_2.iloc[0]])
    return decision_tree

def main():
    # preprocess data
    car_data, train_attribute, train_class, test_attribute, test_class = process_dataset()

    # build_tree
    decision_tree = pd.DataFrame()

    ######### training data ########
    ########## Select First Attribute! 
    print("\n--> Selecting 1st Attribute...")
    level_count = 0
    choosen_attribute_1 = entropy_logic(train_class, train_attribute, 1)
    column_name_1 = car_data.columns[choosen_attribute_1]
    decision_tree = append_to_decision_tree(decision_tree, ['level_0'], [column_name_1])

    ########## Select Second Attribute! 
    unique_1 = unique_attribute(train_attribute)
    attribute_label_1 = unique_1[choosen_attribute_1].dropna().tolist()
    
    # branching for second attribute
    for i in range(len(attribute_label_1)):
        print("\n--> Selecting 2nd Attribute :: " + str(i + 1) + "th Branches...")
        level_count = level_count + 1
        divided_class_2, choosen_attribute_2 = branch_decision(i, train_attribute, choosen_attribute_1, attribute_label_1, train_class)
        
        # check if we can stop or not
        if choosen_attribute_2 == "zero_entropy":
            # Use the process_branch function to handle appending the data to decision tree
            decision_tree = process_branch(decision_tree, column_name_1, attribute_label_1, i, choosen_attribute_2, divided_class_2)
        else:
            # Select Third Attribute!
            print("\n--> Selecting 3rd Attribute...")
            unique_2 = unique_attribute(train_attribute)
            attribute_label_2 = unique_2[choosen_attribute_2].dropna().tolist()
            
            # branching for third attribute
            for j in range(len(attribute_label_2)):
                divided_class_3, branch_attribute_3 = branch_decision(j, train_attribute, choosen_attribute_2, attribute_label_2, train_class)
                
                # check if we can stop or not
                if branch_attribute_3 == "zero_entropy":
                    decision_tree = append_to_decision_tree(decision_tree, 
                                                            ['level_0', 'level_1', 'level_2', 'level_3'], 
                                                            [column_name_1, attribute_label_1[i], attribute_label_2[j], divided_class_3.iloc[0]])

                else:
                    # Further branching logic can be added as needed
                    pass

    print("Final Decision Tree:")
    print(decision_tree)

# ensures that the main function is executed only.
# check update
if __name__ == "__main__":
    main()

