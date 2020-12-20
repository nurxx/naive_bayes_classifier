import pandas as pd 
import numpy as np 

def get_training_dataset(index, folds):
    training_dataset = folds.copy()
    test_dataset = training_dataset.pop(index)
    training_dataset = pd.concat(training_dataset, ignore_index=True, sort=False)
    
    return test_dataset, training_dataset

def ten_folds(instances):
    instances = instances.sample(frac=1).reset_index(drop=True)
    no_of_columns = len(instances.columns)
    no_of_rows = len(instances.index)
   
    folds = []
    for _ in range(10):
        folds.append(pd.DataFrame(columns=(instances.columns)))
 
    actual_class_column = no_of_columns - 1
    unique_class_list_df = instances.iloc[:,actual_class_column]
    unique_class_list_df = unique_class_list_df.sort_values()
    unique_class_list_np = unique_class_list_df.unique() 
    unique_class_list_df = unique_class_list_df.drop_duplicates()
 
    unique_class_list_np_size = unique_class_list_np.size

    for unique_class_list_np_idx in range(0, unique_class_list_np_size):
        counter = 0
        for row in range(0, no_of_rows):
            if unique_class_list_np[unique_class_list_np_idx] == (
                instances.iloc[row,actual_class_column]):
                    new_row = instances.iloc[row,:]
                    folds[counter].loc[len(folds[counter])] = new_row
                    counter += 1
                    if counter == 10: 
                        counter = 0
    return folds

def naive_bayes(training_set, test_set):
    no_of_instances_train = len(training_set.index)
    no_of_columns_train = len(training_set.columns)
    no_of_attributes = no_of_columns_train - 2
    actual_class_column = no_of_columns_train - 1
  
    unique_class_list_df = training_set.iloc[:,actual_class_column]
    unique_class_list_df = unique_class_list_df.sort_values()
    unique_class_list_np = unique_class_list_df.unique() 
    unique_class_list_df = unique_class_list_df.drop_duplicates()
    num_unique_classes = len(unique_class_list_df)
  
    freq_cnt_class = training_set.iloc[:,actual_class_column].value_counts(
        sort=True)
    # list of the class prior probabilities
    class_prior_probs = training_set.iloc[:,actual_class_column].value_counts(
        normalize=True, sort=True)
    test_set = test_set.reindex(
        columns=[*test_set.columns.tolist(
        ), 'Predicted Class', 'Prediction Correct'])
    no_of_instances_test = len(test_set.index) 
    no_of_columns_test = len(test_set.columns) 
    predicted_class_column = no_of_columns_test - 2
    prediction_correct_column = no_of_columns_test - 1
  
    d = {}
    for col in range(1, no_of_attributes + 1):
        colname = training_set.columns[col]
        unique_attribute_values_df = training_set[colname].drop_duplicates()
        unique_attribute_values_np = training_set[colname].unique()
      
        # Calculate likelihood of the attribute given each unique class value
        for class_index in range (0, num_unique_classes):
            # For each unique attribute value, calculate the likelihoods
            for attr_val in range (0, unique_attribute_values_np.size) :
                running_sum = 0
                for row in range(0, no_of_instances_train):
                    if (training_set.iloc[row,col] == (
                        unique_attribute_values_df.iloc[attr_val])) and (
                        training_set.iloc[row, actual_class_column] == (
                        unique_class_list_df.iloc[class_index])):
                            running_sum += 1
                try:
                    denominator = freq_cnt_class[class_index]
                except:
                    denominator = 1.0
                likelihood = min(1.0,(running_sum / denominator))
                search_key = str(colname) + str(
                    unique_attribute_values_df.iloc[
                    attr_val]) + str(unique_class_list_df.iloc[
                    class_index])
                d[search_key] = likelihood

    # Calculate prediction    
    for row in range(0, no_of_instances_test):
        predicted_class = unique_class_list_df.iloc[0]
        max_numerator_of_bayes = 0.0
  
        # Calculate the Bayes equation numerator for each test instance
        for class_index in range (0, num_unique_classes):
            # Prior probabilities
            try:
                running_product = class_prior_probs[class_index]
            except:
                running_product = 0.0000001 
            # Calculation of P(CL) * P(E1|CL) * P(E2|CL) * P(E3|CL)...
            for col in range(1, no_of_attributes + 1):
                attribute_name = test_set.columns[col]
                attribute_value = test_set.iloc[row,col]
                class_value = unique_class_list_df.iloc[class_index]
  
                key = str(attribute_name) + str(attribute_value) + str(class_value)
                try:
                    running_product *= d[key]
                except:
                    running_product *= 0

            if running_product > max_numerator_of_bayes:
                max_numerator_of_bayes = running_product
                predicted_class = unique_class_list_df.iloc[class_index] 
  
        test_set.iloc[row, predicted_class_column] = predicted_class
        # Store if the prediction was correct
        if predicted_class == test_set.iloc[row,actual_class_column]:
            test_set.iloc[row, prediction_correct_column] = 1
        else: 
            test_set.iloc[row, prediction_correct_column] = 0
    predictions = test_set
    accuracy = (test_set.iloc[:,prediction_correct_column].sum())/no_of_instances_test
  
    return accuracy, predictions 

def main():
    data_path = "data.csv"
    pd_data_set = pd.read_csv(data_path, sep = ",")

    NO_OF_FOLDS = 10
    folds = ten_folds(pd_data_set)
 
    training_dataset = None
    test_dataset = None
    accuracy_statistics = np.zeros(NO_OF_FOLDS)
 
    for experiment in range(0, NO_OF_FOLDS):
        print("Experiment: " + str(experiment + 1), end = "\t")
        test_dataset, training_dataset = get_training_dataset(experiment, folds)
        accuracy, predictions = naive_bayes(training_dataset,test_dataset)
        print("Accuracy: {:.4f}".format(accuracy * 100), "%")
        accuracy_statistics[experiment] = accuracy

    print()
    print("Accuracy Statistics: {:.4f}".format(sum(accuracy_statistics) / len(accuracy_statistics)*100), "%")
 
main()