import math
import pandas as pd
import copy
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score
import argparse
parser = argparse.ArgumentParser()

class DecisionTree():
    def __init__(self, outcome):
        """
            Parameters:
                outcome (string): prediction outcome
        """
        self.outcome = outcome

    def get_unique_values(self, data, feature):
        """
            Get unique values of a given feature

            Parameters:
                data (Dataframe): training data or part of training data
                feature (string): feature

            Returns:
                list: unique values for the given feature
        """
        return list(data[feature].unique())
    
    def calculate_frequencies(self, data):
        """
            Frequencies of the outcome for given data

            Parameters:
                data (Dataframe): training data or part of training data
                
            Returns:
                dict: frequencies of the outcome
        """
        frequencies = dict(data[self.outcome].value_counts())
        return frequencies
    
    def calculate_entropy(self, data):
        """
            Calculates Entropy (Measure of disorder in the data)

            Parameters:
                data (Dataframe): training data or part of training data
                
            Returns:
                float: entropy
        """
        entropy = 0
        frequencies = self.calculate_frequencies(data)
        
        for frequency in frequencies.values():
            probability = frequency/len(data)
            if probability == 0:
                entropy = -math.inf
                break
            entropy -= probability*math.log2(probability)
        
        return entropy

    def split_data(self, data, feature):
        """
            Chooses the feature which causes the maximum info gain

            Parameters:
                data (Dataframe): training data or part of training data
                feature (string): feature
            
            Returns:
                list: list of dataframes with unique value for the feature
        """
        subsets = []
        unique_values = self.get_unique_values(data, feature)

        for value in unique_values:
            subset = data[data[feature]==value]
            subsets.append(subset)

        return subsets

    def calculate_entropy_after_split(self, data, feature):
        """
            Calculates Entropy (Measure of disorder in the data)

            Parameters:
                data (Dataframe): training data or part of training data
                feature (string): feature
                
            Returns:
                float: entropy
        """
        entropy = 0
        subsets = self.split_data(data, feature)

        for subset in subsets:
            frequencies = self.calculate_frequencies(subset)

            for frequency in frequencies:
                probability = frequency/len(subset)
                if probability == 0:
                    entropy = -math.inf
                    break
                entropy -= probability*math.log2(probability)
        
        return entropy

    def choose_best_feature(self, data, features):
        """
            Chooses the feature which causes the maximum info gain

            Parameters:
                data (Dataframe): training data or part of training data
                features (list): features
            
            Returns:
                string: feature
        """
        entropy = self.calculate_entropy(data)
        max_info_gain = 0
        best_feature = None

        for feature in features:
            split_entropy = self.calculate_entropy_after_split(data, feature)
            info_gain = entropy - split_entropy
            if info_gain > max_info_gain:
                max_info_gain = info_gain
                best_feature = feature
        
        return best_feature

    def build_decision_tree(self, data, features):
        """
            Builds a decision tree
            
            Parameters:
                data (Dataframe): training data or part of training data
                features (list): features

            Returns:
                dict: decision tree model
        """
        # base case: If all the target values are the same return the target value
        if len(data[self.outcome].unique()) == 1:
            return data[self.outcome].unique()[0]
        if len(data) == 0:
            return self.most_common_target_value
        # base case: if there are no more features to split on, return the most common target value
        if len(features) == 0:
            return data[self.outcome].value_counts().idxmax()
        # choose the best feature to split on
        best_feature = self.choose_best_feature(data, features)
        # create a decision tree for the best feature
        tree = {best_feature: {}}
        # remove the feature from the list of features
        features.remove(best_feature)
        # split the data based on the best feature
        unique_values_of_best_feature = self.get_unique_values(data, best_feature)
        for value in unique_values_of_best_feature:
            sub_data = data[data[best_feature]==value]
            sub_tree = self.build_decision_tree(sub_data, features)
            tree[best_feature][value] = sub_tree
        
        return tree

    def handle_missing_values(self, data):
        """
            Naive fix to handle values missing from training data
        """
        # get the most frequent target value from training data
        most_common_target_value = data[self.outcome].value_counts().idxmax()
        self.most_common_target_value = most_common_target_value
    
    def compute_error_rate(self, tree, data):
        """
            Computes error rate

            Parameters:
                tree (dict): decision tree
                data (Dataframe): Training data
            
            Returns:
                float: Error rate
        """
        # initialize the number of incorrect predictions
        num_incorrect = 0
        
        for index, row in data.iterrows():
            # make a prediction for sample using the tree
            sample = dict(row)
            true_label = sample.pop(outcome)
            predicted_label = self.predict(tree, sample)

            # if the prediction is incorrect, increment the number of incorrect predictions
            if true_label != predicted_label:
                num_incorrect += 1

        return num_incorrect / len(data)

    def prune_tree(self, tree, data):
        """
            Prune given tree

            Parameters:
                tree (dict): decision tree
            
            Returns:
                dict: Prunes tree
        """
        # Check if tree is a leaf node
        if type(tree) != dict:
            return tree
        tree_copy = copy.deepcopy(tree)

        # Traverse the subtrees
        feature = list(tree.keys())[0]
        for subtree in tree[feature].values():
            self.prune_tree(subtree, data)
        
        # If pruning increased the error rate return restored tree
        if self.compute_error_rate(tree, data) > self.compute_error_rate(tree_copy, data):
            return tree_copy
        else:
            return tree
    
    def reduced_error_pruning(self, tree, data):
        """
            Removes subtrees that doesn't reduce the error rate

            Parameters:
                tree (dict): decision tree
                data (Dataframe): Training data
            
            Returns:
                dict: Pruned tree
        """
        # Check if tree is a leaf node
        if type(tree) != dict:
            return tree
        tree_copy = copy.deepcopy(tree)

        # Traverse the subtrees
        feature = list(tree.keys())[0]
        for subtree in tree[feature].values():
            subtree = self.reduced_error_pruning(subtree, data)
        tree_copy = self.prune_tree(tree_copy, data)

        # If pruning increased the error rate return restored tree
        if self.compute_error_rate(tree, data) > self.compute_error_rate(tree_copy, data):
            return tree_copy
        else:
            return tree
    
    def predict(self, tree, sample):
        """
            Predicts the outcome using decision tree
            
            Parameters:
                tree (dict): decision tree
                sample (dict): features and values of the sample to be predicted

            Returns:
                string: outcome
        """
        if type(tree) != dict:
            return tree
        feature = list(tree.keys())[0]
        if sample[feature] not in tree[feature]:
            return self.most_common_target_value
        return self.predict(tree[feature][sample[feature]], sample)

def main(training_data_path, outcome):
    # Load training data
    data = pd.read_csv(training_data_path)
    features = list(data.columns)
    features.remove(outcome)
    
    # Initiate decision tree object
    decision_tree = DecisionTree(outcome)
    train_data, test_data = train_test_split(data, test_size=0.2, random_state=5)
    decision_tree.handle_missing_values(train_data)
    
    # Build decision tree
    model = decision_tree.build_decision_tree(train_data, features)

    # Pruning
    model = decision_tree.reduced_error_pruning(model, train_data)
    
    # Predict for the test data
    true_labels = []
    predicted_labels = []
    for index, row in test_data.iterrows():
        sample = dict(row)
        true_label = sample.pop(outcome)
        predicted_label = decision_tree.predict(model, sample)
        true_labels.append(true_label)
        predicted_labels.append(predicted_label)
    
    # Metrics
    print("precision score", precision_score(true_labels, predicted_labels))
    print("recall score", recall_score(true_labels, predicted_labels))
    print("f1 score", f1_score(true_labels, predicted_labels))

if __name__ == "__main__":
    parser.add_argument("--training_data_path", help="Training data path")
    parser.add_argument("--outcome", help="outcome column name")
    args = parser.parse_args()
    training_data_path = args.training_data_path
    outcome = args.outcome
    main(training_data_path, outcome)
