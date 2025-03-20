import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
import random

class Random_Forest():
    def __init__(self, name, dataset, feature_type, class_col) -> None:
        self.name = name
        self.class_col = class_col
        self.feature_type = feature_type

        print('Starting Random Forest')
        print('Preprocessing data...')

        self.data, self.class_label = self.df_to_array(dataset, class_col)
        self.class_count = len(self.class_label)

        self.stratified_data = self.k_fold_stratified(10)

        print('Done preproccessing data')
        print('Training forest...')

        ntree_values, accuracy, f1 = self.train()

        print('Done training forest')
        print('Graphing results...')

        self.plot_result(ntree_values, accuracy, f1)

    def df_to_array(self, data_df, class_col):
        data = []
        classes = []
        for index, row in data_df.iterrows():
            instance_data = {}
            for key, value in row.items():
                instance_data[key] = value
                if key == class_col:
                    if value not in classes:
                        classes.append(value)
            data.append(instance_data)
        random.shuffle(data)
        return np.array(data), np.array(classes)
    
    def bootstrap_data(self, data): 
        result_data = []
        for i in range(len(data)):
            result_data.append(data[random.randint(0, len(data) - 1)])
        return np.array(result_data)

    def k_fold_stratified(self, k):
        k_folds = [[] for _ in range(k)]
        data_group = [[] for _ in range(self.class_count)]

        for i in range(len(self.data)):
            class_index = np.where(self.class_label == self.data[i].get(self.class_col))[0][0]
            data_group[class_index].append(self.data[i])

        for j in range(len(data_group)):
            fold_size = math.ceil(len(data_group[j]) / k)
            for n in range(k):
                start_index = n * fold_size
                end_index = start_index + fold_size
                k_folds[n].extend(data_group[j][start_index : end_index])
                
        return k_folds
    
    def train(self):
        ntree_values = [5, 15, 25, 35]

        accuracy = []
        f1 = []

        table = pd.DataFrame([], columns=['ntree', 'accuracy', 'F1'])

        for ntree in ntree_values:
            print()
            print('ntree:', ntree)
            print()
            
            ntree_accuracy = []
            ntree_f1 = []

            for i in range(len(self.stratified_data)):
                print('fold:', i + 1, end='\r')
                training = self.stratified_data[:i] + self.stratified_data[i + 1:]
                training = [element for sub_array in training for element in sub_array]

                testing = self.stratified_data[i]

                forest = []
                forest_accuracy = []
                forest_f1 = []

                for _ in range(ntree):
                    bootstrap_training = self.bootstrap_data(training)
                    tree = Decision_Tree_Info_Gain(training, bootstrap_training, self.feature_type, self.class_label, self.class_col, 0.85)
                    forest.append(tree)
                
                for tree in forest:
                    true_predictions = [0 for _ in range(self.class_count)]
                    false_predictions = [0 for _ in range(self.class_count)]
                    for n in range(len(testing)):
                        prediction = tree.classify(testing[n])
                        index = np.where(self.class_label == prediction)[0][0]
                        if testing[n].get(self.class_col) == prediction:
                            true_predictions[index] += 1
                        else:
                            false_predictions[index] += 1

                    tree_accuracy, tree_f1 = self.compute_stats(true_predictions, false_predictions, len(testing))

                    forest_accuracy.append(tree_accuracy)
                    forest_f1.append(tree_f1)
                
                forest_mean_accuracy = np.mean(forest_accuracy)
                forest_mean_f1 = np.mean(forest_f1)

                ntree_accuracy.append(forest_mean_accuracy)
                ntree_f1.append(forest_mean_f1)

            ntree_mean_accuracy = np.mean(ntree_accuracy)
            ntree_mean_f1 = np.mean(ntree_f1)

            accuracy.append(ntree_mean_accuracy)
            f1.append(ntree_mean_f1)

            new_data = {
                'ntree': ntree,
                'accuracy': ntree_mean_accuracy,
                'F1': ntree_mean_f1
            }

            table.loc[len(table)] = new_data
        
        table.to_csv('./tables/' + self.name + '_table.csv', index=False)
    
        return ntree_values, accuracy, f1
                
    def compute_stats(self, true_values, false_values, data_size):
        accuracy = np.sum(true_values) / data_size

        if self.class_count == 2:
            if true_values[0] + false_values[0] == 0:
                precision = 0
            else:
                precision = true_values[0] / (true_values[0] + false_values[0])
            
            if true_values[0] + false_values[1] == 0:
                recall = 0
            else:
                recall = true_values[0] / (true_values[0] + false_values[1])
        else:
            false_values_sum = np.sum(false_values)
            precision = 0
            recall = 0

            for i in range(self.class_count):
                if true_values[i] + false_values[i] == 0:
                    precision = 0
                else:
                    precision += true_values[i] / (true_values[i] + false_values[i])
                if true_values[i] + (false_values_sum - false_values[i]) == 0:
                    recall = 0
                else:
                    recall += true_values[i] / (true_values[i] + (false_values_sum - false_values[i]))
            
            precision /= self.class_count
            recall /= self.class_count

        if precision + recall == 0:
            f1 = 0
        else:
            f1 = 2 * (precision * recall) / (precision + recall)
        
        return accuracy, f1

    def plot_result(self, ntree_values, accuracy, f1):
        plt.plot(ntree_values, accuracy, marker='o')
        plt.xlabel("ntree Values")
        plt.ylabel("Accuracy")
        plt.title(self.name + ' Accuracy')
        plt.savefig('./figures/' + self.name + ' accuracy')
        plt.clf()

        plt.plot(ntree_values, f1, marker='o')
        plt.xlabel("ntree Values")
        plt.ylabel("F1 score")
        plt.title(self.name + ' F1')
        plt.savefig('./figures/' + self.name + ' f1')
        plt.clf()

class Decision_Tree_Info_Gain():

    def __init__(self, training_data, bootstrap_training_data, feature_type, class_label, class_col, prob_limit) -> None:
        self.original = training_data
        self.data = bootstrap_training_data
        self.feature_type = feature_type
        self.class_label = class_label
        self.class_count = len(self.class_label)
        self.class_col = class_col
        self.prob_limit = prob_limit
        self.m = math.ceil(math.sqrt(len(feature_type)))

        self.root = self.build_tree(self.data, self.feature_type)

    # Get the probability of the class labels
    def get_probability(self, data):
        class_prob = np.array([0 for _ in range(self.class_count)], dtype='float64')

        if len(data) == 0:
            return class_prob

        for instance in data:
            for i in range(self.class_count):
                if instance.get(self.class_col) == self.class_label[i]:
                    class_prob[i] += 1
                    break
            
        class_prob /= len(data)
        return class_prob
    
    def select_features(self, features):
        selected_features = []
        temp_copy = features.copy()

        for i in range(self.m):
            if not temp_copy:
                break

            attribute_index = random.randint(0, len(temp_copy) - 1)
            random_attribute = temp_copy[attribute_index]
            selected_features.append(random_attribute)
            temp_copy.pop(attribute_index)

        return selected_features
    
    # Get entropy of the data
    def entropy(self, data):
        class_prob = self.get_probability(data)
        entro_log_sum = 0

        for prob in class_prob:
            if prob == 0:
                return 0
            entro_log_sum += prob * math.log(prob, 2)
        
        return -1 * entro_log_sum

    def info_gain(self, parent_entropy, probabilities, entropies):
        child_entropy = 0
        for prob, entro in zip(probabilities, entropies):
            child_entropy += prob * entro
        return parent_entropy - child_entropy
    
    # Finding the best split among the attributes and the values
    def best_split(self, data, features):
        parent_entropy = self.entropy(data)
        split_values = []
        feature_split = None
        data_type = None
        split_branches = []
        max_info_gain = -1

        # Looping through each attribute
        for feature, feature_type in features:
            feature_values = set()

            # For numerical attributes
            if feature_type == 'n':
                data = sorted(data, key=lambda x: x.get(feature))

                for i in range(len(data) - 1):
                    instance = (data[i].get(feature) + data[i + 1].get(feature)) / 2
                    feature_values.add(instance)
                
                for feature_value in feature_values:
                    branches = [[] for _ in range(2)]
                    left = branches[0]
                    right = branches[1]

                    for instance in data:
                        if instance.get(feature) <= feature_value:
                            left.append(instance)
                        else:
                            right.append(instance)
                    
                    data_size = len(data)
                    probabilities = [len(left) / data_size, len(right) / data_size]
                    entropies = [self.entropy(left), self.entropy(right)]
                    info_gain = self.info_gain(parent_entropy, probabilities, entropies)

                    if info_gain > max_info_gain:
                        max_info_gain = info_gain
                        split_values = [feature_value]
                        feature_split = feature
                        data_type = feature_type
                        split_branches = branches
            # For categorical attributes
            else:
                for instance in self.original:
                    feature_values.add(instance.get(feature))
                
                feature_values = list(feature_values)
            
                branches = [[] for _ in range(len(feature_values))]
                data_size = len(data)

                for instance in data:
                    branch_index = feature_values.index(instance.get(feature))
                    branches[branch_index].append(instance)
                
                probabilities = []
                entropies = []
                
                for branch in branches:
                    probabilities.append(len(branch) / data_size)
                    entropies.append(self.entropy(branch))
                
                info_gain = self.info_gain(parent_entropy, probabilities, entropies)

                if info_gain > max_info_gain:
                    max_info_gain = info_gain
                    split_values = feature_values
                    feature_split = feature
                    data_type = feature_type
                    split_branches = branches

        return feature_split, split_values, data_type, split_branches
    
    def build_tree(self, data, features):
        class_prob = self.get_probability(data)

        selected_features = self.select_features(features)
        best_split = self.best_split(data, selected_features)   

        feature_split = best_split[0]
        split_values = best_split[1]
        feature_type = best_split[2]
        split_branches = best_split[3]

        if feature_split is None or any(prob >= self.prob_limit for prob in class_prob) or any(len(branch) == 0 for branch in split_branches):

            predict_label = None
            highest_prob = -1

            for i in range(self.class_count):
                if class_prob[i] > highest_prob:
                    predict_label = self.class_label[i]
                    highest_prob = class_prob[i]

            return Leaf(predict_label)

        children = []
        for branch in split_branches:
            children.append(self.build_tree(branch, features))

        return Node(feature_split, split_values, feature_type, children)

    # Classify the data
    def classify(self, data):
        return self.root.classify(data)

# This is a node that is used to split the data
class Node():

    def __init__(self, feature_name, threshold, feature_type, children):
        self.feature_name = feature_name
        self.split = threshold
        self.feature_type = feature_type
        self.children = children

    # If less than or equal to then goes to left child, else then goes to right child
    def classify(self, data):
        test_value = data[self.feature_name]
        if self.feature_type == 'n':
            if test_value <= self.split[0]:
                return self.children[0].classify(data)
            else:
                return self.children[1].classify(data)
        else:
            attr_index = self.split.index(test_value)
            return self.children[attr_index].classify(data)

# This is a leaf node that is used to predict the class label of the data
class Leaf():

    def __init__(self, predict_class):
        self.predict_class = predict_class
    
    def classify(self, data):
        return self.predict_class       
