import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import make_column_transformer
import math

class K_NN:

    def __init__(self, name, feature_data, feature_type, class_data, class_col) -> None:
        self.name = name

        print('Starting k-NN')
        print('Preprocessing data...')

        categorical_col, numerical_col = self.identify_col(feature_type)
        numerical_data = feature_data[numerical_col]
        categorical_data = feature_data[categorical_col]

        normalized_numerical_data = np.array(self.normalization(numerical_data).fillna(0))
        categorical_data_transformer = make_column_transformer((OneHotEncoder(sparse_output=False), categorical_col), remainder='passthrough')
        categorical_data_transformed = categorical_data_transformer.fit_transform(categorical_data[categorical_col])

        class_transformer = make_column_transformer((OneHotEncoder(sparse_output=False), [class_col]), remainder='passthrough')
        class_transformed = class_transformer.fit_transform(class_data)

        self.feature_data = np.concatenate((normalized_numerical_data, categorical_data_transformed), axis=1)
        self.class_data = class_transformed
        self.class_count = len(self.class_data[0])

        self.stratified_feature, self.stratified_class = self.k_fold_stratified(10)

        print('Done preprocessing data')
        print('Finding best hyper-parameters...')

        best_k, best_accuracy, best_f1, k_values, accuracy_recorded, f1_recorded = self.find_hyperparameters(self.stratified_feature, self.stratified_class)

        print('Done finding hyperparameters')
        print('Graphing...')

        self.plot_results(k_values, accuracy_recorded, f1_recorded)

        print('Done graphing')

    def identify_col(self, feature_type):
        categorical_col = []
        numerical_col = []
        for col, data_type in feature_type:
            if data_type == 'c':
                categorical_col.append(col)
            else:
                numerical_col.append(col)
        return categorical_col, numerical_col

    def normalization(self, dataset):
        feature_max = dataset.max()
        feature_min = dataset.min()
        return (2 * (dataset - feature_min) / (feature_max - feature_min)) - 1

    def k_fold_stratified(self, k):
        feature_folds = [[] for _ in range(k)]
        class_folds = [[] for _ in range(k)]

        feature_group = [[] for _ in range(self.class_count)]
        class_group = [[] for _ in range(self.class_count)]

        for i in range(len(self.feature_data)):
            class_index = np.where(self.class_data[i] == 1)[0][0]
            feature_group[class_index].append(self.feature_data[i])
            class_group[class_index].append(self.class_data[i])

        for j in range(len(feature_group)):
            fold_size = math.ceil(len(feature_group[j]) / k)
            for n in range(k):
                start_index = n * fold_size
                end_index = start_index + fold_size
                feature_folds[n].extend(feature_group[j][start_index : end_index])
                class_folds[n].extend(class_group[j][start_index : end_index])

        return feature_folds, class_folds

    def calc_euclidean(self, X, x_i):
        return np.sum(np.square(X - x_i))

    def training(self, X, k, feature_data, class_data):
        class_count = {}
        distances = []
        majority = None
        highest_count = 0

        for i in range(len(feature_data)):
            distances.append((np.where(class_data[i] == 1)[0][0], self.calc_euclidean(X, feature_data[i])))

        distances.sort(key = lambda x : x[1])
        k_nearest = distances[:k]

        for class_label, distance in k_nearest:
            if class_label not in class_count:
                class_count[class_label] = 1
            else:
                class_count[class_label] += 1
        
        for key, num in class_count.items():
            if num > highest_count:
                highest_count = num
                majority = key
        
        return majority

    def predict(self, X, k, feature_data, trained_class_label):
        class_count = {}
        distances = []
        majority = None
        highest_count = 0

        for i in range(len(feature_data)):
            distances.append((trained_class_label[i], self.calc_euclidean(X, feature_data[i])))

        distances.sort(key = lambda x : x[1])
        k_nearest = distances[:k]

        for class_label, distance in k_nearest:
            if class_label not in class_count:
                class_count[class_label] = 1
            else:
                class_count[class_label] += 1
        
        for key, num in class_count.items():
            if num > highest_count:
                highest_count = num
                majority = key
        
        return majority
    
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

    def find_hyperparameters(self, feature_data, class_data):
        best_k = -1
        best_accuracy = -1
        best_f1 = -1

        k_values = []
        accuracy_recorded = []
        f1_recorded = []

        table = pd.DataFrame([], columns=['k', 'accuracy', 'F1'])

        for k in range(1, 52, 10):
            print()
            print('k:', k)
            print()
            
            k_accuracy = []
            k_f1 = []
            k_values.append(k)

            for i in range(len(feature_data)):
                training = feature_data[:i] + feature_data[i + 1:]
                training = [element for sub_array in training for element in sub_array]
                training_class_label = class_data[:i] + class_data[i + 1:]
                training_class_label = [element for sub_array in training_class_label for element in sub_array]

                testing = feature_data[i]
                testing_class_label = class_data[i]

                true_predictions = [0 for _ in range(self.class_count)]
                false_predictions = [0 for _ in range(self.class_count)]

                trained_class_label = []

                for n in range(len(training)):
                    trained_class_label.append(self.training(training[n], k, training, training_class_label))
                
                for j in range(len(testing)):
                    predict_result = self.predict(testing[j], k, training, trained_class_label)
                    actual_index = np.where(testing_class_label[j] == 1)[0][0]
                   
                    if predict_result == actual_index:
                        true_predictions[predict_result] += 1
                    else:
                        false_predictions[predict_result] += 1
                
                accuracy, f1 = self.compute_stats(true_predictions, false_predictions, len(testing))
                k_accuracy.append(accuracy)
                k_f1.append(f1)

                if accuracy >= best_accuracy and f1 >= best_f1:
                    best_accuracy = accuracy
                    best_f1 = f1
                    best_k = k
        
            accuracy_recorded.append(np.array(k_accuracy))
            f1_recorded.append(np.array(k_f1))

            new_data = {
                'k': k,
                'accuracy': np.mean(k_accuracy),
                'F1': np.mean(k_f1)
            }

            table.loc[len(table)] = new_data
        
        table.to_csv('./tables/' + self.name + '_table.csv', index=False)

        return best_k, best_accuracy, best_f1, k_values, accuracy_recorded, f1_recorded

    def plot_results(self, k_values, accuracy, f1):
        mean_accuracy = np.mean(accuracy, axis=1)
        mean_f1 = np.mean(f1, axis=1)
        accuracy_std = np.std(accuracy, axis=1)
        f1_std = np.std(f1, axis=1)

        plt.plot(k_values, mean_accuracy)
        plt.errorbar(k_values, mean_accuracy, yerr=accuracy_std, fmt='D-k', capsize=3)
        plt.xlabel('Value of k')
        plt.ylabel('Accuracy')
        plt.xlim(0, 52)
        plt.title(self.name + ' Accuracy')
        plt.savefig('./figures/' + self.name + '_accuracy graph')
        plt.clf()

        plt.plot(k_values, mean_f1)
        plt.errorbar(k_values, mean_f1, yerr=f1_std, fmt='D-k', capsize=3)
        plt.xlabel('Value of k')
        plt.ylabel('F1 score')
        plt.xlim(0, 52)
        plt.title(self.name + ' F1')
        plt.savefig('./figures/' + self.name + '_f1 graph')
        plt.clf()