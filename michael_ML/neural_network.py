import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import make_column_transformer
import matplotlib.pyplot as plt
import random
import math

class Neural_Network:

    def __init__(self, name, feature_data, feature_type, class_data, class_col) -> None:
        self.name = name

        print('Starting neural network')
        print('Preprocessing data...')

        categorical_col, numerical_col = self.identify_col(feature_type)
        numerical_data = feature_data[numerical_col]
        categorical_data = feature_data[categorical_col]
        
        normalized_numerical_data = np.array(self.normalization(numerical_data).fillna(0))
        vectorized_numerical_data = np.expand_dims(normalized_numerical_data, axis=1)

        categorical_data_transformer = make_column_transformer((OneHotEncoder(sparse_output=False), categorical_col), remainder='passthrough')
        categorical_data_transformed = categorical_data_transformer.fit_transform(categorical_data[categorical_col])
        vectorized_categorical_data = np.expand_dims(categorical_data_transformed, axis=1)

        label_transformer = make_column_transformer((OneHotEncoder(sparse_output=False), [class_col]), remainder='passthrough')
        label_transformed = label_transformer.fit_transform(class_data)

        self.data = np.concatenate((vectorized_numerical_data, vectorized_categorical_data), axis=2)
        self.labels = np.expand_dims(label_transformed, axis=1)

        self.input_size = len(self.data[0][0])
        self.output_size = len(self.labels[0][0])

        stratified_data, stratified_label = self.k_fold_stratified(10)

        training_data, testing_data, training_label, testing_label = self.split_data(0.3)

        print('Finding best parameters for', self.name, 'dataset...')

        best_alpha, best_lamb, best_epsilon, best_layers_num, best_neurons_per_layer = self.find_best_parameters(stratified_data, stratified_label)

        print('Found best parameters for', self.name, 'dataset')
        print('Begin learning curve...')

        self.train(training_data, testing_data, training_label, testing_label, best_alpha, best_lamb, best_epsilon, best_layers_num, best_neurons_per_layer)

        print('Done with learning curve')

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
    
    def split_data(self, test_size):
        combined_data_label = list(zip(self.data, self.labels))
        random.shuffle(combined_data_label)

        data = []
        label = []

        for i in range(len(combined_data_label)):
            data.append(combined_data_label[i][0])
            label.append(combined_data_label[i][1])

        split_index = int(len(data) * test_size)

        training_data = np.array(data[split_index:])
        training_label = np.array(label[split_index:])

        testing_data = np.array(data[:split_index])
        testing_label = np.array(label[:split_index])

        return training_data, testing_data, training_label, testing_label
    
    def k_fold_stratified(self, k):
        data_folds = [[] for _ in range(k)]
        label_folds = [[] for _ in range(k)]
        data_group = [[] for _ in range(self.output_size)]
        label_group = [[] for _ in range(self.output_size)]

        for i in range(len(self.data)):
            class_index = np.where(self.labels[i][0] == 1)[0][0]
            data_group[class_index].append(self.data[i])
            label_group[class_index].append(self.labels[i])

        for j in range(len(data_group)):
            fold_size = math.ceil(len(data_group[j]) / k)
            for n in range(k):
                start_index = n * fold_size
                end_index = start_index + fold_size
                data_folds[n].extend(data_group[j][start_index : end_index])
                label_folds[n].extend(label_group[j][start_index : end_index])

        return data_folds, label_folds
    
    def get_class_labels(self, dataset, label):
        class_label = set()
        for index, row in dataset.iterrows():
            class_label.add(row.get(label))
        return np.array(list(class_label))

    def generate_gradients(self):
        for i in range(len(self.weight_matrix)):
            self.gradients.append(np.zeros_like(self.weight_matrix[i]))

    def generate_weight(self, x, y):
        wt = []
        for i in range(x * y):
            wt.append(random.uniform(-1, 1))
        return np.array(wt, dtype='float64').reshape(x, y)

    def generate_weight_matrix(self, input_size, out_size, layers_num, neurons_num):
        weight_matrix = []

        weight_matrix.append(self.generate_weight(input_size + 1, neurons_num[0]))

        for i in range(layers_num - 1):
            weight_matrix.append(self.generate_weight(neurons_num[i] + 1, neurons_num[i + 1]))

        weight_matrix.append(self.generate_weight(neurons_num[-1] + 1, out_size))

        return weight_matrix

    def generate_hidden(self, layers_num, neurons_num):
        layers = []
        for i in range(layers_num):
            layers.append(np.array([[-1 for _ in range(neurons_num[i])]], dtype='float64'))
        return layers
    
    def generate_delta(self):
        for i in range(len(self.hidden_layers)):
            self.delta_matrix.append(np.zeros_like(self.hidden_layers[i]))
        
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
    
    def cost_J(self, predict, label):
        return -label * np.log(predict) - (1 - label) * np.log(1 - predict)

    def final_J(self, costs, lamb, data_size):
        return (np.sum(costs) / data_size) + self.regularized(lamb)
    
    def regularized(self, lamb):
        sum_S = 0
        for i in range(len(self.weight_matrix)):
            sum_S += np.sum(np.square(self.weight_matrix[i]))
        return (lamb / (2 * len(self.data))) * sum_S

    def average_gradient(self, data_size, lamb):
        for i in range(len(self.gradients)):
            # When computing P matrix, set the weight for all bias to be all 0, not the weight matrix
            weights = self.weight_matrix[i]
            weights[0] = 0
            self.gradients[i] = (1 / data_size) * (self.gradients[i] + lamb * weights)

    def update_weights(self, alpha):
        for i in range(len(self.weight_matrix)):
            self.weight_matrix[i] = self.weight_matrix[i] - alpha * self.gradients[i]

    def forward_propagation(self, data, example):
        for i in range(len(self.hidden_layers)):
            if i == 0:
                a = np.insert(data, 0, 1, axis=1)
                z = a.dot(self.weight_matrix[i])
            else:
                a = np.insert(self.hidden_layers[i - 1], 0, 1, axis=1)
                z = a.dot(self.weight_matrix[i])
            self.hidden_layers[i] = self.sigmoid(z)

            if example:
                print('a:', a)
                print('z:', z)

        a = np.insert(self.hidden_layers[-1], 0, 1, axis=1)
        z = a.dot(self.weight_matrix[-1])
        self.output_layer = self.sigmoid(z)

        if example:
            print('a:', a)
            print('z:', z)
            print()

    def backpropagation(self, data, label, example):
        output_delta = self.output_layer - label

        if example:
            print('Output delta:', output_delta)

        for i in range(len(self.delta_matrix) - 1, -1, -1):
            hidden_bias = np.insert(self.hidden_layers[i], 0, 1, axis=1)
            if i == len(self.hidden_layers) - 1:
                self.delta_matrix[i] = np.multiply((self.weight_matrix[i + 1].dot(output_delta.transpose())).transpose(), np.multiply(hidden_bias, 1 - hidden_bias))[:, 1:]
            else:
                self.delta_matrix[i] = np.multiply((self.weight_matrix[i + 1].dot(self.delta_matrix[i + 1].transpose())).transpose(), np.multiply(hidden_bias, 1 - hidden_bias))[:, 1:]
            
            if example:
                print('Delta:', self.delta_matrix[i])
        
        if example:
            print()

        for n in range(len(self.hidden_layers) - 1, -2, -1):
            if n == len(self.hidden_layers) - 1:
                hidden_bias = np.insert(self.hidden_layers[n], 0, 1, axis=1)
                gradient = hidden_bias.transpose().dot(output_delta)
            elif n == -1:
                data_bias = np.insert(data, 0, 1, axis=1)
                gradient = data_bias.transpose().dot(self.delta_matrix[n + 1])
            else:
                hidden_bias = np.insert(self.hidden_layers[n], 0, 1, axis=1)
                gradient = hidden_bias.transpose().dot(self.delta_matrix[n + 1])
            
            if example:
                print('Gradient: \n', gradient.transpose())
            
            self.gradients[n + 1] += gradient
        
        if example:
            print()
    
    def find_best_parameters(self, data, label):
        alphas = [0.6, 0.7, 0.8]

        lambs = [0, 0.1, 0.25]

        epsilons = [math.pow(math.e, -7)]

        layers_nums = [1,2,3]
        neurons_per_layer = [[[4], [6]],
                             [[2, 2], [4, 4]],
                             [[4, 4, 4]]]
        
        best_alpha = -1
        best_lamb = -1
        best_epsilon = -1
        best_layers_num = -1
        best_neurons_per_layer = []

        best_accuracy = -1
        best_f1 = -1

        table = pd.DataFrame([], columns=['alpha', 'lamba', 'epsilon', 'number of hidden layers', 'number of neurons for hidden layers', 'accuracy', 'F1 score'])
        
        for alpha in alphas:
            for lamb in lambs:
                for epsilon in epsilons:
                    for i in range(len(layers_nums)):
                        possible_neurons = neurons_per_layer[i]
                        for neurons in possible_neurons:
                            print()
                            print('alpha:', alpha)
                            print('lambda:', lamb)
                            print('epsilon:', epsilon)
                            print('neurons:', neurons)
                            print()

                            self.hidden_layers = self.generate_hidden(layers_nums[i], neurons)

                            self.weight_matrix = self.generate_weight_matrix(self.input_size, self.output_size, layers_nums[i], neurons)

                            if self.output_size <= 2:
                                self.output_layer = np.array([[-1 for _ in range(1)]], dtype='float64')
                            else:
                                self.output_layer = np.array([[-1 for _ in range(self.output_size)]], dtype='float64')

                            self.delta_matrix = []
                            self.generate_delta()

                            self.gradients = []
                            self.generate_gradients()

                            accuracy = []
                            f1 = []

                            for n in range(len(data)):
                                training = data[:n] + data[n + 1:]
                                training = [element for sub_array in training for element in sub_array]
                                training_label = label[:n] + label[n + 1:]
                                training_label = [element for sub_array in training_label for element in sub_array]

                                testing = data[n]
                                testing_label = label[n]

                                true_predictions = [0 for _ in range(self.output_size)]
                                false_predictions = [0 for _ in range(self.output_size)]

                                train = True
                                prev_J = -1

                                while train:
                                    costs = 0

                                    for j in range(len(training)):
                                        self.forward_propagation(training[j], False)
                                        costs += self.cost_J(self.output_layer, training_label[j])
                                        self.backpropagation(training[j], training_label[j], False)

                                    self.average_gradient(len(training), lamb)
                                    self.update_weights(alpha)

                                    final_J = self.final_J(costs, lamb, len(training))

                                    if prev_J == -1:
                                        prev_J = final_J
                                    else:
                                        if abs(final_J - prev_J) <= epsilon:
                                            train = False
                                        else:
                                            prev_J = final_J

                                for k in range(len(testing)):
                                    self.forward_propagation(testing[k], False)
                                    output_index = self.predict_output()
                                    actual_index = np.where(testing_label[k][0] == 1)[0][0]

                                    if output_index == actual_index:
                                        true_predictions[output_index] += 1
                                    else:
                                        false_predictions[output_index] += 1
                                    
                                result_accuracy, result_f1 = self.compute_stats(true_predictions, false_predictions, len(testing))

                                accuracy.append(result_accuracy)
                                f1.append(result_f1)
                            
                            mean_accuracy = np.mean(accuracy)
                            mean_f1 = np.mean(f1)

                            new_row = {'alpha': alpha,
                                        'lamba': lamb,
                                        'epsilon': f"1e{int(math.log(epsilon))}",
                                        'number of hidden layers': layers_nums[i],
                                        'number of neurons for hidden layers': neurons,
                                        'accuracy': mean_accuracy,
                                        'F1 score': mean_f1}

                            table.loc[len(table)] = new_row

                            if mean_accuracy >= best_accuracy and mean_f1 >= best_f1:
                                best_alpha = alpha
                                best_lamb = lamb
                                best_epsilon = epsilon
                                best_layers_num = layers_nums[i]
                                best_neurons_per_layer = neurons
                                best_accuracy = mean_accuracy
                                best_f1 = mean_f1
    
        table.to_csv('./tables/' + self.name + '_table.csv', index=False)

        filename = self.name + '_parameters.txt'
        file_path = './parameters/' + filename
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(f"Alpha: {best_alpha} \n")
            f.write(f"Lambda: {best_lamb} \n")
            f.write(f"Epsilon: 1e{int(math.log(best_epsilon))} \n")
            f.write(f"Number of hidden layer: {best_layers_num} \n")
            f.write(f"Number of neurons in hidden layers: {best_neurons_per_layer} \n")
            f.write(f"Accuracy: {best_accuracy} \n")
            f.write(f"F1 score: {best_f1} \n")
            
        return best_alpha, best_lamb, best_epsilon, best_layers_num, best_neurons_per_layer

    def train(self, training_data, testing_data, training_label, testing_label, alpha, lamb, epsilon, layers_num, neurons_per_layer):
        steps = int(len(training_data) / 20)

        x = steps
        prev_x = 0

        x_values = []
        J_values = []

        self.hidden_layers = self.generate_hidden(layers_num, neurons_per_layer)
        self.weight_matrix = self.generate_weight_matrix(self.input_size, self.output_size, layers_num, neurons_per_layer)

        if self.output_size <= 2:
            self.output_layer = np.array([[-1 for _ in range(1)]], dtype='float64')
        else:
            self.output_layer = np.array([[-1 for _ in range(self.output_size)]], dtype='float64')

        self.delta_matrix = []
        self.generate_delta()

        self.gradients = []
        self.generate_gradients()

        while x < len(training_data):
            print()
            print('number of instances:', x)
            print()
            
            train = True
            training = training_data[prev_x : x]
            label = training_label[prev_x : x]
            prev_J = -1

            while train:
                costs = 0

                for i in range(len(training)):
                    self.forward_propagation(training[i], False)
                    costs += self.cost_J(self.output_layer, label[i])
                    self.backpropagation(training[i], label[i], False)
                
                self.average_gradient(len(training), lamb)
                self.update_weights(alpha)
                final_J = self.final_J(costs, lamb, len(training))

                if prev_J == -1:
                    prev_J = final_J
                else:
                    if abs(final_J - prev_J) <= epsilon:
                        train = False
                    else:
                        prev_J = final_J
            
            performance_costs = 0
                
            for j in range(len(testing_data)):
                self.forward_propagation(testing_data[j], False)
                performance_costs += self.cost_J(self.output_layer, testing_label[j])
            
            performance_J = self.final_J(performance_costs, lamb, len(testing_data))

            x_values.append(min(x, len(training_data)))
            J_values.append(performance_J)

            prev_x = x
            x += steps
        
        plt.plot(x_values, J_values, marker='o')
        plt.xlabel('Number of training samples')
        plt.ylabel('Cost J')
        plt.title(self.name)
        plt.savefig('./figures/' + self.name)
        plt.clf()

    def predict_output(self):
        if self.output_size <= 2:
            if self.output_layer[0][0] > 1 - self.output_layer[0][0]:
                return 0
            else:
                return 1
        else:
            index = -1
            max_output = -1
            for i in range(len(self.output_layer[0])):
                if self.output_layer[0][i] > max_output:
                    index = i
                    max_output = self.output_layer[0][i]
            return index
    
    def compute_stats(self, true_values, false_values, data_size):
        accuracy = np.sum(true_values) / data_size

        if self.output_size == 2:
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

            for i in range(self.output_size):
                if true_values[i] + false_values[i] == 0:
                    precision = 0
                else:
                    precision += true_values[i] / (true_values[i] + false_values[i])
                if true_values[i] + (false_values_sum - false_values[i]) == 0:
                    recall = 0
                else:
                    recall += true_values[i] / (true_values[i] + (false_values_sum - false_values[i]))
            
            precision /= self.output_size
            recall /= self.output_size

        if precision + recall == 0:
            f1 = 0
        else:
            f1 = 2 * (precision * recall) / (precision + recall)
        
        return accuracy, f1