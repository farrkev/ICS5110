# code inspired by: https://github.com/akshat3492/Multiclass-SVM/blob/main/svm.py

import random

class SupportVectorMachines:
    def __init__(self, alpha:float, epochs:int, C:float = 1.0):
        self.w = None
        self.alpha = alpha
        self.epochs = epochs
        self.C = C
        
        self.class_count = 0
        self.samples_count = 0
        self.dimensions_count = 0

    def _find_gradients(self, x:list, y:list) -> list:
        gradients = [[i * self.C for i in sublist] for sublist in self.w]
        
        # performing matrix dot product multiplication
        epoch_score = []
        for row in x:
            new_row = []
            for column in zip(*self.w):
                total = 0
                for a, b in zip(row, column):
                    total += (a * b)
                new_row.append(total)
            epoch_score.append(new_row)
                
        true_score = [epoch_score[i][j] for i, j in enumerate(y)]
        
        gradient_score = []
        for i in range(len(true_score)):
            new_row = []
            for j in range(len(epoch_score[i])):
                new_row.append(1 if (true_score[i] - epoch_score[i][j]) < 1 else 0)
            gradient_score.append(new_row)
            
        for i, j in enumerate(y):
            gradient_score[i][j] = 0
            
        true_gradient = []
        for i in range(len(gradient_score)):
            true_gradient.append([float(sum(gradient_score[i]) * feature_value) for feature_value in x[i]])

        for i in range(self.samples_count):
            for idx, row in enumerate(gradients):
                row[y[i]] -= true_gradient[i][idx]

            temp = []
            for k in range(len(x[i])):
                result = []
                for j in range(len(gradient_score[i])):
                    result.append(float(x[i][k] * gradient_score[i][j]) + gradients[k][j])
                temp.append(result)
            gradients = temp
        
        return gradients

    def fit(self, x_train:list, y_train:list):
        self.class_count = len(set(y_train))
        self.samples_count = len(x_train)
        self.dimensions_count = len(x_train[0])
               
        self.w = [[random.random() for i in range(self.class_count)] for j in range(self.dimensions_count)]
        
        for epoch in range(self.epochs):
            g = self._find_gradients(x_train, y_train)
            g_a = [[i * self.alpha for i in sublist] for sublist in g]
            self.w = [[x_i - y_i for x_i, y_i in zip(row_x, row_y)] for row_x, row_y in zip(self.w, g_a)]

    def predict(self, x_test:list) -> list:
        predictions = [[sum(a * b for a, b in zip(row_x, col_y)) for col_y in zip(*self.w)] for row_x in x_test]
        return [row.index(max(row)) for row in predictions]