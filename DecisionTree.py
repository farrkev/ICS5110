# code inspired by: https://towardsdatascience.com/implementing-a-decision-tree-from-scratch-f5358ff9c4bb

import math
import random

class DecisionTree:
    def __init__(self):
        self.model = self.TreeNode()
    
    class TreeNode:
        def __init__(self, feature = None, splitPoint = None, leftChild = None, rightChild = None, value = None):
            self.feature = feature
            self.splitPoint = splitPoint
            self.leftChild = leftChild
            self.rightChild = rightChild
            self.value = value

        def is_leaf(self):
            return self.value is not None
    
    def fit(self, x_train:list, y_train:list, depth:int = 0):
        max_depth = 200
        min_samples_split = 3

        def _categorical_count(X:list) -> list:
            '''
            Returns counts of values in a list from 0 to max(X).
            '''

            counts = {}
            for value in X:
                if value in counts:
                    counts[value] += 1
                else:
                    counts[value] = 1

            return [counts.get(i, 0) for i in range(max(counts.keys()) + 1)]

        def _entropy(y:list) -> float:
            '''
            Entropy is a metric that quantifies the amount of disorder or unpredictability in a system.
            The objective in the model is to minimize this uncertainty.
            '''

            classes = list(set(y))
            class_count = {i:0 for i in classes}

            for cls in y:
                class_count[cls] += 1

            class_counts = list(class_count.values())
            prob = [count / sum(class_counts) for count in class_counts]
            entropy_value = -sum([i * math.log2(i) for i in prob])

            return entropy_value

        def _split_data(X:list, split_point) -> (list, list):
            left_node_index = []
            right_node_index = []

            for i, x in enumerate(X):
                if x <= split_point:
                    left_node_index.append(i)
                else:
                    right_node_index.append(i)

            return left_node_index, right_node_index

        def _calculate_information_gain(X:list, y:list, split_point) -> float:
            '''
            The information gain is used to measure how much information a feature provides. 
            1 indicates most information gained and 0 indicates that no information was gained. 
            To calculate the split's information gain, the sum of weighted entropies of the 
            childen are computed and subtracted from the parent's entropy.

            Returns foat number `ig` in range 0 < ig < 1.
            '''

            left_node_index, right_node_index = _split_data(X, split_point)

            if not left_node_index or not right_node_index:
                return 0

            parent_entropy = _entropy(y)
            left_entropy = _entropy([y[lni] for lni in left_node_index])
            right_entropy = _entropy([y[rni] for rni in right_node_index])
            child_entropy = (len(left_node_index) / len(y)) * left_entropy + (len(right_node_index) / len(y)) * right_entropy
            information_gain = parent_entropy - child_entropy

            return information_gain

        def _get_best_split(X:list, y:list, features:list):
            best_feature = None
            best_split_point = None
            best_gain = -1

            for feature in features:
                values = [i[feature] for i in X]
                unique_values = list(set(values))

                for split_point in unique_values:
                    gain = _calculate_information_gain(values, y, split_point)

                    if gain > best_gain:
                        best_gain = gain
                        best_feature = feature
                        best_split_point = split_point

            return best_feature, best_split_point

        def _tree_builder(x_train:list, y_train:list, depth:int = 0):
            n_samples = len(x_train)
            n_features = len(x_train[0])

            if (depth >= max_depth) or (len(set(y_train)) == 1) or (n_samples < min_samples_split):
                class_counts = _categorical_count(y_train)
                most_common_label = class_counts.index(max(class_counts))
                return DecisionTree.TreeNode(value = most_common_label)

            randomised_feature_indexes = random.sample(list(range(n_features)), n_features)
            best_feat, best_split = _get_best_split(x_train, y_train, randomised_feature_indexes)

            left_node_index, right_node_index = _split_data([i[best_feat] for i in x_train], best_split)
            left_child = _tree_builder([x_train[lni] for lni in left_node_index], [y_train[lni] for lni in left_node_index], depth + 1)
            right_child = _tree_builder([x_train[rni] for rni in right_node_index], [y_train[rni] for rni in right_node_index], depth + 1)

            return DecisionTree.TreeNode(best_feat, best_split, left_child, right_child)
        
        self.model = _tree_builder(x_train, y_train)
        
    def predict(self, x_test:list) -> list:
        '''
        To make a prediction the tree is recursively traversed. For every sample in the dataset, 
        the node feature and split point values are compared to the current sample values and a 
        decision is made whether to take a left or right turn. Once a leaf node is reached the most 
        common class label is returned.
        '''
        
        predictions = []

        for x in x_test:
            treeNode = self.model
            
            while not treeNode.is_leaf():
                if x[treeNode.feature] <= treeNode.splitPoint:
                    treeNode = treeNode.leftChild
                else:
                    treeNode = treeNode.rightChild
            predictions.append(treeNode.value)
            
        return predictions