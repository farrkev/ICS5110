class CategoricalNaiveBayes:
    def __init__(self):
        self.model = None
    
    def fit(self, x_train:list, y_train:list) -> dict:
        # building the model

        all_probabilities = []
        
        for feature in [[i[j] for i in x_train] for j in range(len(x_train[0]))]:
            x = feature.copy()
            y = y_train

            categories_x = list(set(x))
            categories_y = list(set(y))
                       
            contingency_matrix = dict()
            for i in range(len(categories_x)):
                for j in range(len(categories_y)):
                    contingency_matrix[(categories_x[i], categories_y[j])] = 0

            for i in zip(x, y):
                contingency_matrix[(int(i[0]), int(i[1]))] += 1

            class_counts = dict()
            for cls in categories_y:
                class_counts[cls] = 0

            for k,v in contingency_matrix.items():
                class_counts[k[1]] += v

            probability_matrix = dict.fromkeys(contingency_matrix, 0)
            probability_matrix

            for k,v in contingency_matrix.items():
                probability_matrix[k] = v / class_counts[k[1]]

            all_probabilities.append(probability_matrix)

        self.model = all_probabilities
    
    def predict(self, x_test:list) -> list:
        # predicting
        
        predictions = []
        
        for row in x_test:
            model_probabilities = []
            feature_probabilities = dict()
            for i in range(len(row)):
                for k,v in self.model[i].items():
                    if k[0] == row[i]:
                        feature_probabilities[(i, k[1])] = v
                model_probabilities.append(feature_probabilities)

            class_probabilities = dict()
            for feature_prob in model_probabilities:
                for k,v in feature_prob.items():
                    if k[1] in class_probabilities.keys():
                        class_probabilities[k[1]] *= v
                    else:
                        class_probabilities[k[1]] = v

            predicted_class = max(class_probabilities, key = class_probabilities.get)
            predictions.append(predicted_class)
            
        return predictions