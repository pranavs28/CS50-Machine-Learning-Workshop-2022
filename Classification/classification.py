#tutorial from https://towardsdatascience.com/solving-a-simple-classification-problem-with-python-fruits-lovers-edition-d20ab6b071d2

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier

fruits = pd.read_table('fruitdata.txt')

feature_names = ['mass', 'width', 'height', 'color_score']
X = fruits[feature_names] #remove labels
y = fruits['fruit_label'] #create separate array for labels


training_inputs, testing_inputs, training_targets, testing_targets = train_test_split(X, y, random_state=0) #split into training and testing data

#scale the data to fit the a MinMax range
scaler = MinMaxScaler()
training_inputs = scaler.fit_transform(training_inputs) 
testing_inputs = scaler.transform(testing_inputs)

#train K-Nearest Neighbors Classifier 
knn = KNeighborsClassifier()
knn.fit(training_inputs, training_targets)
print('Accuracy of K-NN classifier on training set: {:.2f}'
     .format(knn.score(training_inputs, training_targets)))
print('Accuracy of K-NN classifier on test set: {:.2f}'
     .format(knn.score(testing_inputs, testing_targets)))





