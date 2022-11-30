import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import *
from sklearn.metrics import r2_score
from matplotlib import pyplot as plt


df = pd.read_csv("data.csv")

inputs = df.iloc[:,:-1] #remove the sales outcomes from the input data

targets = df.iloc[:,-1:] #get sales outputs

training_inputs, testing_inputs, training_targets, testing_targets = train_test_split(inputs, targets, test_size=0.8, random_state=1)

reg = LinearRegression()

reg.fit(training_inputs, training_targets)

x_train_pred = reg.predict(training_inputs)

accuracy = r2_score(training_targets, x_train_pred)
print("3 Attribute Regression Accuracy: " + str(round(accuracy * 100, 2)) + "%")

