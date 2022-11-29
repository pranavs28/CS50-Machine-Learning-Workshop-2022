import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import *
from sklearn.metrics import r2_score


df = pd.read_csv("data.csv")

inputs = df.iloc[:,:-1] #remove the sales outcomes from the input data

targets = df.iloc[:,-1:] #get sales outputs

#this is a new edit to my file - 2:57PM