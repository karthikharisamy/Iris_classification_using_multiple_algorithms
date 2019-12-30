# importing the libraries
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import math
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import pickle

# importing data set
dataset = pd.read_csv('C:/Users/Kharisamy/Desktop/iris/dataset/Iris.csv')
X=dataset.iloc[:,0:4]
y = dataset.iloc[:,4:]

# checking for null values if any
dataset.isnull().any()

# Fitting the Model 
regressor = LogisticRegression()
model=regressor.fit(X,y)

# Persist model so that it can be used by different consumers
regressorFile = open('IrisModel_lr.pckl', 'wb')
pickle.dump(model, regressorFile)
regressorFile.close()






