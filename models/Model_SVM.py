# importing the libraries
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import math
from sklearn.model_selection import train_test_split
from sklearn import svm
import pickle

# importing data set
dataset = pd.read_csv('C:/Users/Kharisamy/Desktop/iris/dataset/Iris.csv')
X=dataset.iloc[:,0:4]
y = dataset.iloc[:,4:]

# checking for null values if any
dataset.isnull().any()

# Train Support Vector Machine (SVM) model with all data  
# Fitting the Model 
svmModel = svm.SVC(kernel='poly', degree=3, C=1.0).fit(X, y)

# Persist model so that it can be used by different consumers
svmFile = open('IrisModel_svm.pckl', 'wb')
pickle.dump(svmModel, svmFile)
svmFile.close()