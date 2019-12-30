# importing the libraries
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import math
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import KNeighborsClassifier as KNN 
import pickle

# importing data set
dataset = pd.read_csv('C:/Users/Kharisamy/Desktop/iris/dataset/Iris.csv')
X=dataset.iloc[:,0:4]
y = dataset.iloc[:,4:]

# checking for null values if any
dataset.isnull().any()

knn = KNN(n_neighbors = 3) 
# Fitting the Model 
regressor = KNN()
model=knn.fit(X,y)


# Persist model so that it can be used by different consumers
regressorFile = open('IrisModel_knn.pckl', 'wb')
pickle.dump(model, regressorFile)
regressorFile.close()




