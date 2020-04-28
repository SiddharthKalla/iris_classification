# -*- coding: utf-8 -*-
"""
Created on Tue Apr 28 00:50:55 2020

@author: leno
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import datasets
import seaborn as sns

dataset = pd.read_csv('Iris.csv')
X = dataset.iloc[:,1:5].values
y = dataset.iloc[:,5].values

sns.boxplot(x=y, y=X[:,0])

from sklearn.preprocessing import LabelEncoder,OneHotEncoder
labelencoder_y = LabelEncoder()
y = labelencoder_y.fit_transform(y)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.3,random_state = 0)

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train,y_train)

y_pred = regressor.predict(X_test)

print(regressor.score(X_test,y_test))

