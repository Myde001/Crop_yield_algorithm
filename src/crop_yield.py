#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 22 22:49:54 2020

@author: olumideakinola
"""

import numpy as np 
import pandas as pd 

from sklearn.model_selection import train_test_split 

import matplotlib.pyplot as plt
import seaborn as seabornInstance
from sklearn.linear_model import LinearRegression
from sklearn import metrics


#Dataset used : https://daac.ornl.gov/daacdata/global_soil/TopSoil_Erosion_MidWest_US/data/
#read in the dataset

dataset = pd.read_csv('./datatopsoil_loss.csv')

dataset.shape
dataset.describe()

#create the X and Y array

X = dataset[['corn_planted_ac', 'farm_area_mean_ac', 'topsoil_loss_error_frac', 'corn_harvest_ac','corn_econ_loss_usd']].values

Y = dataset['corn_yield_bu_per_ac'].values 

#plot a distribution plot vs Y

plt.figure(figsize = (10,15))
plt.tight_layout()
seabornInstance.distplot(dataset['corn_yield_bu_per_ac'])

#split your dataset
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state=0)


#lets train the model
regressor = LinearRegression()
regressor.fit(X_train, Y_train)

#lets do the prediction
Y_pred = regressor.predict(X_test)

#fix the prediction and actual values into an array
df = pd.DataFrame({'Actual': Y_test, 'Predicted': Y_pred})
df1 = df.head(30)

#Lets see a plot of the actual predctions 
df1.plot(kind='bar',figsize=(10,8))
plt.grid(which='major', linestyle='-', linewidth='0.5', color='green')
plt.grid(which='minor', linestyle=':', linewidth='0.5', color='black')
plt.show()


#lets see the performance of the algorithm 

print('Mean Absolute Error:', metrics.mean_absolute_error(Y_test, Y_pred))  
print('Mean Squared Error:', metrics.mean_squared_error(Y_test, Y_pred))  
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(Y_test, Y_pred)))
     