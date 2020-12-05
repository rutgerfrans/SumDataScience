# -*- coding: utf-8 -*-
"""
Created on Sat Dec  5 13:21:40 2020

@author: Rutger
"""
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
import math

df = pd.read_csv('CarPrice_Assignment.csv')
df.head()

y = df['price']
x = df[['wheelbase','carlength','carwidth','carheight','curbweight','enginesize','boreratio','stroke',
        'compressionratio','horsepower','peakrpm','citympg','highwaympg']]

linear_regression = LinearRegression()

linear_regression.fit(x,y)

y_pred = linear_regression.predict(x)


i = 0;
while i < len(y_pred):
    print(df.CarName[i],':\t', y_pred[i])
    i += 1

mse = mean_squared_error(df.price, y_pred)
rtwo = r2_score(df.price, y_pred)

print('mse: ', mse, '\nrmse: ',math.sqrt(mse), '\nr2: ', rtwo)
