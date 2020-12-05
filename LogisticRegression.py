# -*- coding: utf-8 -*-
"""
Created on Sat Dec  5 14:16:21 2020

@author: Rutger
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression

df = pd.read_csv('Dataset Heart.csv')
df.head()

LogReg = LogisticRegression()

y = df['target']
x = df[['age','sex','cp','trestbps','chol','fbs','restecg','thalach','exang','oldpeak','slope','ca','thal']]

LogReg.fit(x, y)

y_pred = LogReg.predict(x)

score = LogReg.score(x, y)

i = 0
while i < len(y_pred):
    print('Person ', i, ': ', y_pred[i])
    i += 1

print('ROC score: ', score)



