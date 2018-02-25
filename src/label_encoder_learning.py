# -*- coding: utf-8 -*-
"""
Created on Sat Feb 24 15:52:04 2018

@author: V
"""

import numpy as np
import pandas as pd
#import xgboost as xgb
import time
import platform

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn import preprocessing as pp
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, roc_curve

#____________Learning what a label Encoder does

le = pp.LabelEncoder()
cities = ["paris", "paris", "tokyo", "amsterdam"]
fitted = le.fit(cities)


print(le.fit(cities))
print(list(le.classes_))
print(le.transform(cities))

print(type(le.transform(cities)))

trans = le.fit_transform(cities)

print(le.inverse_transform(trans))
print(type(le.fit_transform(cities)))
print(le.transform(cities))

['amsterdam', 'paris', 'tokyo']
le.transform(["tokyo", "tokyo", "paris"]) 
#array([2, 2, 1])
list(le.inverse_transform([2, 2, 1]))
['tokyo', 'tokyo', 'paris']