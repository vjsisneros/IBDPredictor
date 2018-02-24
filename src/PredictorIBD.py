# Python Project Template
# 1. Prepare Problem
# a) Load libraries
# b) Load dataset
# 2. Summarize Data
# a) Descriptive statistics
# b) Data visualizations
# 3. Prepare Data
# a) Data Cleaning
# b) Feature Selection
# c) Data Transforms
# 4. Evaluate Algorithms
# a) Split-out validation dataset
# b) Test options and evaluation metric
# c) Spot Check Algorithms
# d) Compare Algorithms
# 5. Improve Accuracy
# a) Algorithm Tuning
# b) Ensembles
# 6. Finalize Model
# a) Predictions on validation dataset
# b) Create standalone model on entire training dataset
# c) Save model for later use
# 7.
# a) Evaluate model
"""
Predicting IBD Type
UW Tacoma
TCSS499 Undergraduate Research

@author: Vidal Sisneros
@version: 1.0 RandomForestClassifier
@date: 1/20/2018
"""

# 1. Prepare Problem
# a) Load libraries
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn import preprocessing as pp
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, roc_curve
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
import xgboost as xgb
import time
import platform

print(platform.architecture() )

# b) Load dataset

start = time.time()

otu_txt = 'dgerver_cd_otu.txt'
mapfile_txt = 'mapfile_dgerver.txt'
ibd_type = 'gastrointest_disord'

otu_df = pd.read_table(otu_txt, sep='\t', index_col=0, skiprows=1)
metadata_df = pd.read_table(mapfile_txt, sep='\t', index_col=0)
#otu_df.to_csv('otu.csv', sep=',')
#metadata_df.to_csv('meta.csv', sep=',')

#Transpose and merge 
otu_df_trans = otu_df.transpose()
merge_df = pd.concat([otu_df_trans, metadata_df[ibd_type]], axis=1 , join='inner')
#otu_df_trans.to_csv('otu_trans.csv', sep=',')
#merge_df.to_csv('merge.csv', sep=',')

#droping columns
X = merge_df.drop([ibd_type], axis=1)
y = merge_df[ibd_type]
#X.to_csv("X.csv", sep=',')
#y.to_csv("y.csv", sep=',')

#Encoding Data
y = y.replace(to_replace=['CD','UC','IC','no'], value=['IBD','IBD','IBD','noIBD'])
encoder = pp.LabelEncoder()
#y.to_csv("y_rep.csv", sep=',')

y = pd.Series(encoder.fit_transform(y), index=y.index, name=y.name)
#y.to_csv("y_en.csv", sep=',')
#print(y) 

# Model Training----------------------------------------------------------------------

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42, shuffle=True)	

x = xgb.XGBClassifier(objective='binary:logistic', n_estimators=10, seed=123)
x.fit(x_train, y_train)
pred = x.predict(x_test)
accuracy = float(np.sum(pred==y_test))/y_test.shape[0]
print("accuracy: %f" % (accuracy))

forest_clf = RandomForestClassifier(n_estimators=7000,random_state=0,criterion='entropy',min_samples_split=20)
forest_clf.fit(x_train, y_train.values.ravel())
print ("Accuracy of Random Forest Classifier: "+str(forest_clf.score(x_test,y_test)))





#svc_clf = SVC(kernel='rbf',C=1, gamma=0.001,random_state=0,probability=True).fit(A, Y.values.ravel())
#print ("Accuracy of SVM: "+str(svc_clf.score(P,Q)))

end = time.time()
print("Time in seconds: %.2f" , (end - start))

#cls = 0
#
#figsize=(30,15)
#f, ax = plt.subplots(1, 1, figsize=figsize)
#
#params = [(forest_clf,'red',"Random Forest"), (svc_clf,'black',"SVM")]
#
#for x in params:
#    y_true = Q[Q.argsort().index]
#    y_prob = x[0].predict_proba(P.ix[Q.argsort().index,:])
#    fpr, tpr, _ = roc_curve(y_true, y_prob[:, cls], pos_label=cls)
#    roc_auc = roc_auc_score(y_true == cls, y_prob[:, cls])
#    ax.plot(fpr, tpr, color=x[1], alpha=0.8,
#    label='Test data: {} '
#    '(auc = {:.2f})'.format(x[2] ,roc_auc))
#
#
#ax.set_xlabel('False Positive Rate',fontsize=30)
#ax.set_ylabel('True Positive Rate',fontsize=30)
#ax.legend(loc="lower right",fontsize=30)
#ax.tick_params(axis='x', labelsize=30)
#ax.tick_params(axis='y', labelsize=30)

#____________Learning what a label Encoder does

#le = pp.LabelEncoder()
#cities = ["paris", "paris", "tokyo", "amsterdam"]
#fitted = le.fit(cities)
#
#
#print(le.fit(cities))
#print(list(le.classes_))
#print(le.transform(cities))
#
#print(type(le.transform(cities)))
#
#trans = le.fit_transform(cities)
#
#print(le.inverse_transform(trans))
#print(type(le.fit_transform(cities)))
#print(le.transform(cities))

#['amsterdam', 'paris', 'tokyo']
#le.transform(["tokyo", "tokyo", "paris"]) 
#array([2, 2, 1]...)
#list(le.inverse_transform([2, 2, 1]))
#['tokyo', 'tokyo', 'paris']

print("Done")