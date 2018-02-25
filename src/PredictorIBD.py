"""
Predicting Inflammatory Bowel Disease Type
UW Tacoma
TCSS499 Undergraduate Research

@author: Vidal Sisneros
@version: 1.0 
@date: 1/20/2018
"""

# 1. Prepare Problem
# a) Load libraries
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
from pickle import dump
from pickle import load
#import matplotlib.pyplot as plt

#print(platform.architecture())

# b) Load datasets
otu_txt = 'C://Users//V//Desktop//IBDPredictor//src//data//dgerver_cd_otu.txt'
mapfile_txt = 'C://Users//V//Desktop//IBDPredictor//src//data//mapfile_dgerver.txt'
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


# Model Training----------------------------------------------------------------------
start = time.time()
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42, shuffle=True)	

svm_clf = SVC(probability=True, verbose=True,  random_state=7)
svm_clf.fit(x_train, y_train)
print(svm_clf.get_params)
print("svm accuracy: %f" % (svm_clf.score(x_test, y_test)))
print(svm_clf.predict_proba(x_test))


xgB_clf = xgb.XGBClassifier(objective='binary:logistic', n_estimators=10, seed=123, verbose=True)
xgB_clf.fit(x_train, y_train)
#pred = xgB_clf.predict(x_test)
#accuracy = float(np.sum(pred==y_test))/y_test.shape[0]
print("xgB accuracy: %f" % (xgB_clf.score(x_test, y_test)))


forest_clf = RandomForestClassifier(verbose=True, n_estimators=7000,random_state=0,criterion='entropy',min_samples_split=20)
forest_clf.fit(x_train, y_train.values.ravel())
print ("random forrest accuracy: "+ str(forest_clf.score(x_test,y_test)))

start_time = time.time()

#svc_clf = SVC(kernel='rbf',C=1, gamma=0.001,random_state=0,probability=True).fit(A, Y.values.ravel())
#print ("Accuracy of SVM: "+str(svc_clf.score(P,Q)))

end_time = time.time()
print("Time in seconds: %.2f" % (end_time - start_time))

cls = 0

figsize=(30,15)
f, ax = plt.subplots(1, 1, figsize=figsize)

params = [(forest_clf,'red',"Random Forest"), (svm_clf,'black',"SVM")]

for x in params:
    y_true = Q[Q.argsort().index]
    y_prob = x[0].predict_proba(P.ix[Q.argsort().index,:])
    fpr, tpr, _ = roc_curve(y_true, y_prob[:, cls], pos_label=cls)
    roc_auc = roc_auc_score(y_true == cls, y_prob[:, cls])
    ax.plot(fpr, tpr, color=x[1], alpha=0.8,
    label='Test data: {} '
    '(auc = {:.2f})'.format(x[2] ,roc_auc))

ax.set_xlabel('False Positive Rate',fontsize=30)
ax.set_ylabel('True Positive Rate',fontsize=30)
ax.legend(loc="lower right",fontsize=30)
ax.tick_params(axis='x', labelsize=30)
ax.tick_params(axis='y', labelsize=30)

print("Done")