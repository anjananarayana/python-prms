# -*- coding: utf-8 -*-
"""
Created on Wed Oct 25 12:46:33 2017

@author: tcs
"""
import numpy as np
import pandas as pd
from sklearn.cross_validation import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn import tree
#importing data 
datafileNm="/home/tcs/Desktop/1411148/numericpo.csv"
podf=pd.read_csv(datafileNm)
balance_data=(podf[['Acct Assignment Cat.','Order Unit','Material Group']])
#data slicing
x=balance_data.values[:, 0:2]
y=balance_data.values[:,2]
#let split our model into trainign set and testing set
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=200)
clf_gini=DecisionTreeClassifier(criterion="gini",random_State=100,max_depth=3,min_samples_leaf=5)
clf.gini.fit(x_train,y_train)
