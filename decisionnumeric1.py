# -*- coding: utf-8 -*-
"""
Created on Thu Oct 12 09:25:27 2017

@author: tcs
"""

import pandas as pd
import numpy as np
from sklearn.cross_validation import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn import tree
datafileNm="/home/tcs/Desktop/1411148/numericpo.csv"
podata=pd.read_csv(datafileNm)
podf=(podata[["material","order"]])
X=podf.values[:,1]
Y=podf.values[:,0]
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.3,random_state=1000)
clf_gini=DecisionTreeClassifier(criterion="gini",random_state=1000,max_depth=5,min_samples_leaf=5)
clf_gini.fit(X_train,Y_train)
#use entropy as criterion
clf_entropy=DecisionTreeClassifier(criterion="entropy",random_state=1000,max_depth=5,min_samples_leaf=5)
clf_entropy.fit(X_train,Y_train)
y_pred=clf_gini.predict(X_test)
y_pred
##to predict entropy accuracy
y_pred_en=clf_entropy.predict(X_test)
y_pred_en
print "Accuracy of gini is ", accuracy_score(Y_test,y_pred)*100          
print "Accuracy of entroyp is ", accuracy_score(Y_test,y_pred_en)*100    
        
