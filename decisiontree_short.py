import os
from sklearn.feature_extraction import text
from sklearn import ensemble 
import pandas as pd
#import numpy as np
import re
#import nltk
from bs4 import BeautifulSoup
from nltk import corpus
from sklearn.model_selection import cross_val_score
from sklearn.cross_validation import train_test_split
from sklearn.cross_validation import train_test_split

from sklearn.metrics import accuracy_score
from sklearn import tree
#from __future__ import print_function
from sklearn.tree import DecisionTreeClassifier, export_graphviz


def preprocess_review(Description):        #
        # 1. Remove HTML
        review_text = BeautifulSoup(Description).get_text()
        #
        # 2. Remove non-letters
        review_text = re.sub("[^a-zA-Z]"," ", review_text)
        #
        # 3. Convert words to lower case
        return review_text.lower()

def stop_words():
    #nltk.download()
    return list(corpus.stopwords.words("english"))
    
def tokenize(Description):
    return Description.split()


datafileNm="D:\\tcs\\csv\\podata.csv"
movie_train=pd.read_csv(datafileNm)

    
#movie_train = pd.read_csv("labeledTrainData.tsv", header=0, 
#                    delimiter="\t", quoting=3)
type (movie_train)
movie_train.shape
movie_train.isnull()
movie_train.info()
movie_train.loc[0:20156,'Short Text']
y=movie_train.loc[0:20156,'Material Group']

vectorizer = text.CountVectorizer(preprocessor = preprocess_review, \
                                  tokenizer = tokenize,    \
                                  stop_words = 'english',   \
                                  max_features = 3000)

#features = vectorizer.fit_transform(movie_train.loc[0:3,'Description']).toarray()
features = vectorizer.fit_transform(movie_train.loc[0:20156,'Short Text']).toarray()
#features_tab=pd.DataFrame(features)
df = pd.DataFrame(features,y)
#converting the df dataframe into excel in the formate of csv
df.to_csv("D:\\tcs\\csv\\features_short.csv")
X_train,X_test,Y_train,Y_test=train_test_split(features,y,test_size=0.3,random_state=1000)
clf_gini=DecisionTreeClassifier(criterion="gini",random_state=1000,max_depth=5,min_samples_leaf=5)
clf_gini.fit(X_train,Y_train)
clf_entropy=DecisionTreeClassifier(criterion="entropy",random_state=1000,max_depth=5,min_samples_leaf=5)
clf_entropy.fit(X_train,Y_train)
y_pred=clf_gini.predict(X_test)
y_pred
##to predict entropy accuracy
y_pred_en=clf_entropy.predict(X_test)
y_pred_en
accuracy_score(Y_test,y_pred)*100          
#print "Accuracy of entroyp is ", accuracy_score(Y_test,y_pred_en)*100    
#to visulaize the plot in the formate of pdf with dot xml file 
with open("clf_gini.dot", "w") as f:
    f = tree.export_graphviz(clf_gini, out_file=f)
#to visulie online with the text file in our current document where the programme has executed
    #this online tree for the method of gini index
with open("clf_gini.txt", "w") as f:
    f = tree.export_graphviz(clf_gini, out_file=f) 