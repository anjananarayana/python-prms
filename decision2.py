# -*- coding: utf-8 -*-
"""
Created on Tue Oct 17 11:02:32 2017

@author: tcs
"""

import pandas as pd
import numpy as np
import sklearn.utils
import sys
from class_vis import prettyPicture,output_image
from sklearn.cross_validation import train_test_split
import matplotlib.pyplot as plt
import pylab as pl
from classifyDT import classify
import nltk
from nltk.corpus import stopwords 
from nltk.stem.wordnet import WordNetLemmatizer
import string
features_train,labels_train,features_test,labels_test=makeTerrainData()



#input data file
datafileNm="/home/tcs/Desktop/1411148/podata.csv"
podf=pd.read_csv(datafileNm)
##print(podf.head(5))

podf_txt=(podf[['Short Text','Material Group']])
podf_cat_lst=podf_txt['Material Group'].tolist()
podf_cat_lst=set(podf_cat_lst)

stop = set(stopwords.words('english'))
exclude = set(string.punctuation) 
lemma = WordNetLemmatizer()

# clean document
def clean(doc):
    stop_free = " ".join([i for i in doc.lower().split() if i not in stop])
    punc_free = ''.join(ch for ch in stop_free if ch not in exclude)
    normalized = " ".join(lemma.lemmatize(word) for word in punc_free.split())
    return normalized

# define features
def find_features(document):
    words = set(word.lower() for word in document)
    features = {}
    for w in word_features:
        features[w] = (w in words)
    return features


     
podf_txt1=podf_txt['Short Text'].sample(n=5000)
podf_txt_lst=podf_txt1.tolist()
##    print(podf_txt1.head(5).tolist())

doc_clean = [clean(doc).split() for doc in podf_txt_lst]
##print(type(doc_clean))

# get all the words
all_words = []
for stmt in doc_clean:
    for word in stmt:
        all_words.append(word.lower())

all_words = nltk.FreqDist(all_words)
##print(all_words.most_common(15))

# identify the word features from all the class
word_features= list(all_words.keys())[:50]
print('...word features printed ....')
print(word_features)
print('')



# build the data set for feature set creation for algorithm
podf_txt_catg_all=pd.DataFrame()

for catg in podf_cat_lst:
    podf_txt_catg=podf_txt[podf_txt['Material Group'].isin({catg})]
    podf_txt_catg=podf_txt_catg.sample(n=500)
    podf_txt_catg_all=podf_txt_catg_all.append(podf_txt_catg)

podf_txt_catg_all['shortTextClean']=podf_txt_catg_all['Short Text'].apply(lambda x: clean(x))
podf_txt_catg_all=sklearn.utils.shuffle(podf_txt_catg_all)
##print(podf_txt_catg_all)

#debug statement
##print([(find_features(row['shortTextClean']), row['Material Group']) for index, row in podf_txt_catg_all.iterrows()])
featuresets = [(find_features(row['shortTextClean']), row['Material Group']) for index, row in podf_txt_catg_all.iterrows()]

print(type(featuresets))

# no of samples are taken as 300 so training set around 250 and 50 for test set
# todo - need write the code to dynamically get the set rows with 80% 20%
training_set = featuresets[:240]
testing_set = featuresets[240:]

features_train,labels_train,features_test,labels_test=train_test_split(training_set,testing_set,test_size=0.3,random_state=100)
features_train,labels_train,Y_train,Y_test=train_test_split(X,Y,test_size=0.3,random_state=1000)