import os
from sklearn.feature_extraction import text
from sklearn import ensemble 
import pandas as pd
import numpy as np
import re
import nltk
from bs4 import BeautifulSoup
from nltk import corpus
from sklearn.model_selection import cross_val_score

from sklearn.naive_bayes import MultinomialNB


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



os.chdir("/home/tcs/Documents")
    
#movie_train = pd.read_csv("labeledTrainData.tsv", header=0, 
#                    delimiter="\t", quoting=3)
movie_train=pd.read_csv("podata.csv")
type (movie_train)
movie_train.shape
movie_train.isnull()
movie_train.info()
movie_train.loc[0:4,'Short Text']


vectorizer = text.CountVectorizer(preprocessor = preprocess_review, \
                                  tokenizer = tokenize,    \
                                  stop_words = 'english',   \
                                  max_features = 6000)

#features = vectorizer.fit_transform(movie_train.loc[0:3,'Description']).toarray()
features = vectorizer.fit_transform(movie_train.loc[0:20156,'Short Text']).toarray()


vectorizer.get_stop_words()
vectorizer.vocabulary_
vocab = vectorizer.get_feature_names()


forest = ensemble.RandomForestClassifier(n_estimators = 100) 
#forest = MultinomialNB()

# Fit the forest to the training set, using the bag of words as 
# features and the sentiment labels as the response variable

#forest = forest.fit(features, movie_train['Service'] )
forest = forest.fit(features, movie_train.loc[0:2100,'Material Group'] )


#######################################################
X_train = features
y_train = movie_train.loc[0:20156,'Material Group']

scores = cross_val_score(forest, X_train, y_train, cv = 10)
scores.mean()
scores.std()





