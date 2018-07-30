# -*- coding: utf-8 -*-import pandas as pd
import numpy as np
import sklearn.utils
from nltk.classify.scikitlearn import SklearnClassifier
##from sklearn.navie_bayes import MultinomialNB, BernoulliNB
import nltk
from nltk.corpus import stopwords 
from nltk.stem.wordnet import WordNetLemmatizer
import string


import sys
#reload(sys)
#sys.setdefaultencoding('utf8') 


#input data file
datafileNm="D:\\tcs\\csv\\podata.csv"
podf=pd.read_csv(datafileNm)
##print(podf.head(5))

podf_txt=(podf[['Short Text','Material Group']])
podf_cat_lst=podf_txt['Material Group'].tolist()
podf_cat_lst=set(podf_cat_lst)
y=podf.loc[0:20156,'Material Group']

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


     
podf_txt1=podf_txt['Short Text']
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
print(all_words.most_common(15))

# identify the word features from all the class
word_features= list(all_words.keys())[:70]
print('...word features printed ....')
print(word_features)
print('')



# build the data set for feature set creation for algorithm
podf_txt_catg_all=pd.DataFrame()

for catg in podf_cat_lst:
    podf_txt_catg=podf_txt[podf_txt['Material Group'].isin({catg})]
    podf_txt_catg=podf_txt_catg.sample(n=70)
    podf_txt_catg_all=podf_txt_catg_all.append(podf_txt_catg)

podf_txt_catg_all['shortTextClean']=podf_txt_catg_all['Short Text'].apply(lambda x: clean(x))
podf_txt_catg_all=sklearn.utils.shuffle(podf_txt_catg_all)
print(podf_txt_catg_all)

#debug statement
print([(find_features(row['shortTextClean']), row['Material Group']) for index, row in podf_txt_catg_all.iterrows()])
featuresetss = [(find_features(row['shortTextClean']), row['Material Group']) for index, row in podf_txt_catg_all.iterrows()]

fea=np.asarray(featuresets)
df = pd.DataFrame(fea, columns=[word_features])


dff = pd.DataFrame(fea,y)
#converting the df dataframe into excel in the formate of csv
df.to_csv("D:\\tcs\\csv\\features_short1.csv")
X_train,X_test,Y_train,Y_test=train_test_split(fea,y,test_size=0.3,random_state=1000)
clf_gini=DecisionTreeClassifier(criterion="gini",random_state=1000,max_depth=5,min_samples_leaf=5)
clf_gini.fit(X_train,Y_train)
clf_entropy=DecisionTreeClassifier(criterion="entropy",random_state=1000,max_depth=5,min_samples_leaf=5)
clf_entropy.fit(X_train,Y_train)
y_pred=clf_gini.predict(X_test)
y_pred
##to predict entropy accuracy
y_pred_en=clf_entropy.predict(X_test)
y_pred_en
ac=accuracy_score(Y_test,y_pred)*100          
print "Accuracy of entroyp is ", accuracy_score(Y_test,y_pred_en)*100    
#to visulaize the plot in the formate of pdf with dot xml file 
with open("clf_gini.dot", "w") as f:
    f = tree.export_graphviz(clf_gini, out_file=f)
#to visulie online with the text file in our current document where the programme has executed
    #this online tree for the method of gini index
with open("clf_gini.txt", "w") as f:
    f = tree.export_graphviz(clf_gini, out_file=f) 