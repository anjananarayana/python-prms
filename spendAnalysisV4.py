#################################################
#Spend Analytics - PO Clasification
#
##################################################
import pandas as pd
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
featuresets = [(find_features(row['shortTextClean']), row['Material Group']) for index, row in podf_txt_catg_all.iterrows()]
#fea=np.asarray(featuresets)
print(type(featuresets))

# no of samples are taken as 300 so training set around 250 and 50 for test set
# todo - need write the code to dynamically get the set rows with 80% 20%
training_set = featuresets[:160]
testing_set = featuresets[160:]

print(testing_set)


# Classifier generation
classifier=nltk.NaiveBayesClassifier.train(training_set)
print("Classifier accuracy percentage",(nltk.classify.accuracy(classifier,testing_set)*100))
classifier.show_most_informative_features(5)


tstStr='PO_RASO_VN_IN_MSOPT_003_S MDSP_Hatomi'
tstStrCln=clean(tstStr)
tstVal = find_features(tstStrCln)
print(tstVal)
##tstVal={'3rd': False, 'party': False, 'onsite': False, 'support': False,
##        'service': False, 'april': False, '2': False, 'sigmaots': False,
##        'test': False, 'eng': False}
print('')
print('')
print('...........classification...........', tstStr)
print(classifier.classify(tstVal))

dist=classifier.prob_classify(tstVal)
for label in dist.samples():
    print("%s: %f" % (label,dist.prob(label)*100))



# need sklearn module from python library
#
##MNB_classifier= SklearnClassifier(MultinomialNB())
##MNB_classifier.train(training_set)
##print("Classifier accuracy percentage",(nltk.classify.accuracy(MNB_classifier,testing_set)*100))
