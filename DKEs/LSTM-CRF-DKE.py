## This python file is used to training SVM classifier
## Features : TFIDF

import numpy as np
import os
#from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
import csv
import scipy
from scipy import sparse
import re
import string
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.feature_extraction import stop_words
from nltk.stem.snowball import EnglishStemmer
from string import punctuation
from nltk.corpus import stopwords
from nltk import word_tokenize
from gensim.models import TfidfModel
from gensim.corpora import Dictionary
from sklearn.utils import shuffle
import pickle



stop_words = stopwords.words('english') + list(punctuation)

def tokenize(text):
    words = word_tokenize(text)
    words = [w.lower() for w in words]
    return [w for w in words if w not in stop_words and not w.isdigit()]


## loading dictionary and Pubmed tfidf values

pkl_file = open('tfidf_pubmed_values.pkl', 'rb')
mydict2 = pickle.load(pkl_file)
pkl_file.close()

dictionary_loaded = Dictionary.load("dictionary-tfidf.pkl")

#### Loading positive and negative data 

trainning_data=[]
file1 = open("combinedNegAnn.ann", "r")
mm=[]
for i in file1:
    mm.append(i)
gg=[]
for i in mm:
    p= i.strip('\n')
    gg.append(p)
    #trainning_data.append(p)

file2 = open("combinedPosAnn.ann", "r")
mm1=[]
for i in file2:
    mm1.append(i)
gg1=[]
for i in mm1:
    p1= i.strip('\n')
    gg1.append(p)
    #trainning_data.append(p1)

### Use for lebelling
N1=len(gg)
N2=len(gg1)
ngg=[]                        ### balanceing 
for i in range(len(gg1)):
    ngg.append(gg[i])
N1=len(ngg)
Y1=np.zeros(N1)
Y2=np.ones(N2)

Y=np.concatenate((Y1,Y2),axis=0) ### Lebelling

for i in ngg:
    trainning_data.append(i)
for j in gg1:
    trainning_data.append(j)

print(' Initial data length for Negative samples :   ')
print(len(gg))
print(' positive samples : ')
print(len(gg1))
print('After Balancing :   ')
print('positive samples:', len(gg1))
print('Negative samples:', len(ngg))
############ Testing #####
X=[]
for i in trainning_data:
    tt=tokenize(i)
    wordv = 0
    for j in tt:
        try:
            oo = mydict2[dictionary_loaded.token2id[i]]
        except:
            oo = 0
        wordv += oo
    X.append(wordv)


### shuffling

data, label = shuffle(X, Y, random_state=2)
data1=np.expand_dims(data,axis=1)

print('Data-dimension:     ')
print(data1.shape)


###### Train/Test Split
from sklearn.model_selection import train_test_split
xTrain, xTest, yTrain, yTest = train_test_split(data1, label, test_size = 0.2, random_state = 0)

print('Training data:     ')
print(xTrain.shape)

print('Testing data:     ')
print(xTest.shape)

#Import svm model
from sklearn import svm

#Create a svm Classifier
clf = svm.SVC(kernel='linear', C=1).fit(xTrain, yTrain)   # Linear Kernel

#Score
#print(clf.score(xTest, yTest))

resulttrain=clf.predict(xTrain)
resulttest=clf.predict(xTest)


############ Classification Report

from sklearn.metrics import classification_report
print('Classification report Test ')
print(classification_report(yTest, resulttest, labels=[0,1]))

print('Classification report Train ')
print(classification_report(yTrain, resulttrain, labels=[0,1]))
### Cross validation

from sklearn.model_selection import cross_val_score
scores = cross_val_score(clf, xTrain, yTrain, cv=5)
print('Traning cross validation : 5-forld ')
print(scores)


print('Testing Accuracy: ')
print(clf.score(xTest, yTest))


