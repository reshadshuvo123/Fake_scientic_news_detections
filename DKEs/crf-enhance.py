%matplotlib inline
import matplotlib.pyplot as plt
plt.style.use('ggplot')

from itertools import chain

import nltk
import sklearn
import scipy.stats
from sklearn.metrics import make_scorer
#from sklearn.cross_validation import cross_val_score
#from sklearn.grid_search import RandomizedSearchCV

import sklearn_crfsuite
from sklearn_crfsuite import scorers
from sklearn_crfsuite import metrics
import pickle


loc1='features1-train.txt'
loc2='features1-test.txt'

def word2features(sent, i):
    word = sent[i][0]
    postag = sent[i][1]
    tag1=sent[i][2]
    tag2=sent[i][4]
    tag3 = sent[i][5]

    features = {
        'bias': 1.0,
        'word.lower()': word.lower(),
        'word[-3:]': word[-3:],
        'word[-2:]': word[-2:],
        'word.isupper()': word.isupper(),
        'word.istitle()': word.istitle(),
        'word.isdigit()': word.isdigit(),
        'postag': postag,
        'postag[:2]': postag[:2],
        'tag1': tag1,
        'tag1[:2]': tag1[:2],
        'tag2': tag2,
        'tag2[:2]': tag2[:2],
        'tag3': tag3,
        'tag3[:2]': tag3[:2],
        'wordlength': len(word),
        'wordinitialcap': word[0].isupper(),
        'wordmixedcap': len([x for x in word[1:] if x.isupper()])>0,
        'wordallcap': len([x for x in word if x.isupper()])==len(word),
        'distfromsentbegin': i
    }
    if i > 0:
        word1 = sent[i-1][0]
        postag1 = sent[i-1][1]
        features.update({
            '-1:word.lower()': word1.lower(),
            '-1:word.istitle()': word1.istitle(),
            '-1:word.isupper()': word1.isupper(),
            '-1:postag': postag1,
            '-1:postag[:2]': postag1[:2],
            '-1:tag1': tag1,
            '-1:tag1[:2]': tag1[:2],
            '-1:tag2': tag2,
            '-1:tag2[:2]': tag2[:2],
            '-1:tag3': tag3,
            '-1:tag3[:2]': tag3[:2],
        })
    else:
        features['BOS'] = True

    if i < len(sent)-1:
        word1 = sent[i+1][0]
        postag1 = sent[i+1][1]
        features.update({
            '+1:word.lower()': word1.lower(),
            '+1:word.istitle()': word1.istitle(),
            '+1:word.isupper()': word1.isupper(),
            '+1:postag': postag1,
            '+1:postag[:2]': postag1[:2],
            '+1:tag1': tag1,
            '+1:tag1[:2]': tag1[:2],
            '+1:tag2': tag2,
            '+1:tag2[:2]': tag2[:2],
            '+1:tag3': tag3,
            '+1:tag3[:2]': tag3[:2],
        })
    else:
        features['EOS'] = True

    return features


def sent2features(sent):
    return [word2features(sent, i) for i in range(len(sent))]

def sent2labels(sent):
    return [label for token, postag, tag1, label, tag2, tag3 in sent]

def sent2tokens(sent):
    return [token for token, postag, tag1, label, tag2, tag3, tag4, tag5 in sent]

def convertCONLLFormJustExtractionSemEval1(loc):
    dT=open(loc, encoding='utf-8').read().split("\n")[:-2]
    sI = [-1] + [i for i, x in enumerate(dT) if not x.strip()] + [len(dT)]
    sT1s = [dT[sI[i]+1:sI[i+1]] for i in range(len(sI)-1)]
    sTs = []
    for s in sT1s:
        ts= [(x.split("\t")[0],x.split("\t")[1], x.split("\t")[2], x.split("\t")[3], x.split("\t")[4],x.split("\t")[5], x.split("\t")[6], x.split("\t")[7]) for x in s]
        ts1= [(tss[0],tss[1],tss[2],tss[3], tss[6], tss[7]) for tss in ts]
        sTs.append(ts1)
    return sTs
	
train_sents = convertCONLLFormJustExtractionSemEval1(loc1)
test_sents = convertCONLLFormJustExtractionSemEval1(loc2)

X_train = [sent2features(s) for s in train_sents]
y_train = [sent2labels(s) for s in train_sents]

X_test = [sent2features(s) for s in test_sents]
y_test = [sent2labels(s) for s in test_sents]

%%time
crf = sklearn_crfsuite.CRF(
    algorithm='lbfgs',
    c1=0.1,
    c2=0.1,
    max_iterations=100,
    all_possible_transitions=True
)
crf.fit(X_train, y_train)

y_pred = crf.predict(X_test)

labels = list(crf.classes_)
labels.remove('O')
print(labels)

metrics.flat_f1_score(y_test, y_pred,
                      average='weighted', labels=labels)
					  
sorted_labels = sorted(labels,key=lambda name: (name[1:], name[0]))
print((metrics.flat_classification_report(y_test, y_pred, labels=sorted_labels, digits=3)))

pickle.dump(crf,open("linear-chain-crf-enhanced.model.pickle","wb"), protocol = 0, fix_imports = True)