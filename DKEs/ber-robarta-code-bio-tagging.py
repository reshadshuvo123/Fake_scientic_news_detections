import numpy as np
import re
import nltk
import os
#nltk.download('averaged_perceptron_tagger')
import numpy as np
import re
from seqeval.metrics import precision_score, recall_score, f1_score, classification_report

words=[]
wordtags=[]
nanjo=[]
x=open('combinedTrain.txt',encoding="utf8")
for i in x:
    #i=re.sub(r'[^\w\s]','',i)
    a=i.split('\t')
    #print(a)
    w1=a[0].rstrip()
    words.append(w1)
    wordtags.append(a[1].rstrip())
    na=(w1,a[1].rstrip())
    nanjo.append(na)

words1=[]
wordtags1=[]
nanjo1=[]
x1=open('combinedTest.txt',encoding="utf8")
for i in x1:
    #i=re.sub(r'[^\w\s]','',i)
    a=i.split('\t')
    #print(a)
    w1=a[0].rstrip()
    words1.append(w1)
    wordtags1.append(a[1].rstrip())
    na=(w1,a[1].rstrip())
    nanjo1.append(na)


wordsall=[]
wordtagsall=[]
nanjoall=[]

for i in range(len(words)):
    wordsall.append(words[i])
    wordtagsall.append(wordtags[i])
    nanjoall.append(nanjo[i])

for j in range(len(words1)):
    wordsall.append(words1[j])
    wordtagsall.append(wordtags1[j])
    nanjoall.append(nanjo1[j])


def cuu(nanjo):
    cc=[]
    cc1=[]
    for i in nanjo:
        if i != ('',''):
            cc1.append(i)
            #print(cc1)
        else:
            cc.append(cc1)
            cc1=[]
    ss=('.','O')
    for i in range(len(cc)):
        cc[i].append(ss)
    for i in range(len(cc)):
        for j in range(len(cc[i])):
            if cc[i][j] == ('.', 'O'):
                w=cc[i][j-1][0].replace(".","")
                t = cc[i][j-1][1]
                cc[i][j-1]=(w,t)
    return cc

cc1=cuu(nanjo)
cc2=cuu(nanjo1)

word=[]
tag=[]

for i in range(len(cc1)):
    for j in range(len(cc1[i])):
        word.append(cc1[i][j][0])
        tag.append(cc1[i][j][1])

for i in range(len(cc2)):
    for j in range(len(cc2[i])):
        word.append(cc2[i][j][0])
        tag.append(cc2[i][j][1])


words = list(set((word)))
words.append("ENDPAD")


n_words = len(words)
print(n_words)


tags = list(set((tag)))
n_tags = len(tags)
print(n_tags)

################# Simple Transformer Data formation CONLL-2003 ################

def simple_data(seq):
    pp = []
    for i in range(len(cc2)):
        for j in range(len(cc2[i])):
            a = [i, cc2[i][j][0], cc2[i][j][1]]
            pp.append(a)
    return pp

#train_data=simple_data(cc1[:1358])
#eval_data = simple_data(cc1[1358:])
#test_data= simple_data(cc2)

train_data=simple_data(cc1[:1358])
eval_data = simple_data(cc1[1358:])
test_data= simple_data(cc2)


##################### Simple transformer ####################

import logging

import pandas as pd
from simpletransformers.ner import NERModel, NERArgs


logging.basicConfig(level=logging.INFO)
transformers_logger = logging.getLogger("transformers")
transformers_logger.setLevel(logging.WARNING)

train_data = pd.DataFrame(train_data, columns=["sentence_id", "words", "labels"])

eval_data = pd.DataFrame(eval_data, columns=["sentence_id", "words", "labels"])

test_data = pd.DataFrame(test_data, columns=["sentence_id", "words", "labels"])

# Configure the model
model_args = NERArgs()
model_args.train_batch_size = 16
model_args.evaluate_during_training = True
model_args.labels_list = tags
model_args.num_train_epochs= 15

model = NERModel("bert", "bert-base-cased", args=model_args)   ## bert model

#model = NERModel( "roberta", "roberta-base", args=model_args)    ### Roberta model

# Train the model
model.train_model(train_data, eval_data=eval_data)


# Evaluate the model
result, model_outputs, preds_list = model.eval_model(test_data)

print(result)

################# FUnction ###################


def test_data(cc2):
   aa = []
   pp = []
   for i in range(len(cc2)):
      for j in range(len(cc2[i])):
         a = cc2[i][j][0]
         aa.append(a)
      p = " ".join(aa)
      aa = []
      pp.append([p])
   return pp

def test_result(arg1):
   cc=[]
   gg=[]
   for i in range(len(arg1)):
      predictions, _ = model.predict(arg1[i])
      cc.append(tag_get(predictions))
      gg.append(cc)
      cc=[]
   return gg

def tag_get(po):
   pp = []
   tagg = []
   for i in range(len(po)):
      for j in range(len(po[i])):
         p = list(po[i][j].items())
         tags = p[0][1]
         tagg.append(tags)
      pp.append(tagg)
      tagg = []
   return pp


def prediction_extraction(sg):
   cc = []
   for i in sg:
      a = i[0][0]
      cc.append(a)
   return cc

###################################### Testing ################

words=[]
wordtags=[]
nanjo=[]

x1=open('combinedTest.txt',encoding="utf8")
for i in x1:
    #i=re.sub(r'[^\w\s]','',i)
    a=i.split('\t')
    #print(a)
    w1=a[0].rstrip()
    words.append(w1)
    wordtags.append(a[1].rstrip())
    na=(w1,a[1].rstrip())
    nanjo.append(na)

astr = cuu(nanjo)
print(astr)
astr1= test_data(astr)
print(astr1)
prediction = test_result(astr1)
pred_labels1 = prediction_extraction(prediction)
true_labels1 = [[w[1] for w in s] for s in astr]

#print(pred_labels1)
#print(true_labels1)

print("F1-score: {:.1%}".format(f1_score(true_labels1, pred_labels1)))
print("Precision-score: {:.1%}".format(precision_score(true_labels1, pred_labels1)))
print("Recal-score: {:.1%}".format(recall_score(true_labels1, pred_labels1)))


