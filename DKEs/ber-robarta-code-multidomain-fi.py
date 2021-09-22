import nltk
import os
#nltk.download('averaged_perceptron_tagger')
import numpy as np
import re
from seqeval.metrics import precision_score, recall_score, f1_score, classification_report

words=[]
wordtags=[]
nanjo=[]

#os.chdir(r'D:\NLP\FSND\ODU-NLP-master\ODU-NLP-master\OA-STM-domains\all-data')
x=open('combinedtrain.txt',encoding="utf8")
for i in x:
    #i=re.sub(r'[^\w\s]','',i)
    a=i.split('\t')
    #print(a)
    w1=a[0].rstrip()
    #words.append(w1)
    if len(a) <= 1:  # deal with the '\n' in the first row
        continue
    else:
        wt = a[3].rstrip()
    words.append(w1)
    wordtags.append(wt)
    na=(w1,wt)
    nanjo.append(na)

words1=[]
wordtags1=[]
nanjo1=[]
x1=open('combinedtest.txt',encoding="utf8")
for i in x1:
    #i=re.sub(r'[^\w\s]','',i)
    a=i.split('\t')
    #print(a)
    w1=a[0].rstrip()
    #words1.append(w1)
    if len(a) <= 1:  # deal with the '\n' in the first row
        continue
    else:
        wt = a[3].rstrip()
    words1.append(w1)
    wordtags1.append(wt)
    na=(w1,wt)
    nanjo1.append(na)

wordsall = []
wordtagsall = []
nanjoall = []

#print(len(words), len(wordtags), len(words1), len(wordtags1))
for i in range(len(words)):
    wordsall.append(words[i])
    wordtagsall.append(wordtags[i])
    nanjoall.append(nanjo[i])

for j in range(len(words1)):
    wordsall.append(words1[j])
    wordtagsall.append(wordtags1[j])
    nanjoall.append(nanjo1[j])

def cuu(nanjo):
    cc = []
    cc1 = []
    for i in nanjo:
        if i != ('.', 'O'):
            pos= nltk.pos_tag([i[0]])
            tag= pos[0][1]
            p =(i[0],i[1])
            cc1.append(p)
            # print(cc1)
        else:
            #cc1.append(('.','.','_', 'O'))
            cc.append(cc1)
            cc1 = []
    return cc

def cuuold(nanjo):
    #store word-tags in the same one sentence into one list [[(),()]]
    cc = []
    cc1 = []
    for i in nanjo:
        if i != ('',''):
            pos = nltk.pos_tag([i[0]])
            tag = pos[0][1]
            p = (i[0], tag, '_', i[1])
            cc1.append(p)
            #print(cc1)
        else:
            cc1.append(('.', '.', '_', 'O'))
            cc.append(cc1)
            cc1=[]
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

train_data=simple_data(cc1)
test_data= simple_data(cc2)


##################### Simple transformer ####################

import logging

import pandas as pd
from simpletransformers.ner import NERModel, NERArgs


logging.basicConfig(level=logging.INFO)
transformers_logger = logging.getLogger("transformers")
transformers_logger.setLevel(logging.WARNING)

train_data = pd.DataFrame(train_data, columns=["sentence_id", "words", "labels"])

eval_data1 = pd.DataFrame(
    test_data, columns=["sentence_id", "words", "labels"])

# Configure the model
model_args = NERArgs()
model_args.train_batch_size = 16
model_args.evaluate_during_training = True
model_args.labels_list = tags
model_args.num_train_epochs= 15

#model = NERModel("bert", "bert-base-cased", args=model_args)   ## bert model 

model = NERModel( "roberta", "roberta-base", args=model_args)    ### Roberta model 

# Train the model
model.train_model(train_data, eval_data=train_data)


# Evaluate the model
result, model_outputs, preds_list = model.eval_model(eval_data1)

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

print('########### Individual Domain Testing ####################')
print('##################### ARG #################################')
import os 
os.chdir('/home/reshad/fakenews/deeplearnig/OA-STM-domains/Arg')

words=[]
wordtags=[]
nanjo=[]

x1=open('test.txt',encoding="utf8")
for i in x1:
    #i=re.sub(r'[^\w\s]','',i)
    a=i.split('\t')
    #print(a)
    w1=a[0].rstrip()
    #words.append(w1)
    if len(a) <= 1:  # deal with the '\n' in the first row
        continue
    else:
        wt = a[3].rstrip()
    words.append(w1)
    wordtags.append(wt)
    na=(w1,wt)
    nanjo.append(na)

arg = cuu(nanjo)
arg1= test_data(arg)
prediction = test_result(arg1)
pred_labels1 = prediction_extraction(prediction)
true_labels1 = [[w[1] for w in s] for s in arg]

print("F1-score: {:.1%}".format(f1_score(true_labels1, pred_labels1)))
print("Precision-score: {:.1%}".format(precision_score(true_labels1, pred_labels1)))
print("Recal-score: {:.1%}".format(recall_score(true_labels1, pred_labels1)))

print('##################### Astr #################################')
import os
os.chdir('/home/reshad/fakenews/deeplearnig/OA-STM-domains/Astr')

words=[]
wordtags=[]
nanjo=[]

x=open('test.txt',encoding="utf8")
for i in x:
    #i=re.sub(r'[^\w\s]','',i)
    a=i.split('\t')
    #print(a)
    w1=a[0].rstrip()
    #words.append(w1)
    if len(a) <= 1:  # deal with the '\n' in the first row
        continue
    else:
        wt = a[3].rstrip()
    words.append(w1)
    wordtags.append(wt)
    na=(w1,wt)
    nanjo.append(na)

astr = cuu(nanjo)
astr1= test_data(astr)
prediction = test_result(astr1)
pred_labels1 = prediction_extraction(prediction)
true_labels1 = [[w[1] for w in s] for s in astr]

print("F1-score: {:.1%}".format(f1_score(true_labels1, pred_labels1)))
print("Precision-score: {:.1%}".format(precision_score(true_labels1, pred_labels1)))
print("Recal-score: {:.1%}".format(recall_score(true_labels1, pred_labels1)))


print('##################### Bio #################################')
import os
os.chdir('/home/reshad/fakenews/deeplearnig/OA-STM-domains/Bio')

words=[]
wordtags=[]
nanjo=[]

x=open('test.txt',encoding="utf8")
for i in x:
    #i=re.sub(r'[^\w\s]','',i)
    a=i.split('\t')
    #print(a)
    w1=a[0].rstrip()
    #words.append(w1)
    if len(a) <= 1:  # deal with the '\n' in the first row
        continue
    else:
        wt = a[3].rstrip()
    words.append(w1)
    wordtags.append(wt)
    na=(w1,wt)
    nanjo.append(na)

bio = cuu(nanjo)
bio1= test_data(bio)
prediction = test_result(bio1)
pred_labels1 = prediction_extraction(prediction)
true_labels1 = [[w[1] for w in s] for s in bio]

print("F1-score: {:.1%}".format(f1_score(true_labels1, pred_labels1)))
print("Precision-score: {:.1%}".format(precision_score(true_labels1, pred_labels1)))
print("Recal-score: {:.1%}".format(recall_score(true_labels1, pred_labels1)))


print('##################### Chem #################################')
import os
os.chdir('/home/reshad/fakenews/deeplearnig/OA-STM-domains/Chem')

words=[]
wordtags=[]
nanjo=[]

x=open('test.txt',encoding="utf8")
for i in x:
    #i=re.sub(r'[^\w\s]','',i)
    a=i.split('\t')
    #print(a)
    w1=a[0].rstrip()
    #words.append(w1)
    if len(a) <= 1:  # deal with the '\n' in the first row
        continue
    else:
        wt = a[3].rstrip()
    words.append(w1)
    wordtags.append(wt)
    na=(w1,wt)
    nanjo.append(na)

chem = cuu(nanjo)
chem1= test_data(chem)
prediction = test_result(chem1)
pred_labels1 = prediction_extraction(prediction)
true_labels1 = [[w[1] for w in s] for s in chem]

print("F1-score: {:.1%}".format(f1_score(true_labels1, pred_labels1)))
print("Precision-score: {:.1%}".format(precision_score(true_labels1, pred_labels1)))
print("Recal-score: {:.1%}".format(recall_score(true_labels1, pred_labels1)))


print('##################### CS #################################')
import os
os.chdir('/home/reshad/fakenews/deeplearnig/OA-STM-domains/CS')

words=[]
wordtags=[]
nanjo=[]

x=open('test.txt',encoding="utf8")
for i in x:
    #i=re.sub(r'[^\w\s]','',i)
    a=i.split('\t')
    #print(a)
    w1=a[0].rstrip()
    #words.append(w1)
    if len(a) <= 1:  # deal with the '\n' in the first row
        continue
    else:
        wt = a[3].rstrip()
    words.append(w1)
    wordtags.append(wt)
    na=(w1,wt)
    nanjo.append(na)

cs = cuu(nanjo)
cs1= test_data(cs)
prediction = test_result(cs1)
pred_labels1 = prediction_extraction(prediction)
true_labels1 = [[w[1] for w in s] for s in cs]

print("F1-score: {:.1%}".format(f1_score(true_labels1, pred_labels1)))
print("Precision-score: {:.1%}".format(precision_score(true_labels1, pred_labels1)))
print("Recal-score: {:.1%}".format(recall_score(true_labels1, pred_labels1)))


print('##################### Eng #################################')
import os
os.chdir('/home/reshad/fakenews/deeplearnig/OA-STM-domains/Eng')

words=[]
wordtags=[]
nanjo=[]

x=open('test.txt',encoding="utf8")
for i in x:
    #i=re.sub(r'[^\w\s]','',i)
    a=i.split('\t')
    #print(a)
    w1=a[0].rstrip()
    #words.append(w1)
    if len(a) <= 1:  # deal with the '\n' in the first row
        continue
    else:
        wt = a[3].rstrip()
    words.append(w1)
    wordtags.append(wt)
    na=(w1,wt)
    nanjo.append(na)

eng = cuu(nanjo)
eng1= test_data(eng)
prediction = test_result(eng1)
pred_labels1 = prediction_extraction(prediction)
true_labels1 = [[w[1] for w in s] for s in eng]

print("F1-score: {:.1%}".format(f1_score(true_labels1, pred_labels1)))
print("Precision-score: {:.1%}".format(precision_score(true_labels1, pred_labels1)))
print("Recal-score: {:.1%}".format(recall_score(true_labels1, pred_labels1)))
print('##################### ES #################################')
import os
os.chdir('/home/reshad/fakenews/deeplearnig/OA-STM-domains/ES')

words=[]
wordtags=[]
nanjo=[]

x=open('test.txt',encoding="utf8")
for i in x:
    #i=re.sub(r'[^\w\s]','',i)
    a=i.split('\t')
    #print(a)
    w1=a[0].rstrip()
    #words.append(w1)
    if len(a) <= 1:  # deal with the '\n' in the first row
        continue
    else:
        wt = a[3].rstrip()
    words.append(w1)
    wordtags.append(wt)
    na=(w1,wt)
    nanjo.append(na)

es = cuu(nanjo)
es1= test_data(es)
prediction = test_result(es1)
pred_labels1 = prediction_extraction(prediction)
true_labels1 = [[w[1] for w in s] for s in es]

print("F1-score: {:.1%}".format(f1_score(true_labels1, pred_labels1)))
print("Precision-score: {:.1%}".format(precision_score(true_labels1, pred_labels1)))
print("Recal-score: {:.1%}".format(recall_score(true_labels1, pred_labels1)))


print('##################### Math #################################')
import os
os.chdir('/home/reshad/fakenews/deeplearnig/OA-STM-domains/Math')

words=[]
wordtags=[]
nanjo=[]

x=open('test.txt',encoding="utf8")
for i in x:
    #i=re.sub(r'[^\w\s]','',i)
    a=i.split('\t')
    #print(a)
    w1=a[0].rstrip()
    #words.append(w1)
    if len(a) <= 1:  # deal with the '\n' in the first row
        continue
    else:
        wt = a[3].rstrip()
    words.append(w1)
    wordtags.append(wt)
    na=(w1,wt)
    nanjo.append(na)

math = cuu(nanjo)
math1= test_data(math)
prediction = test_result(math1)
pred_labels1 = prediction_extraction(prediction)
true_labels1 = [[w[1] for w in s] for s in math]

print("F1-score: {:.1%}".format(f1_score(true_labels1, pred_labels1)))
print("Precision-score: {:.1%}".format(precision_score(true_labels1, pred_labels1)))
print("Recal-score: {:.1%}".format(recall_score(true_labels1, pred_labels1)))


print('##################### Med #################################')
import os
os.chdir('/home/reshad/fakenews/deeplearnig/OA-STM-domains/Med')

words=[]
wordtags=[]
nanjo=[]

x=open('test.txt',encoding="utf8")
for i in x:
    #i=re.sub(r'[^\w\s]','',i)
    a=i.split('\t')
    #print(a)
    w1=a[0].rstrip()
    #words.append(w1)
    if len(a) <= 1:  # deal with the '\n' in the first row
        continue
    else:
        wt = a[3].rstrip()
    words.append(w1)
    wordtags.append(wt)
    na=(w1,wt)
    nanjo.append(na)

med = cuu(nanjo)
med1= test_data(med)
prediction = test_result(med1)
pred_labels1 = prediction_extraction(prediction)
true_labels1 = [[w[1] for w in s] for s in med]

print("F1-score: {:.1%}".format(f1_score(true_labels1, pred_labels1)))
print("Precision-score: {:.1%}".format(precision_score(true_labels1, pred_labels1)))
print("Recal-score: {:.1%}".format(recall_score(true_labels1, pred_labels1)))


print('##################### MS #################################')
import os
os.chdir('/home/reshad/fakenews/deeplearnig/OA-STM-domains/MS')

words=[]
wordtags=[]
nanjo=[]

x=open('test.txt',encoding="utf8")
for i in x:
    #i=re.sub(r'[^\w\s]','',i)
    a=i.split('\t')
    #print(a)
    w1=a[0].rstrip()
    #words.append(w1)
    if len(a) <= 1:  # deal with the '\n' in the first row
        continue
    else:
        wt = a[3].rstrip()
    words.append(w1)
    wordtags.append(wt)
    na=(w1,wt)
    nanjo.append(na)

ms = cuu(nanjo)
ms1= test_data(ms)
prediction = test_result(ms1)
pred_labels1 = prediction_extraction(prediction)
true_labels1 = [[w[1] for w in s] for s in ms]

print("F1-score: {:.1%}".format(f1_score(true_labels1, pred_labels1)))
print("Precision-score: {:.1%}".format(precision_score(true_labels1, pred_labels1)))
print("Recal-score: {:.1%}".format(recall_score(true_labels1, pred_labels1)))




