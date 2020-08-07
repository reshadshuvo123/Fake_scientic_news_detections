import numpy as np
import re

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


max_len = 130
word2idx = {w: i for i, w in enumerate(words)}
tag2idx = {t: i for i, t in enumerate(tags)}

from keras.preprocessing.sequence import pad_sequences
X = [[word2idx[w[0]] for w in s] for s in cc1]
X = pad_sequences(maxlen=max_len, sequences=X, padding="post", value=n_words - 1)

X1 = [[word2idx[w[0]] for w in s] for s in cc2]
X1 = pad_sequences(maxlen=max_len, sequences=X1, padding="post", value=n_words - 1)

y = [[tag2idx[w[1]] for w in s] for s in cc1]
y = pad_sequences(maxlen=max_len, sequences=y, padding="post", value=tag2idx["O"])

y1 = [[tag2idx[w[1]] for w in s] for s in cc2]
y1 = pad_sequences(maxlen=max_len, sequences=y1, padding="post", value=tag2idx["O"])


from keras.utils import to_categorical
y = [to_categorical(i, num_classes=n_tags) for i in y]

from keras.models import Model, Input
from keras.layers import LSTM, Embedding, Dense, TimeDistributed, Dropout, Bidirectional
from keras_contrib.layers import CRF
from keras.layers.merge import add
import keras
from keras_self_attention import SeqSelfAttention

input = Input(shape=(max_len,))
model = Embedding(input_dim=n_words + 1, output_dim=50,
                  input_length=max_len, mask_zero=True)(input)  # 20-dim embedding
model = Bidirectional(LSTM(units=250, return_sequences=True,
                           recurrent_dropout=0.2))(model)  # variational biLSTM
model1 = Bidirectional(LSTM(units=250, return_sequences=True, recurrent_dropout=0.2))(model)
x = add([model, model1])
'''
attn = SeqSelfAttention(attention_type=SeqSelfAttention.ATTENTION_TYPE_MUL,
                       kernel_regularizer=keras.regularizers.l2(1e-4),
                       bias_regularizer=keras.regularizers.l1(1e-4),
                       attention_regularizer_weight=1e-4,
                       name='Attention')(x)
'''
model = TimeDistributed(Dense(50, activation="relu"))(x)  # a dense layer as suggested by neuralNer
crf = CRF(n_tags)  # CRF layer
out = crf(model)  # output
model = Model(input, out)
model.compile(optimizer="adam", loss=crf.loss_function, metrics=[crf.accuracy])
model.summary()
history = model.fit(X,np.array(y), batch_size=32, epochs=15,verbose=1)


from seqeval.metrics import precision_score, recall_score, f1_score, classification_report
test_pred = model.predict(X, verbose=1)

idx2tag = {i: w for w, i in tag2idx.items()}

def pred2label(pred):
    out = []
    for pred_i in pred:
        out_i = []
        for p in pred_i:
            p_i = np.argmax(p)
            out_i.append(idx2tag[p_i].replace("PAD", "O"))
        out.append(out_i)
    return out
    
pred_labels = pred2label(test_pred)
true_labels = pred2label(y)


print("Traing")
print("F1-score: {:.1%}".format(f1_score(true_labels, pred_labels)))


print('Testing ******************')
y1 = [to_categorical(i, num_classes=n_tags) for i in y1]
test_pred1 = model.predict(X1, verbose=1)

pred_labels1 = pred2label(test_pred1)
true_labels1 = pred2label(y1)

print("F1-score: {:.1%}".format(f1_score(true_labels1, pred_labels1)))





import pickle
import sys
from pprint import pprint

from sklearn_crfsuite import metrics

#from DataExtraction import convertCONLLFormJustExtractionSemEvalPerfile
#from FeatureExtraction import sent2labels,sent2features
from PhraseEval import phrasesFromTestSenJustExtractionWithIndex
import os
import nltk  
#crf = pickle.load(open("linear-chain-crf.model.pickle", "rb"))
os.chdir('/home/mhoque/medke-master/crfModel/medicalData/convertedBIO/test')

fileinLoc = 'abbott-2006-01.txt'
fileoutLoc = sys.argv[0].split(".")[0]+".ann"

def convertCONLLFormJustExtractionSemEvalPerfile(loc):
    dT=open(loc, encoding='utf-8').read().split("\n")[:-2]
    sI = [-1] + [i for i, x in enumerate(dT) if not x.strip()] + [len(dT)]
    sT1s = [dT[sI[i]+1:sI[i+1]] for i in range(len(sI)-1)]
    sTs = []
    sTIs = []
    sTss=[]
    for s in sT1s:
        ts= [(x.split("\t")[0],x.split("\t")[1],x.split("\t")[2]) for x in s]
        ts1 = [(tss[0],tss[1],tss[2].split(',')[0],tss[2].split(',')[1]) for tss in ts]
        tokens = [(tss[0]) for tss in ts1]
        tags=[(tss[1]) for tss in ts1]
        tokenindices = [(tss[2], tss[3]) for tss in ts1]
        sTs.append(tokens)
        sTIs.append(tokenindices)
        sTss.append(tags)
    return (sTs,sTIs, sTss)

def sent2features(sent):          
    X1 = pad_sequences(sequences=[[word2idx.get(w.replace(".",""), 0) for w in s] for s in sent],padding="post", value=0, maxlen=max_len)
    return X1

def sent2labels(sent):
    y1 = [[tag2idx[w[1]] for w in s] for s in sent]
    y1 = pad_sequences(maxlen=max_len, sequences=y1, padding="post", value=tag2idx["O"])
    return y1

from seqeval.metrics import precision_score, recall_score, f1_score, classification_report
from keras.utils import to_categorical
import nltk 

def pred2label(pred):
    out = []
    for pred_i in pred:
        out_i = []
        for p in pred_i:
            p_i = np.argmax(p)
            out_i.append(idx2tag[p_i].replace("PAD", "O"))
        out.append(out_i)
    return out

def getPhraseTokensWithIndex(bDs,iDs,senLength):
    bpTokens=[]
    for i in range(len(bDs)):
        start = bDs[i][0]
        end = senLength
        if i < len(bDs)-1:
            end = bDs[i+1][0]  
        phrase =  ' '.join([x[1] for x in [bDs[i]]+[x for x in iDs if x[0] > start and x[0] < end]])
        phraseStart = min([x[-2] for x in [bDs[i]]+[x for x in iDs if x[0] > start and x[0] < end]])
        phraseEnd =  max([x[-1] for x in [bDs[i]]+[x for x in iDs if x[0] > start and x[0] < end]])
        bpTokens.append((phrase,phraseStart,phraseEnd))
        
    return bpTokens

import copy

data = os.listdir('/home/mhoque/medke-master/crfModel/medicalData/convertedBIO/test')

for iip in data:
    os.chdir('/home/mhoque/medke-master/crfModel/medicalData/convertedBIO/test')
    print(iip)
    (test_sents,test_sents_indices, tagss) = convertCONLLFormJustExtractionSemEvalPerfile(iip)
    Xt=sent2features(test_sents)
    test_predt1 = model.predict(Xt, verbose=1)
    idx2tag = {i: w for w, i in tag2idx.items()}
    pred_labels1 = pred2label(test_predt1)
    yt1 = [[tag2idx[w] for w in s] for s in tagss]
    yt1 = pad_sequences(maxlen=max_len, sequences=yt1, padding="post", value=tag2idx["O"])
    ytt1 = [to_categorical(i, num_classes=n_tags) for i in yt1]
    true_labels1 = pred2label(ytt1)
    
    print("F1-score: {:.1%}".format(f1_score(true_labels1, pred_labels1)))
    print((metrics.flat_classification_report(true_labels1, pred_labels1)))
    
    test_sents_pls = []  #test sentences with predicted labels
    for index,testsent in enumerate(test_sents):
        sent=[]
        pls = pred_labels1[index]
        for (token,pl) in zip(testsent,pls):
            po = nltk.pos_tag([token])[0][1]
            nt=(token, po ,pl)
            sent.append(nt)
        test_sents_pls.append(sent)
        
    x=test_sents_pls
    y=test_sents_indices
    sen=[]
    
    for i in range(len(x)):
        sen1=[]
        for j in range(len(x[i])):
            gg= (x[i][j][0],x[i][j][1],x[i][j][2], int(y[i][j][0]), int(y[i][j][1]))
            #print(gg)
            sen1.append(gg)
        #print(sen1)
        bS = sorted([(f,x1[0],x1[-2],x1[-1]) for (f,x1) in enumerate(sen1) if x1[2] == "B-KP"], key = lambda x1:x1[0])
        iS = sorted([(f,x1[0],x1[-2],x1[-1]) for (f,x1) in enumerate(sen1) if x1[2] == "I-KP"], key = lambda x1:x1[0])
        tSen =copy.deepcopy(sen1)
        tSen.append(
            {
                'phrases': getPhraseTokensWithIndex(bS,iS,len(sen1)),
            }
        )
        sen.append(tSen)
    
    print(len(test_sents),len(sen))
    
    os.chdir('/home/mhoque/result-dke/LSTM-CRF-R')
    ii=0
    with open(fileoutLoc,"w") as f:
        for sen1 in sen:
            phrases=sen1[-1]['phrases']
            for (p,pis,pie) in phrases:
                f.write("T{0}\tKEYPHRASE_NOTYPES {1} {2}\t{3}\n".format(str(ii),pis,pie,p))
                ii+=1
    print("classified file written at",fileoutLoc)
    os.rename('deep222.ann',str(iip) + '__predicted.ann')
