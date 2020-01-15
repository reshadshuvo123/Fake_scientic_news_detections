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
tag2idx = {t: i for i, t in enumerate(tags)}

X = [[w[0] for w in s] for s in cc1]

new_X = []
for seq in X:
    new_seq = []
    for i in range(max_len):
        try:
            new_seq.append(seq[i])
        except:
            new_seq.append("__PAD__")
    new_X.append(new_seq)
X = new_X

y = [[tag2idx[w[1]] for w in s] for s in cc1]
from keras.preprocessing.sequence import pad_sequences
y = pad_sequences(maxlen=max_len, sequences=y, padding="post", value=tag2idx["O"])

batch_size = 32

import tensorflow as tf
import tensorflow_hub as hub
from keras import backend as K
sess = tf.Session()
K.set_session(sess)

elmo_model = hub.Module("https://tfhub.dev/google/elmo/2", trainable=True)
sess.run(tf.global_variables_initializer())
sess.run(tf.tables_initializer())

def ElmoEmbedding(x):
    return elmo_model(inputs={
                            "tokens": tf.squeeze(tf.cast(x, tf.string)),
                            "sequence_len": tf.constant(batch_size*[max_len])
                      },
                      signature="tokens",
                      as_dict=True)["elmo"]

from keras.models import Model, Input
from keras.layers.merge import add
from keras.layers import LSTM, Embedding, Dense, TimeDistributed, Dropout, Bidirectional, Lambda
import keras
from keras_self_attention import SeqSelfAttention

input_text = Input(shape=(max_len,), dtype=tf.string)

embedding = Lambda(ElmoEmbedding, output_shape=(max_len, 1024))(input_text)

x = Bidirectional(LSTM(units=512, return_sequences=True,
                       recurrent_dropout=0.2, dropout=0.2))(embedding)
x_rnn = Bidirectional(LSTM(units=512, return_sequences=True,
                           recurrent_dropout=0.2, dropout=0.2))(x)
x = add([x, x_rnn])  # residual connection to the first biLSTM
att = SeqSelfAttention(attention_type=SeqSelfAttention.ATTENTION_TYPE_MUL,
                       kernel_regularizer=keras.regularizers.l2(1e-4),
                       bias_regularizer=keras.regularizers.l1(1e-4),
                       attention_regularizer_weight=1e-4,
                       name='Attention')(x)
out = TimeDistributed(Dense(n_tags, activation="softmax"))(att)

model = Model(input_text, out)

model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
model.summary()

X_tr, X_val = X[:1312], X[1312:1344]
y_tr, y_val = y[:1312], y[1312:1344]
y_tr = y_tr.reshape(y_tr.shape[0], y_tr.shape[1], 1)
y_val = y_val.reshape(y_val.shape[0], y_val.shape[1], 1)



import time

start = time.time()

history = model.fit(np.array(X_tr), y_tr, validation_data=(np.array(X_val), y_val),
                    batch_size=batch_size, epochs=15, verbose=1)

end = time.time()
print(end - start)

from seqeval.metrics import precision_score, recall_score, f1_score, classification_report
test_pred = model.predict(np.array(X_tr), verbose=1)

# turn indices to tags
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

def original(y):
    sent_tag_list = []
    for sent_tags in y:
        word_tag_list = []
        for tag in sent_tags:
            word_tag_list.append(idx2tag[tag])
        sent_tag_list.append(word_tag_list)
    return sent_tag_list

# predicted labels
pred_labels = pred2label(test_pred)
test_labels=original(y)
# F1 score
print("F1-score: {:.1%}".format(f1_score(test_labels, pred_labels)))


X2 = [[w[0] for w in s] for s in cc2]

new_X = []
for seq in X2:
    new_seq = []
    for i in range(max_len):
        try:
            new_seq.append(seq[i])
        except:
            new_seq.append("__PAD__")
    new_X.append(new_seq)
X2 = new_X[:512]

y2 = [[tag2idx[w[1]] for w in s] for s in cc2]
from keras.preprocessing.sequence import pad_sequences
y2 = pad_sequences(maxlen=max_len, sequences=y2, padding="post", value=tag2idx["O"])
y2=y2[:512]
print(len(y2))

#start = time.time()
test_pred1 = model.predict(np.array(X2), verbose=1)
pred_labels1 = pred2label(test_pred1)
test_labels1=original(y2)
# F1 score
print("F1-score: {:.1%}".format(f1_score(test_labels1, pred_labels1)))




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

def sent2features(X2):
    new_X = []
    for seq in X2:
        new_seq = []
        for i in range(max_len):
            try:
               new_seq.append(seq[i])
            except:
               new_seq.append("__PAD__")
        new_X.append(new_seq)
    return new_X

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

def blank(x,y,z,a):
    p= ("__PAD__")
    ind= ('1000000', '10000000')
    t= ("O")
    for i in range(a):
        x.append([p])
        y.append([ind])
        z.append([t])
    return x,y,z 

import copy

data = os.listdir('/home/mhoque/medke-master/crfModel/medicalData/convertedBIO/test')

start = time.time()
for iip in data:
    os.chdir('/home/mhoque/medke-master/crfModel/medicalData/convertedBIO/test')
    print(iip)
    (test_sents,test_sents_indices, tagss) = convertCONLLFormJustExtractionSemEvalPerfile(iip)
    if len(tagss) > 32:
       a = len(tagss)%32
       b = len(tagss)//32
       c = b*32 - a 
       test_sents,test_sents_indices, tagss = blank(test_sents,test_sents_indices, tagss, c)
    elif len(tagss)<32:
       a = 32 - (len(tagss)%32)
       test_sents,test_sents_indices, tagss = blank(test_sents,test_sents_indices,tagss, a)
    print(len(test_sents), len(test_sents_indices), len(tagss))
    Xt=sent2features(test_sents)
    test_predt1 = model.predict(np.array(Xt), verbose=1)
    #idx2tag = {i: w for w, i in tag2idx.items()}
    pred_labels1 = pred2label(test_predt1)
    yt1 = [[tag2idx[w] for w in s] for s in tagss]
    yt1 = pad_sequences(maxlen=max_len, sequences=yt1, padding="post", value=tag2idx["O"])
    #ytt1 = [to_categorical(i, num_classes=n_tags) for i in yt1]
    true_labels1 = original(yt1)

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
'''
    os.chdir('/home/mhoque/result-dke/LSTM-R-Elmo1-A')
    ii=0
    with open(fileoutLoc,"w") as f:
        for sen1 in sen:
            phrases=sen1[-1]['phrases']
            for (p,pis,pie) in phrases:
                f.write("T{0}\tKEYPHRASE_NOTYPES {1} {2}\t{3}\n".format(str(ii),pis,pie,p))
                ii+=1
    print("classified file written at",fileoutLoc)
    os.rename('deep7.ann',str(iip) + '__predicted.ann')
'''

end = time.time()
print(end - start)
