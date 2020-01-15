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

words = list(set(word))
n_words = len(words)


tags = list(set(tag))
n_tags = len(tags)



max_len = 130
max_len_char = 10


word2idx = {w: i + 2 for i, w in enumerate(words)}
word2idx["UNK"] = 1
word2idx["PAD"] = 0
idx2word = {i: w for w, i in word2idx.items()}
tag2idx = {t: i + 1 for i, t in enumerate(tags)}
tag2idx["PAD"] = 0
idx2tag = {i: w for w, i in tag2idx.items()}

from keras.preprocessing.sequence import pad_sequences
X_word = [[word2idx[w[0]] for w in s] for s in cc1]

X_word = pad_sequences(maxlen=max_len, sequences=X_word, value=word2idx["PAD"], padding='post', truncating='post')

chars = set([w_i for w in words for w_i in w])
n_chars = len(chars)
print(n_chars)

char2idx = {c: i + 2 for i, c in enumerate(chars)}
char2idx["UNK"] = 1
char2idx["PAD"] = 0

X_char = []
for sentence in cc1:
    sent_seq = []
    for i in range(max_len):
        word_seq = []
        for j in range(max_len_char):
            try:
                word_seq.append(char2idx.get(sentence[i][0][j]))
            except:
                word_seq.append(char2idx.get("PAD"))
        sent_seq.append(word_seq)
    X_char.append(np.array(sent_seq))


y = [[tag2idx[w[1]] for w in s] for s in cc1]
y = pad_sequences(maxlen=max_len, sequences=y, value=tag2idx["PAD"], padding='post', truncating='post')

from keras.models import Model, Input
from keras.layers import LSTM, Embedding, Dense, TimeDistributed, Dropout, Conv1D
from keras.layers import Bidirectional, concatenate, SpatialDropout1D, GlobalMaxPooling1D

# input and embedding for words
word_in = Input(shape=(max_len,))
emb_word = Embedding(input_dim=n_words + 2, output_dim=20,
                     input_length=max_len, mask_zero=True)(word_in)

# input and embeddings for characters
char_in = Input(shape=(max_len, max_len_char,))
emb_char = TimeDistributed(Embedding(input_dim=n_chars + 2, output_dim=10,
                           input_length=max_len_char, mask_zero=True))(char_in)
# character LSTM to get word encodings by characters
char_enc = TimeDistributed(LSTM(units=20, return_sequences=False,
                                recurrent_dropout=0.5))(emb_char)

# main LSTM
x = concatenate([emb_word, char_enc])
x = SpatialDropout1D(0.3)(x)
main_lstm = Bidirectional(LSTM(units=50, return_sequences=True,
                               recurrent_dropout=0.6))(x)
out = TimeDistributed(Dense(n_tags + 1, activation="softmax"))(main_lstm)

model = Model([word_in, char_in], out)


model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["acc"])

model.summary()


history = model.fit([X_word,
                     np.array(X_char).reshape((len(X_char), max_len, max_len_char))],
                    np.array(y).reshape(len(y), max_len, 1),
                    batch_size=32, epochs=15, validation_split=0.1, verbose=1)

y_pred = model.predict([X_word,
                        np.array(X_char).reshape((len(X_char),
                                                     max_len, max_len_char))])


from seqeval.metrics import precision_score, recall_score, f1_score, classification_report

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

pred_labels = pred2label(y_pred)

def original(y):
    sent_tag_list = []
    for sent_tags in y:
        word_tag_list = []
        for tag in sent_tags:
            word_tag_list.append(idx2tag[tag])
        sent_tag_list.append(word_tag_list)
    for i in range(len(sent_tag_list)):
        for j in range(len(sent_tag_list[i])):
            if sent_tag_list[i][j]=='PAD':
                sent_tag_list[i][j]=sent_tag_list[i][j].replace('PAD','O')
    return sent_tag_list



true_labels = original(y)

print('Trainning :       ')

print("F1-score: {:.1%}".format(f1_score(true_labels, pred_labels)))



X_word1 = [[word2idx[w[0]] for w in s] for s in cc2]

X_word1 = pad_sequences(maxlen=max_len, sequences=X_word1, value=word2idx["PAD"], padding='post', truncating='post')


X_char1 = []
for sentence in cc2:
    sent_seq = []
    for i in range(max_len):
        word_seq = []
        for j in range(max_len_char):
            try:
                word_seq.append(char2idx.get(sentence[i][0][j]))
            except:
                word_seq.append(char2idx.get("PAD"))
        sent_seq.append(word_seq)
    X_char1.append(np.array(sent_seq))


y2 = [[tag2idx[w[1]] for w in s] for s in cc2]
y2 = pad_sequences(maxlen=max_len, sequences=y2, value=tag2idx["PAD"], padding='post', truncating='post')


y_pred1 = model.predict([X_word1,
                        np.array(X_char1).reshape((len(X_char1),
                                                     max_len, max_len_char))])


from seqeval.metrics import precision_score, recall_score, f1_score, classification_report

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

pred_labels1 = pred2label(y_pred1)
true_labels1 = original(y2)

print('Testing :       ')

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
os.chdir('/home/reshad/medke-master/crfModel/medicalData/convertedBIO/test')

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

def sent2wordfeatures(sent):
    X1 = pad_sequences(sequences=[[word2idx.get(w.replace(".",""), 0) for w in s] for s in sent],padding="post", value=0, maxlen=max_len)
    return X1

def sent2charfeatures(sent):
    X_char1 = []
    for sentence in sent:
        sent_seq = []
        for i in range(max_len):
            word_seq = []
            for j in range(max_len_char):
                try:
                   word_seq.append(char2idx.get(sentence[i][0][j],0))
                except:
                   word_seq.append(char2idx.get("PAD"))
            sent_seq.append(word_seq)
        X_char1.append(np.array(sent_seq))
    return X_char1


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
data = os.listdir('/home/reshad/medke-master/crfModel/medicalData/convertedBIO/test')

for iip in data:
    os.chdir('/home/reshad/medke-master/crfModel/medicalData/convertedBIO/test')    
    print(iip)
    (test_sents,test_sents_indices, tagss) = convertCONLLFormJustExtractionSemEvalPerfile(iip)
    Xwt=sent2wordfeatures(test_sents)
    Xct=sent2charfeatures(test_sents)

    test_predt1 = model.predict([Xwt,np.array(Xct).reshape((len(Xct),max_len, max_len_char))])

    idx2tag = {i: w for w, i in tag2idx.items()}
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

    os.chdir('/home/reshad/fakenews/deeplearnig/result')
    ii=0
    with open(fileoutLoc,"w") as f:
        for sen1 in sen:
            phrases=sen1[-1]['phrases']
            for (p,pis,pie) in phrases:
                f.write("T{0}\tKEYPHRASE_NOTYPES {1} {2}\t{3}\n".format(str(ii),pis,pie,p))
                ii+=1
    print("classified file written at",fileoutLoc)
    os.rename('deep33333.ann',str(iip) + '__predicted.ann')

