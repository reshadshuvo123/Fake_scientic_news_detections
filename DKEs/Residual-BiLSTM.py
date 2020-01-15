import numpy as np
import re

words=[]
wordtags=[]
nanjo=[]
#x=open('combinedTrain.txt',encoding="utf8")
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
n_words = len(words); n_words

tags = list(set((tag)))
n_tags = len(tags); n_tags


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


input_text = Input(shape=(max_len,), dtype=tf.string)
embedding = Lambda(ElmoEmbedding, output_shape=(None, 1024))(input_text)
x = Bidirectional(LSTM(units=512, return_sequences=True,
                       recurrent_dropout=0.2, dropout=0.2))(embedding)
x_rnn = Bidirectional(LSTM(units=512, return_sequences=True,
                           recurrent_dropout=0.2, dropout=0.2))(x)
x = add([x, x_rnn])  # residual connection to the first biLSTM
out = TimeDistributed(Dense(n_tags, activation="softmax"))(x)

model = Model(input_text, out)
model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

model.summary()


history = model.fit(np.array(X), y, batch_size=batch_size, epochs=5, verbose=1)

i = 19
p = model.predict(np.array(X_te[i:i+batch_size]))[0]
p = np.argmax(p, axis=-1)
print("{:15} {:5}: ({})".format("Word", "Pred", "True"))
print("="*30)
for w, true, pred in zip(X_te[i], y_te[i], p):
    if w != "__PAD__":
        print("{:15}:{:5} ({})".format(w, tags[pred], tags[true]))
