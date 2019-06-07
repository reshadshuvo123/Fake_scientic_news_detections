import bs4 as bs
import urllib2
import re
import nltk
import heapq
import pickle
import sys
from pprint import pprint

from sklearn_crfsuite import metrics

from DataExtraction import convertCONLLFormJustExtractionSemEvalPerfile
from FeatureExtraction import sent2labels,sent2features
from PhraseEval import phrasesFromTestSenJustExtractionWithIndex
from nltk import word_tokenize, pos_tag, ne_chunk
from nltk import conlltags2tree, tree2conlltags

link='https://www.sciencealert.com/nasa-has-released-a-breathtaking-new-hubble-image-of-two-galaxies-smashing'


scraped_data = urllib2.Request(link, headers={'User-Agent' : "Magic Browser"})
scraped_data=urllib2.urlopen(scraped_data)
article = scraped_data.read()
parsed_article = bs.BeautifulSoup(article,'lxml')
paragraphs = parsed_article.find_all('p')
article_text = ""
for p in paragraphs:
    article_text += p.text

### Preprocessing
# Removing Square Brackets and Extra Spaces
article_text = re.sub(r'\[[0-9]*\]', ' ', article_text)
article_text = re.sub(r'\s+', ' ', article_text)
# Removing special characters and digits
formatted_article_text = re.sub('[^a-zA-Z]', ' ', article_text )
formatted_article_text = re.sub(r'\s+', ' ', formatted_article_text)
#pp=formatted_article_text
pp=article_text
print("******* Given link *******")

print(link)
print('\n')
print('***************Retrived Text: ************************')
print('\n')
print(pp)

##### Preprocessing the text for input ####################

from nltk.tokenize import sent_tokenize
sent_tokenize_list = sent_tokenize(pp)
#print sent_tokenize_list


tree6 = [ne_chunk(pos_tag(word_tokenize(x))) for x in sent_tokenize_list]
#print (tree6)

iob_tags4=[]
for k in range(len(tree6)):
    #print(k)
    #print(tree6[k])
    iob_tags2 = tree2conlltags(tree6[k])
    #print(iob_tags2)
    iob_tags3=[]
    for i in range(len(iob_tags2)):
        #iob_tags3=[]
        c=iob_tags2[i][2]
        if c[0]=='B':
            iob_tags1=(iob_tags2[i][0], iob_tags2[i][1], 'B')
        elif c[0] == 'I':
            iob_tags1= (iob_tags2[i][0], iob_tags2[i][1], 'I')
        else:
            iob_tags1= (iob_tags2[i][0], iob_tags2[i][1], 'o')
        iob_tags3.append(iob_tags1)
    iob_tags4.append(iob_tags3)

#print(' **** Preprocessed Text given as input to model *******')
#print(iob_tags4)

X_test = [sent2features(s) for s in iob_tags4]

crf = pickle.load(open("linear-chain-crf.model.pickle"))
y_pred = crf.predict(X_test)


labels = list(crf.classes_)
labels.remove('O')

#print labels
#print y_pred

test_sents_pls = []  #test sentences with predicted labels
for index,testsent in enumerate(iob_tags4):
    sent=[]
    pls = y_pred[index]
    for (token,pl) in zip(testsent,pls):
        nt=(token[0],token[1],pl)
        sent.append(nt)
    test_sents_pls.append(sent)

#print(test_sents_pls)
key_pharse=[]
#print('****Model Prediction ***')
for i in range(len(test_sents_pls)):
    #print(test_sents_pls[i])
    p=test_sents_pls[i]
    for j in range(len(p)):
        if p[j][2] in labels:
            key_pharse.append(p[j][0])

#print("Key Pharse : ******** ")
#print(key_pharse)

import nltk
#nltk.download('stopwords')
from nltk.corpus import stopwords

stopwords = nltk.corpus.stopwords.words('english')
stopwords.append('.')
stopwords.append(',')
key_pharse1 = [w for w in key_pharse if not w in stopwords]
#print(" Final Key Pharse : ******** ")

#print(key_pharse1)

##### Grouping Words #####

import itertools
import string
data=test_sents_pls
punctuation = set(string.punctuation)
sentences = [[' '.join(w[0] for w in g) for k, g in itertools.groupby(sen, key=lambda x: x[0] not in punctuation and x[2] != 'O') if k] for sen in data]

#print sentences

key_words=[]
for se in sentences:
    for i in se:
        key_words.append(i)
#key_words= set(key_words)

key_wrods1=[str(r) for r in key_words]


print('\n')
print (' ************** Key Words  ***************')

print('\n')

for i in key_wrods1:
    print i
