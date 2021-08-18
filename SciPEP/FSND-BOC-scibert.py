import bs4 as bs
import urllib
#import urllib.request
import re
import nltk
import heapq


link = 'https://www.sciencealert.com/brain-to-brain-mind-connection-lets-three-people-share-thoughts'

### read data
req = urllib.request.Request(link, headers={'User-Agent' : "Magic Browser"})
scraped_data = urllib.request.urlopen(req)
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
processed_text=article_text

print("******* Given link *******")

print(link)
print('\n')

print('***************Retrived Text: ************************')
print('\n')
print(processed_text)
print('\n')
##################################### DKE Extraction ##################################
from simpletransformers.ner import NERModel, NERArgs
import nltk
from nltk.tokenize import sent_tokenize

model = NERModel("bert", "/home/reshad/fakenews/deeplearnig/outputs-bio/checkpoint-180-epoch-15")   ### our data
#model = NERModel("roberta", "/home/reshad/fakenews/deeplearnig/outputs-multidomian/checkpoint-210-epoch-15")   ### Multi domain Bert

text = sent_tokenize(processed_text)

test_data=[]
for i in range(len(text)):
    test_data.append([text[i]])


########## Function For DKE Extraction #############
def list_data(p):
   cc=[]
   gg=[]
   for i in range(len(p)):
      for j in range(len(p1[i][0][0])):
         word=list(p[i][0][0][j].keys())[0]
         tag=p[i][0][0][j][word]
         token=(word, tag)
         cc.append(token)
      gg.append(cc)
      cc=[]
   return gg

def test_result(arg1):
   cc=[]
   gg=[]
   for i in range(len(arg1)):
      predictions, _ = model.predict([arg1[i]])
      cc.append(predictions)
      gg.append(cc)
      cc=[]
   return gg

def label_correction(p):
   for i in range(len(p)):
      for j in range(len(p[i][0])):
         try:
            if p[i][0][j][list(p[i][0][j].keys())[0]]=='I-KP':
               if p[i][0][j-1][list(p[i][0][j-1].keys())[0]]=='O':
                  p[i][0][j][list(p[i][0][j].keys())[0]] = 'B-KP'
         except:
            pass
   return p
####################################################
pp1=test_result(test_data)
p1= label_correction(pp1)
pp1= list_data(p1)

import itertools
import string
punctuation = set(string.punctuation)
dke = [[' '.join(w[0] for w in g) for k, g in itertools.groupby(sen, key=lambda x: x[0] and x[1] != 'O') if k] for sen in pp1]
print(' ********************* Extracted DKE *****************')
print('\n')
#print(dke)
key_words=[]
for i in range(len(dke)):
    for j in range(len(dke[i])):
        p=dke[i][j].rstrip('.')
        p=p.rstrip(',')
        key_words.append(p)

print(key_words)

print('\n')
print("Total number of key_words :  ")
print(len(key_words))
print('\n')

def dke_p(pp1):
    dke = [[' '.join(w[0] for w in g) for k, g in itertools.groupby(sen, key=lambda x: x[0] and x[1] != 'O') if k] for sen in pp1]
    key_words=[]
    for i in range(len(dke)):
        for j in range(len(dke[i])):
            p=dke[i][j].rstrip('.')
            p=p.rstrip(',')
            key_words.append(p)
    return key_words

############################################# Paper Extraction Using Arxiv #####################

papers_name= []
papers_links=[]
papers_abstract=[]

import urllib.request as ur
from bs4 import BeautifulSoup
num=0
for i in range(len(key_words)):
    weblinks=[]
    papers=[]
    abstract_arxiv=[]
    num=num+1
   # print(num)
    query = urllib.parse.quote(key_words[i])
    query1= 'all:' + query
    url = 'http://export.arxiv.org/api/query?search_query=' + query1 +'&start=0&max_results=10'
    #url = 'http://export.arxiv.org/api/query?search_query=all:{}'.format(key_words[i])
    s = ur.urlopen(url)
    sl = s.read()
    soup = BeautifulSoup(sl, 'html.parser')
    papers=[soup.find_all('title')]
    weblinks=[soup.find_all('link')]
    abstract_arxiv=[soup.find_all('summary')]
    for links in range(1,len(papers[0])):
        papers_abstract.append(abstract_arxiv[0][links-1].string)
        papers_name.append(papers[0][links].string)
        papers_links.append(weblinks[0][links].string)

N=len(key_words)
cc=[]
for i in range(len(key_words)):
    queryaa = urllib.parse.quote(key_words[i])
    cc.append(query)

Query_all=('ANDall').join(cc)

weblinks_all=[]
papers_all=[]
abstract_arxiv_all=[]

# print(num)
#query = urllib.parse.quote(key_words[i])
#query1= 'all:' + query
urlall = 'http://export.arxiv.org/api/query?search_query=' + Query_all +'&start=0&max_results=30'

#url = 'http://export.arxiv.org/api/query?search_query=all:{}'.format(key_words[i])

s = ur.urlopen(url)
sl = s.read()
soup = BeautifulSoup(sl, 'html.parser')

papers_all=[soup.find_all('title')]
weblinks_all=[soup.find_all('link')]
abstract_arxiv_all=[soup.find_all('summary')]
for links in range(1,len(papers_all[0])):
    papers_abstract.append(abstract_arxiv_all[0][links-1].string)
    papers_name.append(papers_all[0][links].string)
    papers_links.append(weblinks_all[0][links].string)

N=len(key_words)
for i in range(len(key_words)-2):
    weblinks3=[]
    papers3=[]
    abstract_arxiv3=[]
    #num=num+1
   # print(num)
    query1 = urllib.parse.quote(key_words[i])
    query11= 'all:' + query1
    query2= urllib.parse.quote(key_words[i+1])
    query22= 'all:' + query2
    query3 = urllib.parse.quote(key_words[i+2])
    query33= 'all:' + query3
    url3 = 'http://export.arxiv.org/api/query?search_query=' + query11 + 'AND'+ query22 + 'AND' + query33 + '&start=0&max_results=10'
    #url = 'http://export.arxiv.org/api/query?search_query=all:{}'.format(key_words[i])
    s = ur.urlopen(url)
    sl = s.read()
    soup = BeautifulSoup(sl, 'html.parser')
    papers3=[soup.find_all('title')]
    weblinks3=[soup.find_all('link')]
    abstract_arxiv3=[soup.find_all('summary')]
    for links in range(1,len(papers3[0])):
        papers_abstract.append(abstract_arxiv3[0][links-1].string)
        papers_name.append(papers3[0][links].string)
        papers_links.append(weblinks3[0][links].string)



print(len(papers_name))
print(len(papers_links))
print(len(papers_abstract))

non_duplicate_titles=[]
non_duplicate_links=[]
non_duplicate_abstract =[]

for i in range(len(papers_name)):
    if papers_name[i] not in non_duplicate_titles:
        non_duplicate_titles.append(papers_name[i])
        non_duplicate_abstract.append(papers_abstract[i])
        non_duplicate_links.append(papers_links[i])

all_abstract=[article_text]
for i in range(len(non_duplicate_abstract)):
    all_abstract.append(non_duplicate_abstract[i])

print('\n')
print('Ranking based on Cosine similarity : using Bag of scibert')



#########################################
from simpletransformers.language_representation import RepresentationModel
import nltk 
from nltk.tokenize import sent_tokenize
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import scipy
import torch
from transformers import BertTokenizer, BertModel
from sklearn.metrics.pairwise import cosine_similarity

model_version = 'allenai/scibert_scivocab_uncased'
do_lower_case = True
model1 = BertModel.from_pretrained(model_version)
tokenizer = BertTokenizer.from_pretrained(model_version, do_lower_case=do_lower_case)

from sklearn.metrics.pairwise import cosine_similarity
def embed_text(text, model):
    input_ids = torch.tensor(tokenizer.encode(text)).unsqueeze(0)  # Batch size 1
    outputs = model(input_ids)
    last_hidden_states = outputs[0]  # The last hidden-state is the first element of the output tuple
    return last_hidden_states 

def get_similarity(em, em2):
    return cosine_similarity(em.detach().numpy(), em2.detach().numpy())

def scibert_embedding(text):
    em= embed_text(text[0], model1).mean(1)
    em=em.detach().numpy()
    for i in range(1,len(text)):
        em1=embed_text(text[i], model1).mean(1)
        em1=em1.detach().numpy()
        em=np.concatenate((em,em1), axis=0)
        em=np.mean(em,axis=0)
        return np.expand_dims(em,axis=1)

extracted_abstract_score=[]
news_vectors = scibert_embedding(key_words)


for i in range(len(non_duplicate_abstract)):
    text=non_duplicate_abstract[i].replace('\n','')
    text = sent_tokenize(text)
    pp1=test_result(text)
    p1= label_correction(pp1)
    pp1= list_data(p1)
    pp1= dke_p(pp1)
    word_vectors =  scibert_embedding(pp1)
    cosine=cosine_similarity(news_vectors,word_vectors)
    extracted_abstract_score.append(cosine)


ranks_list={}

for i in range(len(non_duplicate_titles)):
    ranks_list[non_duplicate_titles[i]]=extracted_abstract_score[i]

lists=sorted(ranks_list.items(), key=lambda x: x[1], reverse=True)
for i in range(50):
    print(lists[i])





    

