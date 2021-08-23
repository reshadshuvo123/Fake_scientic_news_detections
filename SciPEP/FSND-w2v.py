import bs4 as bs
import urllib
#import urllib.request
import re
import nltk
import heapq
import requests
from bs4 import BeautifulSoup
import time 
start_time = time.time()

# 1
#link = 'https://www.sciencealert.com/distant-stars-are-showing-peculiar-outburst-activity-that-scientists-can-t-explain?perpetual=yes&limitstart=1'
#2
#link = 'https://www.sciencealert.com/a-physicist-has-proposed-a-pretty-depressing-explanation-for-why-we-never-see-aliens?perpetual=yes&limitstart=1'
#3
#link = 'https://www.sciencealert.com/something-s-hiding-in-our-outer-solar-system-but-it-might-not-be-planet-nine?perpetual=yes&limitstart='
#4
#link = 'https://www.sciencealert.com/groundbreaking-sharp-images-of-a-protoplanetary-system-find-no-planets?perpetual=yes&limitstart=1'
#5 same
#link = 'https://www.sciencealert.com/distant-stars-are-showing-peculiar-outburst-activity-that-scientists-can-t-explain?perpetual=yes&limitstart=1'
#6
#link = 'https://www.sciencealert.com/this-ai-tries-to-guess-what-you-look-like-based-on-your-voice'
#7
#link = 'https://www.livescience.com/65029-dueling-reality-photons.html'
#8
#link = 'https://www.eurekalert.org/pub_releases/2018-04/au-2bt042418.php'
#9
#link = 'https://www.bbc.com/news/science-environment-43063379'
#10
#link = 'https://www.newsweek.com/brainnet-scientists-connect-three-peoples-brains-so-they-can-communicate-1150448'
#11
#link = 'https://futurism.com/ai-generates-fake-news'
#12
#link= 'https://www.forbes.com/sites/briankoberlein/2019/01/27/is-science-fiction-right-about-wormholes/#7b0683764cb0'
#13
#link = 'https://www.sciencealert.com/harvard-scientists-say-earth-was-stuck-by-an-interstellar-object-5-years-ago'
#14
#link = 'https://www.sciencemag.org/news/2014/12/study-massive-preprint-archive-hints-geography-plagiarism'
#15
#link = 'https://www.eurekalert.org/pub_releases/2017-11/uoc--rtn112817.php'
#16
#link = 'https://www.nbcnews.com/mach/science/why-scientists-are-teaching-ai-think-dog-ncna869266'
# 17
#link = 'https://www.independent.co.uk/news/science/multiple-realities-quantum-physics-experiment-research-study-a8833341.html'
##18
#link = 'https://techxplore.com/news/2018-10-brainnet-people-brainwaves-tetris.html'
#19
#link = 'https://www.theatlantic.com/technology/archive/2016/07/the-six-main-arcs-in-storytelling-identified-by-a-computer/490733/'
#20
#link = 'https://www.insider.com/echoes-in-gravitational-waves-hint-at-a-breakdown-of-einsteins-theory-of-relativity-2016-12'
#21
#link = 'https://earthsky.org/space/solar-flare-10-billion-times-more-powerful-than-sunshttps://earthsky.org/space/solar-flare-10-billion-times-more-powerful-than-suns'
##22
#link = 'https://www.sciencealert.com/physicists-say-there-could-be-a-strange-source-of-negative-gravity-all-around-us'
#23
#link = 'https://www.sciencenewsforstudents.org/article/scientists-enlist-computers-hunt-down-fake-news'
#24
#link = 'https://nexusnewsfeed.com/article/science-futures/this-megamerger-of-14-galaxies-could-become-the-most-massive-structure-in-our-universe/'
#25
#link ='https://www.smithsonianmag.com/smart-news/artificial-intelligence-generates-humans-faces-based-their-voices-180972402/'
#26
#link='https://www.nature.com/articles/d41586-019-03334-5'
#27
#link = 'http://scienceglobalnews.com/space/nasas-new-planet-hunter-has-detected-an-exocomet-orbiting-an-alien-star/'
#28
#link = 'https://news.stanford.edu/2017/11/15/algorithm-outperforms-radiologists-diagnosing-pneumonia/'
#29
#link = 'https://www.cnet.com/news/scientists-propose-a-new-type-of-dark-matter-and-how-we-can-find-it/'
#30
#link='https://www.nature.com/news/quantum-boost-for-artificial-intelligence-1.13453'
#31
#link ='https://www.vice.com/en/article/a3mqy8/this-ai-can-insert-your-selfie-into-famous-paintings'
#32
link = 'https://www.forbes.com/sites/williamfalcon/2018/09/01/facebook-ai-just-set-a-new-record-in-translation-and-why-it-matters/?sh=363ea1723124'
#33
#link = 'https://techxplore.com/news/2019-05-pal-wearable-context-aware-health-cognition.html'
#34
#link = 'https://www.sciencealert.com/neural-networks-performance-increases-if-they-re-allowed-to-sleep-and-dream'
#35
#link = 'https://www.sciencealert.com/hot-jupiter-wasp-104b-one-of-the-darkest-planets-ever'
#36
#link='https://allthatsinteresting.com/brainnet-three-brains-connected'
#37
#link='https://www.sciencenewsforstudents.org/article/camera-caught-small-rock-hit-moon-impact'
#38
#link='https://www.sciencealert.com/breaking-astronomers-are-closing-in-on-the-cause-of-a-mysterious-explosion-they-called-the-cow/amp'
#39
#link='https://www.sciencealert.com/vast-cosmic-fountain-blasting-out-of-and-falling-into-abell-2597-supermassive-black-hole?perpetual=yes&limitstart=1'
#40 ### Suscribe
#link='https://www.sciencenews.org/article/quantum-strategy-could-verify-solutions-unsolvable-problems-theory'
#link = 'https://www.sciencealert.com/hyperion-proto-supercluster-high-redshift-11-billion-light-years-most-distant'
#41
#link='https://www.sciencealert.com/black-holes-expand-in-not-out-and-the-reason-why-could-fix-physics/amp'
#42
#link='https://www.sciencealert.com/scientists-successfully-connected-the-brains-of-3-people-enabling-them-to-share-thoughts/amp'
#43
#link = 'https://nypost.com/2018/10/03/were-one-step-closer-to-being-able-to-read-each-others-minds/'
#44
#link = 'https://nypost.com/2020/03/02/scientists-claim-theyve-discovered-the-first-extraterrestrial-protein/'
#45
#link='https://sciencesprings.wordpress.com/2020/03/17/from-science-alert-the-supermassive-black-hole-at-the-centre-of-our-galaxy-is-becoming-more-active'
#link = 'https://www.sciencealert.com/the-supermassive-black-hole-at-our-galaxy-s-centre-is-growing-more-active'
#46
#link = 'https://www.sciencealert.com/an-absolutely-gargantuan-black-hole-has-been-found-as-massive-as-40-billion-suns'
#47
#link = 'https://www.scientificamerican.com/article/midsize-black-holes-may-explain-the-milky-ways-speediest-stars/'
#48
#link = 'https://www.sciencealert.com/coders-mutate-ai-systems-to-make-them-evolve-faster-than-we-can-program-them'
#49
#link = 'https://www.foxnews.com/science/scientists-claim-theyve-discovered-first-extraterrestrial-protein-in-meteorite-that-fell-to-earth-30-years-ago'
#50
#link = 'https://www.intelligentliving.co/hemolithin-the-first-extraterrestrial-protein-found-on-a-meteorite/'
#51
#link = 'https://www.livescience.com/scientists-detected-26-million-possible-technosignatures-from-us.html'
#52
#link = 'https://www.sciencenews.org/article/seti-anniversary-new-search-methods-hunt-alien-intelligence'
#53
#link = 'https://www.discovermagazine.com/planet-earth/tunguska-explosion-caused-by-asteroid-grazing-the-earth-say-scientists'
##54
#link = 'https://www.sciencealert.com/computer-scientists-solve-one-of-history-s-weirdest-da-vinci-painting-mysteries'
#55
#link = 'https://www.sciencealert.com/a-new-study-has-found-that-one-dark-matter-candidate-hasn-t-killed-anyone'
#56
#link='https://sciencesprings.wordpress.com/2020/05/01/from-science-alert-strangely-flaring-dead-star-could-be-the-missing-link-between-magnetars-and-pulsars/'
#link = 'https://www.sciencealert.com/this-really-weird-star-magnets-like-a-magnetar-but-pulses-like-a-pulsar'
#57
#link ='https://www.sciencealert.com/controversial-ai-has-been-trained-to-kill-humans-in-a-doom-deathmatch'
#58
#link ='https://www.sciencealert.com/necroplanetology-the-study-of-planets-dismembered-remains'
#59
#link = 'https://www.sciencenews.org/article/clumpy-universe-disagreement-physics-cosmology'
#60
#link = 'https://www.discovermagazine.com/mind/how-artificial-neural-networks-paved-the-way-for-a-dramatic-new-theory-of'
#61
#link = 'https://news.artnet.com/art-world/scientists-solve-mystery-salvator-mundi-orb-1745037'
#62
#link = 'https://www.washingtonpost.com/science/2019/09/11/found-first-potentially-habitable-super-earth-with-water-its-skies/'
#63
#link ='https://www.sciencealert.com/human-consciousness-could-be-a-result-of-entropy-study-science'
#64
#link ='https://www.sciencealert.com/our-bossy-black-hole-kicked-out-a-star-and-it-s-shooting-through-the-galaxy-insanely-fast'
#65
#link = 'https://www.sciencealert.com/dark-matter-is-once-again-a-likely-cause-of-the-strange-glow-in-the-galactic-center'
#66
#link = 'https://www.sciencealert.com/mathematicians-discover-a-strange-pattern-hiding-in-prime-numbers'
#67
#link = 'https://www.sciencealert.com/new-paper-extreme-solar-system-objects-contain-no-evidence-of-planet-nine'
# 68
#link = 'https://www.sciencealert.com/fascinating-new-paper-lays-out-a-navigation-system-for-interstellar-space'
# 69
#link = 'https://www.sciencealert.com/theoretical-constraints-masses-early-organic-molecules'
# 70
#link = 'https://www.sciencealert.com/this-nearby-alien-world-may-have-lost-its-atmosphere-and-then-grown-a-new-one'
# 71
##link = 'https://www.the-sun.com/lifestyle/tech/480269/first-known-alien-protein-unlike-anything-on-earth-found-inside-meteorite-by-scientists/'
# 72
##link ='https://www.sciencealert.com/human-consciousness-could-be-a-result-of-entropy-study-science'
# 73
#link = 'https://futurism.com/astronomers-found-ancient-remains-big-bang-fossil-cloud'
# 74 ###
#link = 'https://www.sciencealert.com/the-supermassive-black-hole-at-our-galaxy-s-centre-is-growing-more-active'
# 75
#link = 'https://venturebeat.com/2021/03/05/study-warns-deepfakes-can-fool-facial-recognition/'
# 76
#link = 'http://www.buffalo.edu/news/releases/2021/03/010.html'
# 77
#link = 'https://www.nature.com/articles/d41586-020-03277-2'
# 78
#link = 'https://www.sciencemag.org/news/2020/11/potential-signs-life-venus-are-fading-astronomers-downgrade-their-original-claims'
# 79
#link = 'https://www.nature.com/articles/d41586-020-00031-6'
# 80
#link = 'https://www.sciencealert.com/this-german-retiree-just-solved-one-of-world-s-most-complex-maths-problems-and-no-one-noticed'
# 81
#link = 'https://www.sciencealert.com/amateur-solves-decades-old-maths-problem-about-colours-that-can-never-touch-hadwiger-nelson-problem'
# 82
#link = 'http://astrobiology.com/2020/10/no-phosphine-in-the-atmosphere-of-venus.html'
# 83
#link = 'https://www.sciencealert.com/math-genius-has-come-up-with-a-wildly-simple-new-way-to-solve-quadratic-equations'
# 84
#link = 'https://www.sciencemag.org/news/2020/04/artificial-intelligence-evolving-all-itself'
# 85 aaa
#link = 'https://singularityhub.com/2020/07/26/deepminds-newest-ai-programs-itself-to-make-all-the-right-decisions/'
# 86
#link = 'https://www.newsweek.com/picasso-painted-over-portrait-sitting-woman-ai-reconstructed-lost-art-1461052'
# 87
#link = 'https://venturebeat.com/2020/04/07/microsoft-ai-fake-news-better-than-state-of-the-art-baselines/'
#88
#link = 'https://venturebeat.com/2020/06/08/googles-performer-ai-architecture-could-advance-protein-analysis-and-cut-compute-costs/'
# 89
#link = 'https://scitechdaily.com/wealth-of-discoveries-from-gravitational-wave-data-leads-to-most-detailed-black-hole-family-portrait/'
# 90
#link = 'https://www.sciencealert.com/this-is-how-online-dating-has-changed-the-very-fabric-of-society'
#91
#link = 'https://www.livescience.com/black-holes-gravitational-molecules-evidence.html'
#92
#link = 'https://www.eurekalert.org/pub_releases/2020-12/kift-pbh122520.php'
#93
#link = 'https://www.universetoday.com/138980/proxima-centauri-just-released-a-flare-so-powerful-it-was-visible-to-the-unaided-eye-planets-there-would-get-scorched-1/'
#94 aa
#link = 'https://www.sciencealert.com/astronomers-have-finally-pinpointed-the-source-of-one-of-those-mysterious-cosmic-radio-bursts'
#95
#link = 'https://www.sciencealert.com/amazing-images-of-uranus-rings-show-they-re-unlike-anything-else-in-the-solar-system'
# 96
#link = 'https://www.sciencealert.com/nuclear-pasta-10-billion-times-stronger-than-steel-gravitational-wave-neutron-stars'
# 97
#link = 'https://www.sciencealert.com/adorably-named-super-puff-planets-are-like-nothing-in-the-solar-system'
#link = 'https://www.sciencealert.com/ai-can-now-enhance-pixelated-photos-to-make-them-over-60-times-sharper'
# 98
#link = 'https://www.sciencealert.com/there-s-a-way-to-start-a-turing-machine-while-playing-magic-the-gathering'
# 99
#link = 'https://www.sciencealert.com/this-new-optical-device-could-one-day-detect-plant-life-on-distant-alien-worlds?fbclid=IwAR0lAol5GlXBzFtz0MJGGo3Su0FWGlMbn6z8z6H_IGuWulId2qjlwR5QaE0'
# 100
#link = 'https://www.sciencealert.com/scientists-want-to-build-a-super-fast-self-replicating-computer-that-grows-as-it-computes'
# 101
#link = 'https://www.sciencealert.com/experts-think-this-is-how-long-we-have-before-ai-takes-all-of-our-jobs'
#102
#link = 'https://www.sciencealert.com/is-there-a-message-from-the-universe-s-creator-in-the-cosmic-microwave-background'
#103
#link = 'https://www.sciencealert.com/could-humans-live-in-a-megasatellite-settlement-around-dwarf-planet-ceres'

### read data
req = urllib.request.Request(link, headers={'User-Agent' : "Magic Browser"})
scraped_data = urllib.request.urlopen(req)
article = scraped_data.read()

parsed_article = bs.BeautifulSoup(article,'lxml')
paragraphs = parsed_article.find_all('p')

for title in parsed_article.find_all('title'):
    news_title =title.get_text()

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

model = NERModel("bert", "/data/reshad/NLP/FakeNews/outputs-bio/checkpoint-180-epoch-15")   ### our data
#model = NERModel("bert", "/home/reshad/NLP/FakeNews/outputs-bio/checkpoint-180-epoch-15")   ### our data
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
      predictions, _ = model.predict(arg1[i])
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
    query = urllib.parse.quote(key_words[i])
    query1= 'all:' + query
    url = 'http://export.arxiv.org/api/query?search_query=' + query1 +'&start=0&max_results=10' 
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
    query1 = urllib.parse.quote(key_words[i])
    query11= 'all:' + query1
    query2= urllib.parse.quote(key_words[i+1])
    query22= 'all:' + query2
    query3 = urllib.parse.quote(key_words[i+2])
    query33= 'all:' + query3
    url3 = 'http://export.arxiv.org/api/query?search_query=' + query11 + 'AND'+ query22 + 'AND' + query33 + '&start=0&max_results=10'
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
'''
### Extracted news title and articles
reqs = requests.get(link)
# using the BeaitifulSoup module
soup = BeautifulSoup(reqs.text, 'html.parser')

for title in soup.find_all('title'):
    news_title=title.get_text()
'''
all_title=[news_title]
all_abstract=[article_text]
for i in range(len(non_duplicate_abstract)):
    all_abstract.append(non_duplicate_abstract[i])
    all_title.append(non_duplicate_titles[i])

print('\n')
print('Ranking based on Cosine similarity : SBert ')

#########################################
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from nltk.tokenize import word_tokenize, sent_tokenize
import nltk
nltk.download('punkt')
import pandas as pd
import numpy as np
from nltk.corpus import stopwords
import nltk
nltk.download('stopwords')
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics.pairwise import euclidean_distances
import torch
import json
import os

print('\n')
print('*********** Pretrained Google Word2Vec ****************')
print('\n')

from gensim.models import KeyedVectors
import numpy as np
# Load vectors directly from the file
model1 = KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin', binary=True)

def get_features(words):
    vectors=[]
    for i in words:
        try:
            vectors.append(model1[i])
        except:
            continue
    return vectors

def preprocess(raw_text):

    # keep only words
    letters_only_text = re.sub("[^a-zA-Z]", " ", raw_text)

    # convert to lower case and split
    words = letters_only_text.lower().split()

    # remove stopwords
    stopword_set = set(stopwords.words("english"))
    cleaned_words = list(set([w for w in words if w not in stopword_set]))

    return cleaned_words


def cosine_distance_wordembedding_method1(s1, s2):
    import scipy
    word1=preprocess(s1)
    word2=preprocess(s2)
    vec1=get_features(word1)
    vec2=get_features(word2)
    vector_1 = np.mean(vec1,axis=0)
    vector_2 = np.mean(vec2,axis=0)
    cosine = scipy.spatial.distance.cosine(vector_1, vector_2)
    return round((1-cosine)*100,2)

import re
from nltk.corpus import stopwords
import pandas as pd

results2=[]
num=0
pp= article_text
start_time1 = time.time()
for i in range(len(non_duplicate_abstract)):
    num=num+1
    #print(num)
    #results.append(cosine_distance_wordembedding_method1(gg, aa[i]))
    results2.append(cosine_distance_wordembedding_method1(pp, non_duplicate_abstract[i]))

ranks_list2={}

for i in range(len(non_duplicate_titles)):
    ranks_list2[non_duplicate_titles[i]]=results2[i]

#lists=sorted(ranks_list.items(), key= lamda x:x[1], reverse= True)

lists2=sorted(ranks_list2.items(), key=lambda x: x[1], reverse=True)

#print(lists)
for i in range(50):
    print(i,lists2[i])

print("--- %s seconds ---" % (time.time() - start_time))
print("--- %s seconds ---" % (time.time() - start_time1))

