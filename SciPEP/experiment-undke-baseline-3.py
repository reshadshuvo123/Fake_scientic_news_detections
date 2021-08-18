import bs4 as bs
import urllib
#import urllib.request
import re
import nltk
import heapq
from sklearn_crfsuite import metrics
import heapq
import pickle
import sys
from pprint import pprint
import requests
from DataExtraction import convertCONLLFormJustExtractionSemEvalPerfile
from FeatureExtraction import sent2labels,sent2features
from PhraseEval import phrasesFromTestSenJustExtractionWithIndex
from nltk import word_tokenize, pos_tag, ne_chunk
from nltk import conlltags2tree, tree2conlltags
import os
import io
import unicodedata
import sys
import re
import string
import nltk
from nltk.parse import CoreNLPParser
from nltk.tokenize.punkt import PunktSentenceTokenizer, PunktParameters
from nltk.tokenize import word_tokenize,WhitespaceTokenizer
import config

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
#link = 'https://www.forbes.com/sites/williamfalcon/2018/09/01/facebook-ai-just-set-a-new-record-in-translation-and-why-it-matters/?sh=363ea1723124'
#33
#link = 'https://techxplore.com/news/2019-05-pal-wearable-context-aware-health-cognition.html'
#34
#link = 'https://www.sciencealert.com/neural-networks-performance-increases-if-they-re-allowed-to-sleep-and-dream'
#35
#link = 'https://www.sciencealert.com/hot-jupiter-wasp-104b-one-of-the-darkest-planets-ever'
#36
#link='https://allthatsinteresting.com/brainnet-three-brains-connected'
##37
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
#link = 'https://www.the-sun.com/lifestyle/tech/480269/first-known-alien-protein-unlike-anything-on-earth-found-inside-meteorite-by-scientists/'
# 72
#link ='https://www.sciencealert.com/human-consciousness-could-be-a-result-of-entropy-study-science'
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
link = 'https://www.sciencealert.com/ai-can-now-enhance-pixelated-photos-to-make-them-over-60-times-sharper'
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


def get_leaves(ptree):
    for subtree in ptree.subtrees(filter=lambda t: t.label() == 'NP' or t.label() == 'PP' or t.label() == 'VP'):
        yield (subtree.treeposition(), subtree.label(), subtree.leaves())

def unsupervised_dke(data):
    # Define stopword list
    stopwords = nltk.corpus.stopwords.words('english')

    # Initialize Stanford POS Tagger
    parser = CoreNLPParser(url=config.corenlpserverurl)
    pos_tagger = CoreNLPParser(url=config.corenlpserverurl, tagtype='pos')

    # Nltk sentence split based on periods.
    punkt_param = PunktParameters()
    # avoid i.e., e.g., Fig. or Tab. being split in separate sentences
    punkt_param.abbrev_types = set(['i.e', 'e.g', 'fig', 'tab', 'no', 'et', 'al', 'Figs'])
    sentence_splitter = PunktSentenceTokenizer(punkt_param)

    text = data

    toktext = sentence_splitter.tokenize(text)
    s_spans = sentence_splitter.span_tokenize(text)
    sentence_spans = []
    for ss in s_spans:
        sss = []
        start = ss[0]
        end = ss[1]

        sss.append(start)
        sss.append(end)
        sentence_spans.append(sss)

    z = 0
    cha = []
    dha = []
    for s in toktext:
        sentence_re = r'''(?x)        # set flag to allow verbose regexps
                    (?:[A-Z])(?:\.[A-Z])+\.?    # abbreviations, e.g. U.S.A.
                    | \w+(?:-\w+)*            # words with optional internal hyphens
                    | \$?\d+(?:\.\d+)?%?        # currency and percentages, e.g. $12.40, 82%
                    | \.\.\.                # ellipsis
                    | [][.,;"'?():-_`]        # these are separate tokens
                '''
        tokenizer = WhitespaceTokenizer()
        tokenwords = WhitespaceTokenizer().tokenize(s)
        t_spans = tokenizer.span_tokenize(s)
        t_spans_l = []
        word_spans_ = []
        for w in t_spans:
            t_spans_l.append(w)
        k = 0
        for t in tokenwords:
            w = t_spans_l[k]
            try:
                wss = []
                length = w[1] - w[0]
                if z == 0:  # first sentence
                    start = w[0]
                    end = w[1]
                    end = start + length
                else:
                    if w[0] == 0:
                        start = sentence_spans[z][0]
                        end = start + w[1]
                        newstart = end + 1
                    else:
                        start = newstart
                        end = start + length
                        newstart = end + 1
                if t[len(t) - 1] == ',' or t[len(t) - 1] == '.':
                    end = end - 1
                wss.append(start)
                wss.append(end)
                word_spans_.append(wss)
                k += 1
            except:
                k += 1
                continue
        grammar = r"""
                NP: {<NN.*|JJ>*<NN.*>} # NP
                PP: {<IN> <NP>}      # PP -> P NP
                VP: {<V.*> <NP|PP>*}  # VP -> V (NP|PP)*
                """
        chunker = nltk.RegexpParser(grammar)
        postag = list(pos_tagger.tag(tokenwords))
        tree = chunker.parse(postag)
        partree = nltk.ParentedTree.convert(tree)
        positions = ()
        phrases = []
        temp = -1
        leaves = get_leaves(partree)
        cleaves = get_leaves(partree)
        try:
            next(cleaves)
            for l in leaves:
                current = l[0][0]
                try:
                    m = next(cleaves)
                    temp = m[0][0]
                    try:
                        check = l[0][1]
                        current2 = l[0][1]
                        temp2 = m[0][1]
                        if current == temp and current2 == temp2:
                            try:
                                check = l[0][2]
                                for n in l[2]:
                                    phrases.append(l[1])
                                    phrases.append(n)
                            except:
                                phrases.append(l[1])
                                phrases.append(l[2][0])
                        else:
                            for n in l[2]:
                                phrases.append(l[1])
                                phrases.append(n)
                    except:
                        if current != temp:
                            for n in l[2]:
                                phrases.append(l[1])
                                phrases.append(n)
                        else:
                            phrases.append(l[1])
                            phrases.append(l[2][0])
                except:
                    for n in l[2]:
                        phrases.append(l[1])
                        phrases.append(n)
        except:
            print(tokenwords)
            print(partree)
        phraseout = []
        ptagout = []
        for j in range(1, len(phrases), 2):
            try:
                phraseout.append(phrases[j][0])
            except:
                break
        for j in range(0, len(phrases), 2):
            try:
                ptagout.append(phrases[j])
            except:
                break
        z += 1
        cha.append(phrases)
        dha.append(phraseout)

    dke_result=[]
    for i in range(len(dha)):
        for j in range(len(dha[i])):
            dke_result.append(dha[i][j])
    return dke_result

eke=unsupervised_dke(processed_text)

from nltk.tokenize import sent_tokenize
sent_tokenize_list = sent_tokenize(processed_text)
#print sent_tokenize_list


tree6 = [ne_chunk(pos_tag(word_tokenize(x))) for x in sent_tokenize_list]

##### Domain entities extraction

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

crf= pickle.load(open("linear-chain-crf.model.pickle", 'rb'))
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
key_pharse_eke = [w for w in eke if not w in stopwords]

##### Grouping Words #####

import itertools
import string
data=test_sents_pls
punctuation = set(string.punctuation)
sentences = [[' '.join(w[0] for w in g) for k, g in itertools.groupby(sen, key=lambda x: x[0] not in punctuation and x[2] != 'O') if k] for sen in data]

#print sentences

key_words1=[]
for se in sentences:
    for i in se:
        dkp=i.encode("utf-8")
        key_words1.append(i)
#key_words= set(key_words)

#key_wrods1=[str(r) for r in key_words]

print('\n')
print (' ************** Key Words DENS ***************')

print('\n')


final_dke=key_words1 + key_pharse_eke
set1=set(final_dke)
key_words=[]
for i in set1:
    key_words.append(i)
print(key_words)

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
print('Ranking based on Cosine similarity : ')

################## Rerank/learning to rank ###################
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import time

#start = time.time()

tfidf_vectorizer = TfidfVectorizer()
tfidf_matrix = tfidf_vectorizer.fit_transform(all_abstract)
bb=[cosine_similarity(tfidf_matrix[0:1], tfidf_matrix)]

ranks_list={}

for i in range(len(non_duplicate_titles)):
    ranks_list[non_duplicate_titles[i]]=bb[0][0][i+1]

#lists=sorted(ranks_list.items(), key= lamda x:x[1], reverse= True)

lists=sorted(ranks_list.items(), key=lambda x: x[1], reverse=True)

#print(lists)
for i in range(50):
    try:
        print(i+1,lists[i])
    except:
        break


