from __future__ import division
import numpy as np
import copy
import random
from pprint import pprint

def getPhraseTokens(bDs,iDs,senLength):
    bpTokens=[]
    for i in range(len(bDs)):
        start = bDs[i][0]
        end = senLength
        if i < len(bDs)-1:
            end = bDs[i+1][0]  
        phrase =  ' '.join([x[1] for x in [bDs[i]]+[x for x in iDs if x[0] > start and x[0] < end]])
        bpTokens.append(phrase)
        
    return bpTokens

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
            
def phrasesFromTestSenJustExtractionWithIndex(sen,tokenIndices):
    sen = [(x[0],x[1],x[2],int(y.split(",")[0]),int(y.split(",")[1])) for (x,y) in zip(sen,tokenIndices)]
    bS = sorted([(i,x[0],x[-2],x[-1]) for (i,x) in enumerate(sen) if x[2] == "B"], key = lambda x:x[0])
    iS = sorted([(i,x[0],x[-2],x[-1]) for (i,x) in enumerate(sen) if x[2] == "I"], key = lambda x:x[0])

    tSen = copy.deepcopy(sen)
    tSen.append(
    {
    'phrases': getPhraseTokensWithIndex(bS,iS,len(sen)),
    }
    )
    return tSen

def phrasesFromTestSenJustExtraction(sen):
    bS = sorted([(i,x[0]) for (i,x) in enumerate(sen) if x[2] == "B"], key = lambda x:x[0])
    iS = sorted([(i,x[0]) for (i,x) in enumerate(sen) if x[2] == "I"], key = lambda x:x[0])
   
    tSen = copy.deepcopy(sen)
    tSen.append(
    {
    'phrases': getPhraseTokens(bS,iS,len(sen)),
    }
    )
    return tSen




def matchAbs(i1,i2):
    return i1.lower() == i2.lower()

def matchIn(iP,iG):
    return iG.lower() in iP.lower()

def calc_result(gl,pl,style):
    #gL = [a,b,c], pL = [d,c,a,e]. match = 2, precision = 2/4, recall = 2/3
    if not gl and not pl:
        return None
    if (gl and not pl) or (pl and not gl):
        return (0.0,0.0)        
    nmatch = 0
    for i1 in pl:
        for i2 in gl:
            if match(i1,i2):
                nmatch += 1
    return (nmatch/len(pl),nmatch/len(gl))     
    
def phrase_extraction_report(gl,pl):
    gl = list(set([x.lower() for x in gl]))
    pl = list(set([x.lower() for x in pl])) 
    print("phrase extraction results: gold standard: ",len(gl),"phrases, predicted: ",len(pl),"phrases")
    nmatch = 0
    matches = []
    for i1 in pl:
        for i2 in gl:
            if matchAbs(i1,i2): 
                matches.append({'predicted':i1,'gold':i2})
                nmatch += 1
                break
    random.shuffle(matches)
    print(nmatch,"phrases matched")
    #pprint(matches[:10])
    return {'precision':nmatch/len(pl),'recall':nmatch/len(gl)}

