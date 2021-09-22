from simpletransformers.ner import NERModel, NERArgs
import nltk
from nltk.tokenize import sent_tokenize
import os
model = NERModel("bert", "/home/reshad/fakenews/deeplearnig/outputs-bio/checkpoint-180-epoch-15")   ### our data
#model = NERModel("roberta", "/home/reshad/fakenews/deeplearnig/outputs-multidomian/checkpoint-210-epoch-15")   ### Multi domain Bert 
os.chdir('/home/reshad/fakenews/deeplearnig/other-student-dr.wu/Corpus/Testing Abstracts')

f=open("Abstract3.txt", "r")
text =f.read()
print(text)
'''
Abstract105.txt  Abstract141.txt  Abstract187.txt  Abstract248.txt  Abstract292.txt  Abstract32.txt   Abstract363.txt  Abstract3.txt    Abstract477.txt  Abstract60.txt
Abstract107.txt  Abstract144.txt  Abstract18.txt   Abstract250.txt  Abstract293.txt  Abstract332.txt  Abstract371.txt  Abstract408.txt  Abstract481.txt  Abstract71.txt
Abstract108.txt  Abstract145.txt  Abstract191.txt  Abstract251.txt  Abstract294.txt  Abstract335.txt  Abstract373.txt  Abstract416.txt  Abstract485.txt  Abstract72.txt
Abstract110.txt  Abstract164.txt  Abstract195.txt  Abstract255.txt  Abstract298.txt  Abstract337.txt  Abstract378.txt  Abstract419.txt  Abstract48.txt   Abstract77.txt
Abstract112.txt  Abstract166.txt  Abstract199.txt  Abstract265.txt  Abstract300.txt  Abstract339.txt  Abstract381.txt  Abstract422.txt  Abstract494.txt  Abstract81.txt
Abstract123.txt  Abstract16.txt   Abstract209.txt  Abstract266.txt  Abstract308.txt  Abstract349.txt  Abstract390.txt  Abstract435.txt  Abstract496.txt  Abstract82.txt
Abstract129.txt  Abstract172.txt  Abstract216.txt  Abstract271.txt  Abstract310.txt  Abstract351.txt  Abstract397.txt  Abstract442.txt  Abstract500.txt  Abstract83.txt
Abstract130.txt  Abstract173.txt  Abstract221.txt  Abstract281.txt  Abstract319.txt  Abstract354.txt  Abstract398.txt  Abstract45.txt   Abstract50.txt   Abstract88.txt
Abstract13.txt   Abstract184.txt  Abstract233.txt  Abstract282.txt  Abstract321.txt  Abstract355.txt  Abstract399.txt  Abstract466.txt  Abstract52.txt   Abstract89.txt
Abstract140.txt  Abstract185.txt  Abstract23.txt   Abstract28.txt   Abstract328.txt  Abstract362.txt  Abstract39.txt   Abstract476.txt  Abstract57.txt   Abstract9.txt

text= 'Dual-energy computed tomography (DECT) has been widely used in many applications that need material decomposition. Image-domain methods directly decompose material images from high- and low-energy attenuation images, and thus, are susceptible to noise and artifacts on attenuation images. To obtain high-quality material images, various data-driven methods have been proposed. Iterative neural network (INN) methods combine regression NNs and model-based image reconstruction algorithm. INNs reduced the generalization error of (noniterative) deep regression NNs, and achieved high-quality reconstruction in diverse medical imaging applications. BCD-Net is a recent INN architecture that incorporates imaging refining NNs into the block coordinate descent (BCD) model-based image reconstruction algorithm. We propose a new INN architecture, distinct cross-material BCD-Net, for DECT material decomposition. The proposed INN architecture uses distinct cross-material convolutional neural network (CNN) in image refining modules, and uses image decomposition physics in image reconstruction modules. The distinct cross-material CNN refiners incorporate distinct encoding-decoding filters and cross-material model that captures correlations between different materials. We interpret the distinct cross-material CNN refiner with patch perspective. Numerical experiments with extended cardiactorso (XCAT) phantom and clinical data show that proposed distinct cross-material BCD-Net significantly improves the image quality over several image-domain material decomposition methods, including a conventional model-based image decomposition (MBID) method using an edge-preserving regularizer, a state-of-the-art MBID method using pre-learned material-wise sparsifying transforms, and a noniterative deep CNN denoiser.'
'''
text = sent_tokenize(text)

test_data=[]
for i in range(len(text)):
    test_data.append([text[i]])


########## Function #############
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
######################
print('***************** Given Data *************')
print('\n')
#print(test_data)

pp=test_result(test_data)
p1= label_correction(pp)
pp= list_data(p1)

import itertools
import string
punctuation = set(string.punctuation)
dke = [[' '.join(w[0] for w in g) for k, g in itertools.groupby(sen, key=lambda x: x[0] and x[1] != 'O') if k] for sen in pp]
print(' ********************* Extracted DKE *****************')
print('\n')
print(dke)

