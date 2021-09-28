import os
import time
import gensim
from collections import Counter
import torch
from torch import nn
from torch.optim import Adam
from torchtext.data import Field, BucketIterator
from torchtext.datasets import SequenceTaggingDataset
from torchtext.vocab import Vocab
from spacy.lang.id import Indonesian
import gensim.models.keyedvectors as word2vec
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
import re
from seqeval.metrics import precision_score, recall_score, f1_score, classification_report

import math
import time
import gensim
import torch
import matplotlib.pyplot as plt
from torch import nn
from torch.optim import Adam
from torchtext.data import Field, NestedField, BucketIterator
from torchtext.datasets import SequenceTaggingDataset
from torchtext.vocab import Vocab
from torchcrf import CRF
from collections import Counter
from spacy.lang.id import Indonesian

DRIVE_ROOT = "/home/reshad/fakenews/deeplearnig/OA-STM-domains/all-data/dd"

available_gpu = torch.cuda.is_available()
if available_gpu:
    print(f"GPU is available: {torch.cuda.get_device_name(0)}")
    use_device = torch.device("cuda")
else:
    use_device = torch.device("cpu")


class Corpus(object):

    def __init__(self, input_folder, min_word_freq, batch_size, wv_file=None):
        # list all the fields
        self.word_field = Field(lower=True)  # [sent len, batch_size]
        self.tag_field = Field(unk_token=None)  # [sent len, batch_size]
        # Character-level input
        self.char_nesting_field = Field(tokenize=list)
        self.char_field = NestedField(self.char_nesting_field)  # [batch_size, sent len, max len char]
        # create dataset using built-in parser from torchtext
        self.train_dataset, self.val_dataset, self.test_dataset = SequenceTaggingDataset.splits(
            path=input_folder,
            train="train-new.tsv",
            validation="train-new.tsv",
            test="test-new.tsv",
            fields=(
                (("word", "char"), (self.word_field, self.char_field)),
                ("tag", self.tag_field)
            )
        )
        # convert fields to vocabulary list
        if wv_file:
            #self.wv_model = gensim.models.word2vec.Word2Vec.load(wv_file)
            self.wv_model = word2vec.KeyedVectors.load_word2vec_format(wv_file, binary=True)
            self.embedding_dim = self.wv_model.vector_size
            word_freq = {word: self.wv_model.wv.vocab[word].count for word in self.wv_model.wv.vocab}
            word_counter = Counter(word_freq)
            self.word_field.vocab = Vocab(word_counter, min_freq=min_word_freq)
            vectors = []
            for word, idx in self.word_field.vocab.stoi.items():
                if word in self.wv_model.wv.vocab.keys():
                    vectors.append(torch.as_tensor(self.wv_model.wv[word].tolist()))
                else:
                    vectors.append(torch.zeros(self.embedding_dim))
            self.word_field.vocab.set_vectors(
                stoi=self.word_field.vocab.stoi,
                vectors=vectors,
                dim=self.embedding_dim
            )
        else:
            self.word_field.build_vocab(self.train_dataset.word, min_freq=min_word_freq)
        # build vocab for tag and characters
        self.char_field.build_vocab(self.train_dataset.char)
        self.tag_field.build_vocab(self.train_dataset.tag)
        # create iterator for batch input
        self.train_iter, self.val_iter, self.test_iter = BucketIterator.splits(
            datasets=(self.train_dataset, self.val_dataset, self.test_dataset),
            batch_size=batch_size
        )
        # prepare padding index to be ignored during model training/evaluation
        self.word_pad_idx = self.word_field.vocab.stoi[self.word_field.pad_token]
        self.char_pad_idx = self.char_field.vocab.stoi[self.char_field.pad_token]
        self.tag_pad_idx = self.tag_field.vocab.stoi[self.tag_field.pad_token]


corpus = Corpus(
    input_folder=f"{DRIVE_ROOT}",
    min_word_freq=3,
    batch_size=64,
    #wv_file=f"{DRIVE_ROOT}/GoogleNews-vectors-negative300.bin/GoogleNews-vectors-negative300.bin"
    wv_file = '/home/reshad/fakenews/deeplearnig/OA-STM-domains/all-data/dd/GoogleNews-vectors-negative300.bin'
)
print(f"Train set: {len(corpus.train_dataset)} sentences")
print(f"Val set: {len(corpus.val_dataset)} sentences")
print(f"Test set: {len(corpus.test_dataset)} sentences")

### BEGIN MODIFIED SECTION: TRANSFORMER ###
# source: https://pytorch.org/tutorials/beginner/transformer_tutorial.html
class PositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout=0.1, max_len=500):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)
### END MODIFIED SECTION ###

class Transformer(nn.Module):

    def __init__(self,
                 input_dim,
                 embedding_dim,
                 char_emb_dim,
                 char_input_dim,
                 char_cnn_filter_num,
                 char_cnn_kernel_size,
                 attn_heads,
                 fc_hidden,
                 trf_layers,
                 output_dim,
                 emb_dropout,
                 cnn_dropout,
                 trf_dropout,
                 fc_dropout,
                 word_pad_idx,
                 char_pad_idx,
                 tag_pad_idx,
                 device):  # NEWLY ADDED: GPU
        super().__init__()
        self.char_pad_idx = char_pad_idx
        self.word_pad_idx = word_pad_idx
        self.tag_pad_idx = tag_pad_idx
        self.device = device  # NEWLY ADDED: GPU
        # LAYER 1A: Word Embedding
        self.embedding_dim = embedding_dim
        self.embedding = nn.Embedding(
            num_embeddings=input_dim,
            embedding_dim=embedding_dim,
            padding_idx=word_pad_idx
        )
        self.emb_dropout = nn.Dropout(emb_dropout)
        # LAYER 1B: Char Embedding-CNN
        self.char_emb_dim = char_emb_dim
        self.char_emb = nn.Embedding(
            num_embeddings=char_input_dim,
            embedding_dim=char_emb_dim,
            padding_idx=char_pad_idx
        )
        self.char_cnn = nn.Conv1d(
            in_channels=char_emb_dim,
            out_channels=char_emb_dim * char_cnn_filter_num,
            kernel_size=char_cnn_kernel_size,
            groups=char_emb_dim  # different 1d conv for each embedding dim
        )
        self.cnn_dropout = nn.Dropout(cnn_dropout)
        ### BEGIN MODIFIED SECTION: TRANSFORMER ###
        # LAYER 2: Transformer
        all_emb_size = embedding_dim + (char_emb_dim * char_cnn_filter_num)
        self.position_encoder = PositionalEncoding(
            d_model=all_emb_size
        )
        encoder_layers = nn.TransformerEncoderLayer(
            d_model=all_emb_size,
            nhead=attn_heads,
            activation="relu",
            dropout=trf_dropout
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer=encoder_layers,
            num_layers=trf_layers
        )
        # LAYER 3: 2-layers fully-connected with GELU activation in-between
        self.fc1 = nn.Linear(
            in_features=all_emb_size,
            out_features=fc_hidden
        )
        self.fc1_gelu = nn.GELU()
        self.fc1_norm = nn.LayerNorm(fc_hidden)
        self.fc2_dropout = nn.Dropout(fc_dropout)
        self.fc2 = nn.Linear(
            in_features=fc_hidden,
            out_features=output_dim
        )
        ### END MODIFIED SECTION ###
        # LAYER 4: CRF
        self.crf = CRF(num_tags=output_dim)
        # init weights from normal distribution
        for name, param in self.named_parameters():
            nn.init.normal_(param.data, mean=0, std=0.1)

    def forward(self, words, chars, tags=None):
        # words = [sentence length, batch size]
        # chars = [batch size, sentence length, word length)
        # tags = [sentence length, batch size]
        # embedding_out = [sentence length, batch size, embedding dim]
        embedding_out = self.emb_dropout(self.embedding(words))
        # character cnn layer forward
        # reference: https://github.com/achernodub/targer/blob/master/src/layers/layer_char_cnn.py
        # char_emb_out = [batch size, sentence length, word length, char emb dim]
        char_emb_out = self.emb_dropout(self.char_emb(chars))
        batch_size, sent_len, word_len, char_emb_dim = char_emb_out.shape
        char_cnn_max_out = torch.zeros(batch_size, sent_len, self.char_cnn.out_channels, device=self.device)  # NEWLY MODIFIED: GPU
        for sent_i in range(sent_len):
            # sent_char_emb = [batch size, word length, char emb dim]
            sent_char_emb = char_emb_out[:, sent_i, :, :]
            # sent_char_emb_p = [batch size, char emb dim, word length]
            sent_char_emb_p = sent_char_emb.permute(0, 2, 1)
            # char_cnn_sent_out = [batch size, out channels * char emb dim, word length - kernel size + 1]
            char_cnn_sent_out = self.char_cnn(sent_char_emb_p)
            char_cnn_max_out[:, sent_i, :], _ = torch.max(char_cnn_sent_out, dim=2)
        char_cnn = self.cnn_dropout(char_cnn_max_out)
        # concat word and char embedding
        # char_cnn_p = [sentence length, batch size, char emb dim * num filter]
        char_cnn_p = char_cnn.permute(1, 0, 2)
        word_features = torch.cat((embedding_out, char_cnn_p), dim=2)
        ### BEGIN MODIFIED SECTION: TRANSFORMER ###
        # Transformer
        key_padding_mask = torch.as_tensor(words == self.word_pad_idx).permute(1, 0)
        # pos_out = [sentence length, batch size, embedding dim + char emb dim * num filter]
        pos_out = self.position_encoder(word_features)
        # enc_out = [sentence length, batch size, embedding dim + char emb dim * num filter]
        enc_out = self.encoder(pos_out, src_key_padding_mask=key_padding_mask)
        # Fully-connected
        # fc1_out = [sentence length, batch size, fc hidden]
        fc1_out = self.fc1_norm(self.fc1_gelu(self.fc1(enc_out)))
        # fc2_out = [sentence length, batch size, output dim]
        fc2_out = self.fc2(self.fc2_dropout(fc1_out))
        ### END MODIFIED SECTION ###
        # CRF
        crf_mask = words != self.word_pad_idx
        crf_out = self.crf.decode(fc2_out, mask=crf_mask)
        crf_loss = -self.crf(fc2_out, tags=tags, mask=crf_mask) if tags is not None else None
        return crf_out, crf_loss

    def init_embeddings(self, pretrained=None, freeze=True):
        # initialize embedding for padding as zero
        self.embedding.weight.data[self.word_pad_idx] = torch.zeros(self.embedding_dim)
        self.char_emb.weight.data[self.char_pad_idx] = torch.zeros(self.char_emb_dim)
        if pretrained is not None:
            self.embedding = nn.Embedding.from_pretrained(
                embeddings=torch.as_tensor(pretrained),
                padding_idx=self.word_pad_idx,
                freeze=freeze
            )
    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

transformer = Transformer(
    input_dim=len(corpus.word_field.vocab),
    embedding_dim=300,
    char_emb_dim=37,  # NEWLY MODIFIED: TRANSFORMER
    char_input_dim=len(corpus.char_field.vocab),
    char_cnn_filter_num=4,  # NEWLY MODIFIED: TRANSFORMER
    char_cnn_kernel_size=3,
    attn_heads=16,  # NEWLY MODIFIED: TRANSFORMER
    fc_hidden=256,  # NEWLY MODIFIED: TRANSFORMER
    trf_layers=1,
    output_dim=len(corpus.tag_field.vocab),
    emb_dropout=0.5,
    cnn_dropout=0.25,
    trf_dropout=0.1,  # NEWLY MODIFIED: TRANSFORMER
    fc_dropout=0.25,
    word_pad_idx=corpus.word_pad_idx,
    char_pad_idx=corpus.char_pad_idx,
    tag_pad_idx=corpus.tag_pad_idx,
    device=use_device
)
transformer.init_embeddings(
    pretrained=corpus.word_field.vocab.vectors if corpus.wv_model else None,
    freeze=True
)

print(f"The model has {transformer.count_parameters():,} trainable parameters.")
print(transformer)

class Trainer(object):

    def __init__(self, model, data, optimizer_cls, device):  # NEWLY MODIFIED: GPU
        self.device = device  # NEWLY ADDED: GPU
        self.model = model.to(self.device)  # NEWLY MODIFIED: GPU
        self.data = data
        self.optimizer = optimizer_cls(model.parameters())

    @staticmethod
    def epoch_time(start_time, end_time):
        elapsed_time = end_time - start_time
        elapsed_mins = int(elapsed_time / 60)
        elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
        return elapsed_mins, elapsed_secs

    def accuracy(self, preds, y):
        flatten_preds = [pred for sent_pred in preds for pred in sent_pred]
        flatten_y = [tag for sent_tag in y for tag in sent_tag]
        correct = [pred == tag for pred, tag in zip(flatten_preds, flatten_y)]
        return sum(correct) / len(correct) if len(correct) > 0 else 0

    def epoch(self):
        epoch_loss = 0
        epoch_acc = 0
        self.model.train()
        for batch in self.data.train_iter:
            # words = [sent len, batch size]
            words = batch.word.to(self.device)  # NEWLY MODIFIED: GPU
            # chars = [batch size, sent len, char len]
            chars = batch.char.to(self.device)  # NEWLY MODIFIED: GPU
            # tags = [sent len, batch size]
            true_tags = batch.tag.to(self.device)  # NEWLY MODIFIED: GPU
            self.optimizer.zero_grad()
            pred_tags_list, batch_loss = self.model(words, chars, true_tags)
            # to calculate the loss and accuracy, we flatten true tags
            true_tags_list = [
                [tag for tag in sent_tag if tag != self.data.tag_pad_idx]
                for sent_tag in true_tags.permute(1, 0).tolist()
            ]
            batch_acc = self.accuracy(pred_tags_list, true_tags_list)
            batch_loss.backward()
            self.optimizer.step()
            epoch_loss += batch_loss.item()
            epoch_acc += batch_acc
        return epoch_loss / len(self.data.train_iter), epoch_acc / len(self.data.train_iter)

    def evaluate(self, iterator):
        epoch_loss = 0
        epoch_acc = 0
        self.model.eval()
        with torch.no_grad():
            # similar to epoch() but model is in evaluation mode and no backprop
            for batch in iterator:
                words = batch.word.to(self.device)  # NEWLY MODIFIED: GPU
                chars = batch.char.to(self.device)  # NEWLY MODIFIED: GPU
                true_tags = batch.tag.to(self.device)  # NEWLY MODIFIED: GPU
                pred_tags, batch_loss = self.model(words, chars, true_tags)
                true_tags_list = [
                    [tag for tag in sent_tag if tag != self.data.tag_pad_idx]
                    for sent_tag in true_tags.permute(1, 0).tolist()
                ]
                batch_acc = self.accuracy(pred_tags, true_tags_list)
                epoch_loss += batch_loss.item()
                epoch_acc += batch_acc
        return epoch_loss / len(iterator), epoch_acc / len(iterator)

    def train(self, n_epochs):
        for epoch in range(n_epochs):
            start_time = time.time()
            train_loss, train_acc = self.epoch()
            end_time = time.time()
            epoch_mins, epoch_secs = Trainer.epoch_time(start_time, end_time)
            print(f"Epoch: {epoch + 1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s")
            print(f"\tTrn Loss: {train_loss:.3f} | Trn Acc: {train_acc * 100:.2f}%")
            val_loss, val_acc = self.evaluate(self.data.val_iter)
            print(f"\tVal Loss: {val_loss:.3f} | Val Acc: {val_acc * 100:.2f}%")
        test_loss, test_acc = self.evaluate(self.data.test_iter)
        print(f"Test Loss: {test_loss:.3f} |  Test Acc: {test_acc * 100:.2f}%")

    def infer(self, sentence, true_tags=None):
        self.model.eval()
        # tokenize sentence
        nlp = Indonesian()
        tokens = [token.text for token in nlp(sentence)]
        max_word_len = max([len(token) for token in tokens])
        # transform to indices based on corpus vocab
        numericalized_tokens = [self.data.word_field.vocab.stoi[token.lower()] for token in tokens]
        numericalized_chars = []
        char_pad_id = self.data.char_pad_idx
        for token in tokens:
            numericalized_chars.append(
                [self.data.char_field.vocab.stoi[char] for char in token]
                + [char_pad_id for _ in range(max_word_len - len(token))]
            )
        # find unknown words
        unk_idx = self.data.word_field.vocab.stoi[self.data.word_field.unk_token]
        unks = [t for t, n in zip(tokens, numericalized_tokens) if n == unk_idx]
        # begin prediction
        token_tensor = torch.as_tensor(numericalized_tokens)
        token_tensor = token_tensor.unsqueeze(-1).to(self.device)  # NEWLY MODIFIED: GPU
        char_tensor = torch.as_tensor(numericalized_chars)
        char_tensor = char_tensor.unsqueeze(0).to(self.device)  # NEWLY MODIFIED: GPU
        predictions, _ = self.model(token_tensor, char_tensor)
        # convert results to tags
        predicted_tags = [self.data.tag_field.vocab.itos[t] for t in predictions[0]]
        # print inferred tags
        max_len_token = max([len(token) for token in tokens] + [len('word')])
        max_len_tag = max([len(tag) for tag in predicted_tags] + [len('pred')])
        '''
        #print(
            f"{'word'.ljust(max_len_token)}\t{'unk'.ljust(max_len_token)}\t{'pred tag'.ljust(max_len_tag)}"
            + ("\ttrue tag" if true_tags else "")
        )
        for i, token in enumerate(tokens):
            is_unk = "✓" if token in unks else ""
            print(
                f"{token.ljust(max_len_token)}\t{is_unk.ljust(max_len_token)}\t{predicted_tags[i].ljust(max_len_tag)}"
                + (f"\t{true_tags[i]}" if true_tags else "")
            )
        '''
        return predicted_tags

trainer = Trainer(
    model=transformer,
    data=corpus,
    optimizer_cls=Adam,
    device=use_device
)
trainer.train(75)

####################### Function ##################

def cuu(nanjo):
    cc = []
    cc1 = []
    for i in nanjo:
        if i != ('.', 'O') and i != ('',''):
            cc1.append(i)
            # print(cc1)
        else:
            cc1.append(('.', 'O'))
            cc.append(cc1)
            cc1 = []
    return cc

def test_data(cc2):
   aa = []
   pp = []
   for i in range(len(cc2)):
      for j in range(len(cc2[i])):
         a = cc2[i][j][0]
         aa.append(a)
      p = " ".join(aa)
      aa = []
      pp.append([p])
   return pp

def test_result(arg1):
    cc=[]
    gg=[]
    for i in range(len(arg1)):
        taggs = trainer.infer(sentence=arg1[i][0], true_tags=None)
        cc.append(taggs)
    return cc

print('################ Testing #####################')
print('*********************** Arg ******************************')
import os
import re

words1=[]
wordtags1=[]
nanjo1=[]
os.chdir('/home/reshad/fakenews/deeplearnig/OA-STM-domains/all-data/Arg')
x1=open('test_com1.txt',encoding="utf8")
for i in x1:
    #i=re.sub(r'[^\w\s]','',i)
    a=i.split('\t')
    w1=a[0].rstrip()
    #words1.append(w1)
    if len(a) <= 1:  # deal with the '\n' in the first row
        continue
    else:
        wt = a[1].rstrip()
    words1.append(w1)
    wordtags1.append(wt)
    na=(w1,wt)
    if na !=('', '\n'):
        if na != ('', ''):
            nanjo1.append(na)


arg =cuu(nanjo1)
arg1=test_data(arg)
pred_labels1 = test_result(arg1)
true_labels1 = [[w[1] for w in s] for s in arg]

print("Precision-score: {:.1%}".format(precision_score(true_labels1, pred_labels1)))
print("Recal-score: {:.1%}".format(recall_score(true_labels1, pred_labels1)))
print("F1-score: {:.1%}".format(f1_score(true_labels1, pred_labels1)))

print('*********************** Astr ******************************')
import os
words1=[]
wordtags1=[]
nanjo1=[]
os.chdir('/home/reshad/fakenews/deeplearnig/OA-STM-domains/all-data/Astr')
x1=open('test_com2.txt',encoding="utf8")
for i in x1:
    #i=re.sub(r'[^\w\s]','',i)
    a=i.split('\t')
    w1=a[0].rstrip()
    #words1.append(w1)
    if len(a) <= 1:  # deal with the '\n' in the first row
        continue
    else:
        wt = a[1].rstrip()
    words1.append(w1)
    wordtags1.append(wt)
    na=(w1,wt)
    if na !=('', '\n'):
        if na != ('', ''):
            nanjo1.append(na)


astr =cuu(nanjo1)
astr1=test_data(astr)
pred_labels1 = test_result(astr1)
true_labels1 = [[w[1] for w in s] for s in astr]

print("Precision-score: {:.1%}".format(precision_score(true_labels1, pred_labels1)))
print("Recal-score: {:.1%}".format(recall_score(true_labels1, pred_labels1)))
print("F1-score: {:.1%}".format(f1_score(true_labels1, pred_labels1)))

print('*********************** Bio ******************************')
import os
words1=[]
wordtags1=[]
nanjo1=[]
os.chdir('/home/reshad/fakenews/deeplearnig/OA-STM-domains/all-data/Bio')
x1=open('test_com3.txt',encoding="utf8")
for i in x1:
    #i=re.sub(r'[^\w\s]','',i)
    a=i.split('\t')
    w1=a[0].rstrip()
    #words1.append(w1)
    if len(a) <= 1:  # deal with the '\n' in the first row
        continue
    else:
        wt = a[1].rstrip()
    words1.append(w1)
    wordtags1.append(wt)
    na=(w1,wt)
    if na !=('', '\n'):
        if na != ('', ''):
            nanjo1.append(na)


bio =cuu(nanjo1)
bio1=test_data(bio)
pred_labels1 = test_result(bio1)
true_labels1 = [[w[1] for w in s] for s in bio]

print("Precision-score: {:.1%}".format(precision_score(true_labels1, pred_labels1)))
print("Recal-score: {:.1%}".format(recall_score(true_labels1, pred_labels1)))
print("F1-score: {:.1%}".format(f1_score(true_labels1, pred_labels1)))

print('*********************** Chem ******************************')
import os
words1=[]
wordtags1=[]
nanjo1=[]
os.chdir('/home/reshad/fakenews/deeplearnig/OA-STM-domains/all-data/Chem')
x1=open('test_com4.txt',encoding="utf8")
for i in x1:
    #i=re.sub(r'[^\w\s]','',i)
    a=i.split('\t')
    w1=a[0].rstrip()
    #words1.append(w1)
    if len(a) <= 1:  # deal with the '\n' in the first row
        continue
    else:
        wt = a[1].rstrip()
    words1.append(w1)
    wordtags1.append(wt)
    na=(w1,wt)
    if na !=('', '\n'):
        if na != ('', ''):
            nanjo1.append(na)


chem =cuu(nanjo1)
chem1=test_data(chem)
pred_labels1 = test_result(chem1)
true_labels1 = [[w[1] for w in s] for s in chem]

print("Precision-score: {:.1%}".format(precision_score(true_labels1, pred_labels1)))
print("Recal-score: {:.1%}".format(recall_score(true_labels1, pred_labels1)))
print("F1-score: {:.1%}".format(f1_score(true_labels1, pred_labels1)))

print('*********************** CS ******************************')
import os
words1=[]
wordtags1=[]
nanjo1=[]
os.chdir('/home/reshad/fakenews/deeplearnig/OA-STM-domains/all-data/CS')
x1=open('test_com5.txt',encoding="utf8")
for i in x1:
    #i=re.sub(r'[^\w\s]','',i)
    a=i.split('\t')
    w1=a[0].rstrip()
    #words1.append(w1)
    if len(a) <= 1:  # deal with the '\n' in the first row
        continue
    else:
        wt = a[1].rstrip()
    words1.append(w1)
    wordtags1.append(wt)
    na=(w1,wt)
    if na !=('', '\n'):
        if na != ('', ''):
            nanjo1.append(na)


cs =cuu(nanjo1)
cs1=test_data(cs)
pred_labels1 = test_result(cs1)
true_labels1 = [[w[1] for w in s] for s in cs]

print("Precision-score: {:.1%}".format(precision_score(true_labels1, pred_labels1)))
print("Recal-score: {:.1%}".format(recall_score(true_labels1, pred_labels1)))
print("F1-score: {:.1%}".format(f1_score(true_labels1, pred_labels1)))

print('*********************** ES ******************************')
import os
words1=[]
wordtags1=[]
nanjo1=[]
os.chdir('/home/reshad/fakenews/deeplearnig/OA-STM-domains/all-data/ES')
x1=open('test_com7.txt',encoding="utf8")
for i in x1:
    #i=re.sub(r'[^\w\s]','',i)
    a=i.split('\t')
    w1=a[0].rstrip()
    #words1.append(w1)
    if len(a) <= 1:  # deal with the '\n' in the first row
        continue
    else:
        wt = a[1].rstrip()
    words1.append(w1)
    wordtags1.append(wt)
    na=(w1,wt)
    if na !=('', '\n'):
        if na != ('', ''):
            nanjo1.append(na)


es =cuu(nanjo1)
es1=test_data(es)
pred_labels1 = test_result(es1)
true_labels1 = [[w[1] for w in s] for s in es]

print("Precision-score: {:.1%}".format(precision_score(true_labels1, pred_labels1)))
print("Recal-score: {:.1%}".format(recall_score(true_labels1, pred_labels1)))
print("F1-score: {:.1%}".format(f1_score(true_labels1, pred_labels1)))



print('*********************** Eng ******************************')
import os
words1=[]
wordtags1=[]
nanjo1=[]
os.chdir('/home/reshad/fakenews/deeplearnig/OA-STM-domains/all-data/Eng')
x1=open('test_com6.txt',encoding="utf8")
for i in x1:
    #i=re.sub(r'[^\w\s]','',i)
    a=i.split('\t')
    w1=a[0].rstrip()
    #words1.append(w1)
    if len(a) <= 1:  # deal with the '\n' in the first row
        continue
    else:
        wt = a[1].rstrip()
    words1.append(w1)
    wordtags1.append(wt)
    na=(w1,wt)
    if na !=('', '\n'):
        if na != ('', ''):
            nanjo1.append(na)


eng =cuu(nanjo1)
eng1=test_data(eng)
pred_labels1 = test_result(eng1)
true_labels1 = [[w[1] for w in s] for s in eng]

print("Precision-score: {:.1%}".format(precision_score(true_labels1, pred_labels1)))
print("Recal-score: {:.1%}".format(recall_score(true_labels1, pred_labels1)))
print("F1-score: {:.1%}".format(f1_score(true_labels1, pred_labels1)))

print('*********************** Math ******************************')
import os
words1=[]
wordtags1=[]
nanjo1=[]
os.chdir('/home/reshad/fakenews/deeplearnig/OA-STM-domains/all-data/Math')
x1=open('test_com8.txt',encoding="utf8")
for i in x1:
    #i=re.sub(r'[^\w\s]','',i)
    a=i.split('\t')
    w1=a[0].rstrip()
    #words1.append(w1)
    if len(a) <= 1:  # deal with the '\n' in the first row
        continue
    else:
        wt = a[1].rstrip()
    words1.append(w1)
    wordtags1.append(wt)
    na=(w1,wt)
    if na !=('', '\n'):
        if na != ('', ''):
            nanjo1.append(na)


math =cuu(nanjo1)
math1=test_data(math)
pred_labels1 = test_result(math1)
true_labels1 = [[w[1] for w in s] for s in math]

print("Precision-score: {:.1%}".format(precision_score(true_labels1, pred_labels1)))
print("Recal-score: {:.1%}".format(recall_score(true_labels1, pred_labels1)))
print("F1-score: {:.1%}".format(f1_score(true_labels1, pred_labels1)))

print('*********************** Med ******************************')
import os
words1=[]
wordtags1=[]
nanjo1=[]
os.chdir('/home/reshad/fakenews/deeplearnig/OA-STM-domains/all-data/Med')
x1=open('test_com9.txt',encoding="utf8")
for i in x1:
    #i=re.sub(r'[^\w\s]','',i)
    a=i.split('\t')
    w1=a[0].rstrip()
    #words1.append(w1)
    if len(a) <= 1:  # deal with the '\n' in the first row
        continue
    else:
        wt = a[1].rstrip()
    words1.append(w1)
    wordtags1.append(wt)
    na=(w1,wt)
    if na !=('', '\n'):
        if na != ('', ''):
            nanjo1.append(na)


med =cuu(nanjo1)
med1=test_data(med)
pred_labels1 = test_result(med1)
true_labels1 = [[w[1] for w in s] for s in med]

print("Precision-score: {:.1%}".format(precision_score(true_labels1, pred_labels1)))
print("Recal-score: {:.1%}".format(recall_score(true_labels1, pred_labels1)))
print("F1-score: {:.1%}".format(f1_score(true_labels1, pred_labels1)))

print('*********************** Ms ******************************')
import os
words1=[]
wordtags1=[]
nanjo1=[]
os.chdir('/home/reshad/fakenews/deeplearnig/OA-STM-domains/all-data/MS')
x1=open('test_com10.txt',encoding="utf8")
for i in x1:
    #i=re.sub(r'[^\w\s]','',i)
    a=i.split('\t')
    w1=a[0].rstrip()
    #words1.append(w1)
    if len(a) <= 1:  # deal with the '\n' in the first row
        continue
    else:
        wt = a[1].rstrip()
    words1.append(w1)
    wordtags1.append(wt)
    na=(w1,wt)
    if na !=('', '\n'):
        if na != ('', ''):
            nanjo1.append(na)


ms =cuu(nanjo1)
ms1=test_data(ms)
pred_labels1 = test_result(ms1)
true_labels1 = [[w[1] for w in s] for s in ms]

print("Precision-score: {:.1%}".format(precision_score(true_labels1, pred_labels1)))
print("Recal-score: {:.1%}".format(recall_score(true_labels1, pred_labels1)))
print("F1-score: {:.1%}".format(f1_score(true_labels1, pred_labels1)))

