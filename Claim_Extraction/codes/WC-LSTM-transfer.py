import os
import torch
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report
from helper_functions import calculate_results
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
import matplotlib.pyplot as plt
from tensorflow.keras.layers.experimental.preprocessing import TextVectorization
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
nltk.download('stopwords')
from keras.models import Model
from keras.models import Sequential
import random
# Download pretrained TensorFlow Hub USE
import tensorflow_hub as hub
tf_hub_embedding_layer = hub.KerasLayer("https://tfhub.dev/google/universal-sentence-encoder/4",
                                        trainable=False,
                                        name="universal_sentence_encoder")

data_dir = r"D:\NLP\FSND\claim-extraction\data\disclose\pubmed-rct-master\pubmed-rct-master\PubMed_20k_RCT_numbers_replaced_with_at_sign/"
filenames = [data_dir +filename for filename in os.listdir(data_dir)]

def get_lines(filename):
  """
  """
  with open(filename, "r") as f:
    return f.readlines()

train_lines = get_lines(filenames[2])
train_lines[:10]


def preprocess_text_with_line_numbers(filename):
    input_lines = get_lines(filename)  # get all lines from filename
    abstract_lines = ""  # create an empty abstract
    abstract_samples = []  # create an empty list of abstracts

    for line in input_lines:
        if line.startswith("###"):  # check to see if line is an ID line
            abstract_id = line
            abstract_lines = ""  # reset the abstract string
        elif line.isspace():
            abstract_line_split = abstract_lines.splitlines()  # split the abstract into separate lines

            for abstract_line_number, abstract_line in enumerate(abstract_line_split):
                line_data = {}
                target_text_split = abstract_line.split("\t")
                line_data["target"] = target_text_split[0]
                line_data["text"] = target_text_split[1].lower()
                line_data["line_number"] = abstract_line_number
                line_data["total_lines"] = len(abstract_line_split) - 1
                abstract_samples.append(line_data)

        else:
            abstract_lines += line

    return abstract_samples

# Get data from file and preprocess it
train_samples = preprocess_text_with_line_numbers(data_dir + "train.txt")
val_samples = preprocess_text_with_line_numbers(data_dir + "dev.txt") # dev is another name for validation set
test_samples = preprocess_text_with_line_numbers(data_dir + "test.txt")
print(len(train_samples), len(val_samples), len(test_samples))

train_df = pd.DataFrame(train_samples)
val_df = pd.DataFrame(val_samples)
test_df = pd.DataFrame(test_samples)
train_df.head(14)

# Distribution of labels in training dat
print(train_df["target"].value_counts())

train_df["target"].value_counts().plot(kind = 'bar')

# Convert abstract text lines into lists
train_sentences = train_df["text"].tolist()
val_sentences = val_df["text"].tolist()
test_sentences = test_df["text"].tolist()
print(len(train_sentences), len(val_sentences), len(test_sentences))

# One hot encode labels
from sklearn.preprocessing import OneHotEncoder
one_hot_encoder = OneHotEncoder(sparse=False)
train_labels_one_hot = one_hot_encoder.fit_transform(train_df["target"].to_numpy().reshape(-1, 1))
val_labels_one_hot = one_hot_encoder.transform(val_df["target"].to_numpy().reshape(-1, 1))
test_labels_one_hot = one_hot_encoder.transform(test_df["target"].to_numpy().reshape(-1, 1))

# Extract labels ("target" columns) and encode them into integers
from sklearn.preprocessing import LabelEncoder
label_encoder = LabelEncoder()
train_labels_encoded = label_encoder.fit_transform(train_df["target"].to_numpy())
val_labels_encoded = label_encoder.transform(val_df["target"].to_numpy())
test_labels_encoded = label_encoder.transform(test_df["target"].to_numpy())

# Get class names and number of classes from LabelEncoder instance
num_classes = len(label_encoder.classes_)
class_names = label_encoder.classes_
print(num_classes, class_names)


sen_len = [len(sentences.split()) for sentences in train_sentences]
avg_sen_len = np.mean(sen_len)
print(avg_sen_len)

# Test out the embedding on a random sentence
random_training_sentence = random.choice(train_sentences)
print(f"Random training sentence:\n{random_training_sentence}\n")
use_embedded_sentence = tf_hub_embedding_layer([random_training_sentence])
print(f"Sentence after embedding:\n{use_embedded_sentence[0][:30]} (truncated output)...\n")
print(f"Length of sentence embedding:\n{len(use_embedded_sentence[0])}")

def split_chars(text):
  return " ".join(list(text))

# Split sequence-level data splits into character-level data splits
train_chars = [split_chars(sentence) for sentence in train_sentences]
val_chars = [split_chars(sentence) for sentence in val_sentences]
test_chars = [split_chars(sentence) for sentence in test_sentences]

# Check the Avrage Char Length in the training Sentences
char_lens = [len(sentence) for sentence in train_sentences]
avg_char_lens = sum(char_lens)/len(char_lens)
print(avg_char_lens)
output_seq_char_len = int(np.percentile(char_lens, 95))

# Get all keyboard characters for char-level embedding
import string
alphabet = string.ascii_lowercase + string.digits + string.punctuation

# Create char-level token vectorizer instance
NUM_CHAR_TOKENS = len(alphabet) + 2 # num characters in alphabet + space + OOV token
char_vectorizer = TextVectorization(max_tokens=NUM_CHAR_TOKENS,
                                    output_sequence_length=output_seq_char_len,
                                    standardize="lower_and_strip_punctuation",
                                    name="char_vectorizer")

# Adapt character vectorizer to training characters
char_vectorizer.adapt(train_chars)

# Check character vocabulary characteristics
char_vocab = char_vectorizer.get_vocabulary()
print(f"Number of different characters in character vocab: {len(char_vocab)}")
print(f"5 most common characters: {char_vocab[:5]}")
print(f"5 least common characters: {char_vocab[-5:]}")

# Test out character vectorizer
random_train_chars = random.choice(train_chars)
print(f"Charified text:\n{random_train_chars}")
print(f"\nLength of chars: {len(random_train_chars.split())}")
vectorized_chars = char_vectorizer([random_train_chars])
print(f"\nVectorized chars:\n{vectorized_chars}")
print(f"\nLength of vectorized chars: {len(vectorized_chars[0])}")

char_embed = layers.Embedding(input_dim=NUM_CHAR_TOKENS,
                              output_dim= 25,
                              mask_zero= True,
                              name= "char_embed")

# Test out character embedding layer
print(f"Charified text (before vectorization and embedding):\n{random_train_chars}\n")
char_embed_example = char_embed(char_vectorizer([random_train_chars]))
print(f"Embedded chars (after vectorization and embedding):\n{char_embed_example}\n")
print(f"Character embedding shape: {char_embed_example.shape}")

# Token_level Model (using Pretrained -- Universal Sentence Encoder)
token_inputs = layers.Input(shape = [], dtype= tf.string, name = "token_input")
token_embedding = tf_hub_embedding_layer(token_inputs)
token_dense = layers.Dense(128,activation="relu")(token_embedding)
token_model = tf.keras.Model(inputs = token_inputs,
                             outputs = token_dense)
# char_level Model
char_inputs = layers.Input(shape=(1,), dtype= tf.string, name="char_input")
char_vectors = char_vectorizer(char_inputs)
char_embedding = char_embed(char_vectors)
char_bi_lstm = layers.Bidirectional(layers.LSTM(25,activation="relu"))(char_embedding)
char_model = tf.keras.Model(inputs= char_inputs,    # char_dense = layers.Dense(128,activation="relu")(char_bilstm)
                            outputs =char_bi_lstm)


# Now Concatenate token_model and char_model
concat_layer = layers.Concatenate(name = "token_char_hybrid")([token_model.output,
                                                               char_model.output])

# Add Some Layer on top of concat_layer
concat_dropout = layers.Dropout(0.5)(concat_layer)
concat_dense = layers.Dense(256,activation="relu")(concat_dropout)
final_dropout = layers.Dropout(0.2)(concat_dense)
output_layer = layers.Dense(num_classes,activation="softmax")(final_dropout)

model_4 = tf.keras.Model(inputs = [token_model.input, char_model.input],
                         outputs = output_layer,
                         name="model_4_token_and_char_embeddings")

model_4.summary()

# Compile token char model
model_4.compile(loss="categorical_crossentropy",
                optimizer=tf.keras.optimizers.Adam(), # section 4.2 of https://arxiv.org/pdf/1612.05251.pdf mentions using SGD but we'll stick with Adam
                metrics=["accuracy"])

# Combine chars and tokens into a dataset
train_char_token_data = tf.data.Dataset.from_tensor_slices((train_sentences, train_chars)) # make data
train_char_token_labels = tf.data.Dataset.from_tensor_slices(train_labels_one_hot) # make labels
train_char_token_dataset = tf.data.Dataset.zip((train_char_token_data, train_char_token_labels)) # combine data and labels

# Prefetch and batch train data
train_char_token_dataset = train_char_token_dataset.batch(32).prefetch(tf.data.AUTOTUNE)

# Repeat same steps validation data
val_char_token_data = tf.data.Dataset.from_tensor_slices((val_sentences, val_chars))
val_char_token_labels = tf.data.Dataset.from_tensor_slices(val_labels_one_hot)
val_char_token_dataset = tf.data.Dataset.zip((val_char_token_data, val_char_token_labels))
val_char_token_dataset = val_char_token_dataset.batch(32).prefetch(tf.data.AUTOTUNE)

# Repeat same steps testing data
test_char_token_data = tf.data.Dataset.from_tensor_slices((test_sentences, test_chars))
test_char_token_labels = tf.data.Dataset.from_tensor_slices(test_labels_one_hot)
test_char_token_dataset = tf.data.Dataset.zip((test_char_token_data, test_char_token_labels))
test_char_token_dataset = test_char_token_dataset.batch(32).prefetch(tf.data.AUTOTUNE)

# Fit the model on tokens and chars
model_4_history = model_4.fit(train_char_token_dataset, # train on dataset of token and characters
                              steps_per_epoch=int(0.1 * len(train_char_token_dataset)),
                              epochs=10,
                              validation_data=val_char_token_dataset,
                              validation_steps=int(0.1 * len(val_char_token_dataset)))

# Evaluate on the whole validation dataset
model_4.evaluate(val_char_token_dataset)
# Make predictions using the token-character model hybrid
model_4_pred_probs = model_4.predict(val_char_token_dataset)
# Turn prediction probabilities into prediction classes
model_4_preds = tf.argmax(model_4_pred_probs, axis=1)
# Get results of token-char-hybrid model
model_4_results = calculate_results(y_true=val_labels_encoded,
                                    y_pred=model_4_preds)
print(model_4_results)

# Evaluate on the whole test dataset
model_4.evaluate(test_char_token_dataset)
# Make predictions using the token-character model hybrid
model_4_pred_probs = model_4.predict(test_char_token_dataset)
# Turn prediction probabilities into prediction classes
model_4_preds = tf.argmax(model_4_pred_probs, axis=1)
# Get results of token-char-hybrid model
model_4_results = calculate_results(y_true=test_labels_encoded,
                                    y_pred=model_4_preds)
print(model_4_results)

####################### Fine-Tuning ####################

claim_data_dir = r"D:\NLP\FSND\claim-extraction\data\disclose\detecting-scientific-claim-master\detecting-scientific-claim-master\dataset/"
claim_filenames = [claim_data_dir + filename for filename in os.listdir(claim_data_dir)]

claim_train_data= pd.read_json(claim_filenames[2], lines=True)
claim_test_data= pd.read_json(claim_filenames[1], lines=True)
claim_val_data= pd.read_json(claim_filenames[3], lines=True)

def sentences_label(claim_train_data):
    ID =[]
    sentences=[]
    labels=[]
    list_df=[]
    claim_sentences=claim_train_data['sentences']
    claim_labels= claim_train_data['labels']
    paper_id = claim_train_data['paper_id']
    for i in range(len(claim_sentences)):
        for j in range(len(claim_sentences[i])):
            sentences.append(claim_sentences[i][j])
            labels.append(claim_labels[i][j])
            ID.append(paper_id[i])
            list_df.append([paper_id[i],claim_labels[i][j],claim_sentences[i][j]])
    df = pd.DataFrame(list_df, columns=['Paper Ids', 'Label','Sentences'])
    return df

claim_train_sentences_df = sentences_label(claim_train_data)
claim_train_sentences = claim_train_sentences_df["Sentences"].tolist()
claim_train_labels_one_hot = one_hot_encoder.fit_transform(claim_train_sentences_df["Label"].to_numpy().reshape(-1, 1))
claim_label_encoder = LabelEncoder()
claim_train_labels_encoded = claim_label_encoder.fit_transform(claim_train_sentences_df["Label"].to_numpy())

claim_test_sentences_df = sentences_label(claim_test_data)
claim_test_sentences = claim_test_sentences_df["Sentences"].tolist()
claim_test_labels_one_hot = one_hot_encoder.fit_transform(claim_test_sentences_df["Label"].to_numpy().reshape(-1, 1))
claim_test_labels_encoded = claim_label_encoder.fit_transform(claim_test_sentences_df["Label"].to_numpy())

claim_val_sentences_df = sentences_label(claim_val_data)
claim_val_sentences = claim_val_sentences_df["Sentences"].tolist()
claim_val_labels_one_hot = one_hot_encoder.fit_transform(claim_val_sentences_df["Label"].to_numpy().reshape(-1, 1))
claim_val_labels_encoded = claim_label_encoder.fit_transform(claim_val_sentences_df["Label"].to_numpy())

def split_chars(text):
  return " ".join(list(text))

del train_chars, val_chars, test_chars, random_train_chars
# Split sequence-level data splits into character-level data splits
train_chars = [split_chars(sentence) for sentence in claim_train_sentences]
val_chars = [split_chars(sentence) for sentence in claim_val_sentences]
test_chars = [split_chars(sentence) for sentence in claim_test_sentences]

# Check the Avrage Char Length in the training Sentences
char_lens = [len(sentence) for sentence in claim_train_sentences]
avg_char_lens = sum(char_lens)/len(char_lens)
print(avg_char_lens)
output_seq_char_len = int(np.percentile(char_lens, 95))

'''
# Get all keyboard characters for char-level embedding
import string
alphabet = string.ascii_lowercase + string.digits + string.punctuation

# Create char-level token vectorizer instance
NUM_CHAR_TOKENS = len(alphabet) + 2 # num characters in alphabet + space + OOV token
char_vectorizer = TextVectorization(max_tokens=NUM_CHAR_TOKENS,
                                    output_sequence_length=output_seq_char_len,
                                    standardize="lower_and_strip_punctuation",
                                    name="char_vectorizer")
'''
# Adapt character vectorizer to training characters
char_vectorizer.adapt(train_chars)

# Check character vocabulary characteristics
char_vocab = char_vectorizer.get_vocabulary()
print(f"Number of different characters in character vocab: {len(char_vocab)}")
print(f"5 most common characters: {char_vocab[:5]}")
print(f"5 least common characters: {char_vocab[-5:]}")

# Test out character vectorizer
random_train_chars = random.choice(train_chars)
print(f"Charified text:\n{random_train_chars}")
print(f"\nLength of chars: {len(random_train_chars.split())}")
vectorized_chars = char_vectorizer([random_train_chars])
print(f"\nVectorized chars:\n{vectorized_chars}")
print(f"\nLength of vectorized chars: {len(vectorized_chars[0])}")

'''
char_embed = layers.Embedding(input_dim=NUM_CHAR_TOKENS,
                              output_dim= 25,
                              mask_zero= True,
                              name= "char_embed")
'''
# Test out character embedding layer
print(f"Charified text (before vectorization and embedding):\n{random_train_chars}\n")
char_embed_example = char_embed(char_vectorizer([random_train_chars]))
print(f"Embedded chars (after vectorization and embedding):\n{char_embed_example}\n")
print(f"Character embedding shape: {char_embed_example.shape}")

# Combine chars and tokens into a dataset
train_char_token_data = tf.data.Dataset.from_tensor_slices((claim_train_sentences, train_chars)) # make data
train_char_token_labels = tf.data.Dataset.from_tensor_slices(claim_train_labels_one_hot) # make labels
train_char_token_dataset = tf.data.Dataset.zip((train_char_token_data, train_char_token_labels)) # combine data and labels

# Prefetch and batch train data
train_char_token_dataset = train_char_token_dataset.batch(32).prefetch(tf.data.AUTOTUNE)

# Repeat same steps validation data
val_char_token_data = tf.data.Dataset.from_tensor_slices((claim_val_sentences, val_chars))
val_char_token_labels = tf.data.Dataset.from_tensor_slices(claim_val_labels_one_hot)
val_char_token_dataset = tf.data.Dataset.zip((val_char_token_data, val_char_token_labels))
val_char_token_dataset = val_char_token_dataset.batch(32).prefetch(tf.data.AUTOTUNE)

# Repeat same steps testing data
test_char_token_data = tf.data.Dataset.from_tensor_slices((claim_test_sentences, test_chars))
test_char_token_labels = tf.data.Dataset.from_tensor_slices(claim_test_labels_one_hot)
test_char_token_dataset = tf.data.Dataset.zip((test_char_token_data, test_char_token_labels))
test_char_token_dataset = test_char_token_dataset.batch(32).prefetch(tf.data.AUTOTUNE)

model_4.layers.pop()
model_4.layers.pop()
'''
for layer in model_4.layers:
    layer.trainable = False
'''
# recover the output from the last layer in the model and use as input to new Dense layer
last = model_4.layers[-2].output
x = layers.Dense(2, activation="softmax")(last)
claim_model = Model(model_4.input, x)
claim_model.trainable=True
claim_model.summary()

# Compile token char model
claim_model.compile(loss="categorical_crossentropy",
                optimizer=tf.keras.optimizers.Adam(), # section 4.2 of https://arxiv.org/pdf/1612.05251.pdf mentions using SGD but we'll stick with Adam
                metrics=["accuracy"])

# Fit the model on tokens and chars
model_claim_history = claim_model.fit(train_char_token_dataset, # train on dataset of token and characters
                              steps_per_epoch=int(0.1 * len(train_char_token_dataset)),
                              epochs=10,
                              validation_data=val_char_token_dataset,
                              validation_steps=int(0.1 * len(val_char_token_dataset)))

# Evaluate on the whole validation dataset
claim_model.evaluate(val_char_token_dataset)
# Make predictions using the token-character model hybrid
claim_model_pred_probs = claim_model.predict(val_char_token_dataset)
# Turn prediction probabilities into prediction classes
claim_model_preds = tf.argmax(claim_model_pred_probs, axis=1)
# Get results of token-char-hybrid model
claim_model_results = calculate_results(y_true=claim_val_labels_encoded,y_pred=claim_model_preds)
print(claim_model_results)

# Evaluate on the whole test dataset
claim_model.evaluate(test_char_token_dataset)
# Make predictions using the token-character model hybrid
claim_model_pred_probs = claim_model.predict(test_char_token_dataset)
# Turn prediction probabilities into prediction classes
claim_model_preds = tf.argmax(claim_model_pred_probs, axis=1)
# Get results of token-char-hybrid model
claim_model_results = calculate_results(y_true=claim_test_labels_encoded,y_pred=claim_model_preds)
print(claim_model_results)



