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
from keras.layers import Activation, Dense, Flatten
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
nltk.download('stopwords')
from keras.models import Sequential
import tensorflow as tf
import tensorflow_addons as tfa
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers

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

plt.hist(sen_len,bins=21) # Checking the Sequence Length Distribution and getting most occurance  sequence length

max_tokens = 68000
text_vectorizer = TextVectorization(max_tokens=max_tokens,standardize='lower_and_strip_punctuation',output_sequence_length=55)
# Adapt text vectorizer to training sentences
text_vectorizer.adapt(train_sentences)

# viewing vectorize training sentences
import random
target_sentence  = random.choice(train_sentences)
print(f"Text:\n{target_sentence}")
print(f"\nLength of text: {len(target_sentence.split())}")
print(f"\nVectorized text:\n{text_vectorizer([target_sentence])}")

# Getting the vocabulary and showing most frequent and least frequest words in the vocabulary
rct_20k_text_vocab = text_vectorizer.get_vocabulary()
most_common = rct_20k_text_vocab[:5]
least_common = rct_20k_text_vocab[-5:]
print(f"Number of words in vocabulary: {len(rct_20k_text_vocab)}"),
print(f"Most common words in the vocabulary: {most_common}")
print(f"Least common words in the vocabulary: {least_common}")

# Get the config of our text vectorizer
text_vectorizer.get_config()

token_embed = layers.Embedding(input_dim=len(rct_20k_text_vocab),
                               output_dim= 128,
                               mask_zero=True,
                               input_length=55)



print(f"Sentence before Vectorization : \n{target_sentence}\n")
vec_sentence = text_vectorizer([target_sentence])
print(f"Sentence After vectorization :\n {vec_sentence}\n")
embed_sentence = token_embed(vec_sentence)
print(f"Embedding Sentence :\n{embed_sentence}\n")

# Turn our data into TensorFlow Dataset
train_dataset = tf.data.Dataset.from_tensor_slices((train_sentences, train_labels_one_hot))
valid_dataset = tf.data.Dataset.from_tensor_slices((val_sentences, val_labels_one_hot))
test_dataset = tf.data.Dataset.from_tensor_slices((test_sentences, test_labels_one_hot))

# Take the TensorSliceDataset's and turn them into prefetched batches
train_dataset = train_dataset.batch(32).prefetch(tf.data.AUTOTUNE)
valid_dataset = valid_dataset.batch(32).prefetch(tf.data.AUTOTUNE)
test_dataset = test_dataset.batch(32).prefetch(tf.data.AUTOTUNE)

inputs = layers.Input(shape = (1,),dtype = tf.string)
text_vector = text_vectorizer(inputs)
embed = token_embed(text_vector)
x = layers.Conv1D(filters = 64, kernel_size= 5, padding="same",activation="relu",kernel_regularizer=tf.keras.regularizers.L2(0.01))(embed)
x = layers.GlobalMaxPool1D()(x)
x = layers.Dropout(0.2)(x)
outputs = layers.Dense(num_classes,activation="softmax")(x)

model = tf.keras.Model(inputs,outputs)
model.summary()

model.compile(optimizer='Adam', loss="categorical_crossentropy", metrics=["accuracy"])
model_1_history =  model.fit(train_dataset,
                             steps_per_epoch=int(0.1 * len(train_dataset)),
                             epochs = 10,
                             validation_data = valid_dataset,
                            validation_steps=int(0.1 * len(valid_dataset)),)

'''
claim_model=Sequential()
for layer in model.layers[:-1]:
    claim_model.add(layer)

for layer in claim_model.layers:
    layer.trainable = False

claim_model.add(layers.Dense(units=2, activation='softmax'))
'''
from keras.models import Model
'''
model.layers.pop()
model.layers.pop()
for layer in model.layers:
    layer.trainable = False
'''
def create_encoder():
    outputs=model.layers[2].output
    models = keras.Model(inputs=model.input, outputs=outputs, name="claim-encoder")
    return models

encoder = create_encoder()

def create_classifier(encoder, model,trainable=True):
    claim_model = Sequential()
    for layer in encoder.layers:
        layer.trainable = trainable
    #inputs = keras.Input(shape = (1,),dtype = tf.string)
    #embd = encoder()
    claim_model.add(encoder)
    claim_model.add(model.layers[3])
    claim_model.add(model.layers[4])
    claim_model.add(model.layers[5])
    claim_model.add(model.layers[6])
    claim_model.add(layers.Dense(units=2, activation='softmax'))
    #model = tf.keras.Model(inputs, outputs)
    claim_model.compile(optimizer='Adam', loss="categorical_crossentropy", metrics=["accuracy"])
    return claim_model

claim_model= create_classifier(encoder, model)
claim_model.summary()

learning_rate = 0.001
batch_size = 265
hidden_units = 512
projection_units = 128
num_epochs = 50
dropout_rate = 0.5
temperature = 0.05

class SupervisedContrastiveLoss(keras.losses.Loss):
    def __init__(self, temperature=1, name=None):
        super(SupervisedContrastiveLoss, self).__init__(name=name)
        self.temperature = temperature

    def __call__(self, labels, feature_vectors, sample_weight=None):
        # Normalize feature vectors
        feature_vectors_normalized = tf.math.l2_normalize(feature_vectors, axis=1)
        # Compute logits
        logits = tf.divide(
            tf.matmul(
                feature_vectors_normalized, tf.transpose(feature_vectors_normalized)
            ),
            self.temperature,
        )
        return tfa.losses.npairs_loss(tf.squeeze(labels), logits)

def add_projection_head(encoder,trainable=True):
    inputs = keras.Input(shape = (1,),dtype = tf.string)
    for layer in encoder.layers:
        layer.trainable = trainable
    features = encoder(inputs)
    features = Flatten()(features)
    outputs = layers.Dense(projection_units, activation="relu")(features)
    model = keras.Model(
        inputs=inputs, outputs=outputs, name="cifar-encoder_with_projection-head"
    )
    return model

encoder = create_encoder()

encoder_with_projection_head = add_projection_head(encoder,trainable=True)
encoder_with_projection_head.compile(
    optimizer=keras.optimizers.Adam(learning_rate),
    loss=SupervisedContrastiveLoss(temperature),
)

encoder_with_projection_head.summary()

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
            labels.append(int(claim_labels[i][j]))
            ID.append(paper_id[i])
            list_df.append([paper_id[i],claim_labels[i][j],claim_sentences[i][j]])
    df = pd.DataFrame(list_df, columns=['Paper Ids', 'Label','Sentences'])
    return df

claim_train_sentences_df = sentences_label(claim_train_data)
claim_train_sentences = claim_train_sentences_df["Sentences"].tolist()
one_hot_encoder = OneHotEncoder(sparse=False)
claim_train_labels_one_hot = one_hot_encoder.fit_transform(claim_train_sentences_df["Label"].to_numpy().reshape(-1, 1))
claim_label_encoder = LabelEncoder()
claim_train_labels_encoded = claim_label_encoder.fit_transform(claim_train_sentences_df["Label"].to_numpy())
# Get class names and number of classes from LabelEncoder instance
num_classes = len(claim_label_encoder.classes_)
class_names = claim_label_encoder.classes_
print(num_classes, class_names)

sen_len = [len(sentences.split()) for sentences in claim_train_sentences]
avg_sen_len = np.mean(sen_len)
print(avg_sen_len)

def count_words(claim_train_sentences):
    total=[]
    total_unique=[]
    for i in range(len(claim_train_sentences)):
        text_tokens=word_tokenize(claim_train_sentences[i])
        #tokens_without_sw = [word for word in text_tokens if not word in stopwords.words()]
        total.append(text_tokens)
    for i in range(len(total)):
        for j in range(len(total[i])):
            total_unique.append(total[i][j])
    words = list(set((total_unique)))
    n_words = len(words)
    return n_words

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





# Get class names and number of classes from LabelEncoder instance
num_classes = len(claim_label_encoder.classes_)
class_names = claim_label_encoder.classes_
print(num_classes, class_names)

sen_len = [len(sentences.split()) for sentences in claim_train_sentences]
avg_sen_len = np.mean(sen_len)
print(avg_sen_len)

def count_words(claim_train_sentences):
    total=[]
    total_unique=[]
    for i in range(len(claim_train_sentences)):
        text_tokens=word_tokenize(claim_train_sentences[i])
        #tokens_without_sw = [word for word in text_tokens if not word in stopwords.words()]
        total.append(text_tokens)
    for i in range(len(total)):
        for j in range(len(total[i])):
            total_unique.append(total[i][j])
    words = list(set((total_unique)))
    n_words = len(words)
    return n_words

claim_max_token= count_words(claim_train_sentences)

claim_text_vectorizer = TextVectorization(max_tokens=claim_max_token,standardize='lower_and_strip_punctuation',output_sequence_length=55)
# Adapt text vectorizer to training sentences
claim_text_vectorizer.adapt(claim_train_sentences)

# viewing vectorize training sentences
target_sentence  = random.choice(claim_train_sentences)
print(f"Text:\n{target_sentence}")
print(f"\nLength of text: {len(target_sentence.split())}")
print(f"\nVectorized text:\n{claim_text_vectorizer([target_sentence])}")


# Getting the vocabulary and showing most frequent and least frequest words in the vocabulary
claim_text_vocab = claim_text_vectorizer.get_vocabulary()
most_common = claim_text_vocab[:5]
least_common = claim_text_vocab[-5:]
print(f"Number of words in vocabulary: {len(claim_text_vocab)}"),
print(f"Most common words in the vocabulary: {most_common}")
print(f"Least common words in the vocabulary: {least_common}")

# Get the config of our text vectorizer
claim_text_vectorizer.get_config()

token_embed = layers.Embedding(input_dim=len(claim_text_vocab),
                               output_dim= 128,
                               mask_zero=True,
                               input_length=55)



print(f"Sentence before Vectorization : \n{target_sentence}\n")
vec_sentence = claim_text_vectorizer([target_sentence])
print(f"Sentence After vectorization :\n {vec_sentence}\n")
embed_sentence = token_embed(vec_sentence)
print(f"Embedding Sentence :\n{embed_sentence}\n")

# Turn our data into TensorFlow Dataset
train_dataset = tf.data.Dataset.from_tensor_slices((claim_train_sentences, claim_train_labels_one_hot))
valid_dataset = tf.data.Dataset.from_tensor_slices((claim_val_sentences, claim_val_labels_one_hot))
test_dataset = tf.data.Dataset.from_tensor_slices((claim_test_sentences, claim_test_labels_one_hot))

# Take the TensorSliceDataset's and turn them into prefetched batches
train_dataset = train_dataset.batch(32).prefetch(tf.data.AUTOTUNE)
valid_dataset = valid_dataset.batch(32).prefetch(tf.data.AUTOTUNE)
test_dataset = test_dataset.batch(32).prefetch(tf.data.AUTOTUNE)


claim_train_t = claim_train_sentences_df["Label"].to_numpy()
claim_train_t1=[]
for i in claim_train_t:
    claim_train_t1.append(int(i))
train_dataset1 = tf.data.Dataset.from_tensor_slices((claim_train_sentences, claim_train_t1))
train_dataset2 = train_dataset1.batch(32).prefetch(tf.data.AUTOTUNE)

history = encoder_with_projection_head.fit(
    train_dataset2, batch_size=batch_size, epochs=num_epochs
)

claim_model= create_classifier(encoder, model,trainable=False)

model_claim_history =  claim_model.fit(train_dataset,
                             steps_per_epoch=int(0.1 * len(train_dataset)),
                             epochs = 15,
                             validation_data = valid_dataset,
                             validation_steps=int(0.1 * len(valid_dataset)),)

# Evaluate on whole validation dataset (we only validated on 10% of batches during training)
claim_model.evaluate(valid_dataset)

model_1_pred_probs = claim_model.predict(valid_dataset)
# Convert pred probs to classes
model_1_preds = tf.argmax(model_1_pred_probs, axis=1)

# Calculate model_1 results
model_1_results = calculate_results(y_true=claim_val_labels_encoded,
                                    y_pred=model_1_preds)
print(model_1_results)

# Make predictions (our model outputs prediction probabilities for each class)
model_1_pred_probs = claim_model.predict(test_dataset)
# Convert pred probs to classes
model_1_preds = tf.argmax(model_1_pred_probs, axis=1)

# Calculate model_1 results
model_1_results = calculate_results(y_true=claim_test_labels_encoded,
                                    y_pred=model_1_preds)
print(model_1_results)
