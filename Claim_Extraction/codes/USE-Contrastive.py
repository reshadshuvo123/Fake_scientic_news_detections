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
from keras.models import Sequential
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
import random
# Download pretrained TensorFlow Hub USE
import tensorflow_hub as hub
import tensorflow as tf
import tensorflow_addons as tfa
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
from keras.layers import Activation, Dense, Flatten
tf_hub_embedding_layer = hub.KerasLayer("https://tfhub.dev/google/universal-sentence-encoder/4",
                                        trainable=False,
                                        name="universal_sentence_encoder")

label_encoder = LabelEncoder()
one_hot_encoder = OneHotEncoder(sparse=False)

claim_data_dir = r"D:\NLP\FSND\claim-extraction\data\disclose\detecting-scientific-claim-master\detecting-scientific-claim-master\dataset/"
claim_filenames = [claim_data_dir + filename for filename in os.listdir(claim_data_dir)]

claim_train_data= pd.read_json(claim_filenames[2], lines=True)
claim_test_data= pd.read_json(claim_filenames[1], lines=True)
claim_val_data= pd.read_json(claim_filenames[3], lines=True)

learning_rate = 0.001
batch_size = 265
hidden_units = 512
projection_units = 128
num_epochs = 50
dropout_rate = 0.5
temperature = 0.05

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


# Test out the embedding on a random sentence
random_training_sentence = random.choice(claim_train_sentences)
print(f"Random training sentence:\n{random_training_sentence}\n")
use_embedded_sentence = tf_hub_embedding_layer([random_training_sentence])
print(f"Sentence after embedding:\n{use_embedded_sentence[0][:30]} (truncated output)...\n")
print(f"Length of sentence embedding:\n{len(use_embedded_sentence[0])}")

# Turn our data into TensorFlow Dataset
train_dataset = tf.data.Dataset.from_tensor_slices((claim_train_sentences, claim_train_labels_one_hot))
valid_dataset = tf.data.Dataset.from_tensor_slices((claim_val_sentences, claim_val_labels_one_hot))
test_dataset = tf.data.Dataset.from_tensor_slices((claim_test_sentences, claim_test_labels_one_hot))

# Take the TensorSliceDataset's and turn them into prefetched batches
train_dataset = train_dataset.batch(32).prefetch(tf.data.AUTOTUNE)
valid_dataset = valid_dataset.batch(32).prefetch(tf.data.AUTOTUNE)
test_dataset = test_dataset.batch(32).prefetch(tf.data.AUTOTUNE)

learning_rate = 0.001
batch_size = 265
hidden_units = 512
projection_units = 128
num_epochs = 50
dropout_rate = 0.5
temperature = 0.05

def create_encoder():
    inputs = layers.Input(shape=[], dtype=tf.string)
    outputs = tf_hub_embedding_layer(inputs)
    #outputs = token_embed(text_vector)
    model = keras.Model(inputs=inputs, outputs=outputs, name="claim-encoder")
    return model

encoder = create_encoder()
def create_classifier(encoder, trainable=True):

    for layer in encoder.layers:
        layer.trainable = trainable

    inputs = layers.Input(shape=[], dtype=tf.string)
    embd = encoder(inputs)
    x = layers.Dense(128, activation="relu")(embd)  # add a fully connected layer on top of the embedding
    outputs = layers.Dense(2, activation="softmax", kernel_regularizer=None)(x)  # create the output layer
    model = tf.keras.Model(inputs=inputs,
                             outputs=outputs)
    # Compile the model
    model.compile(loss="categorical_crossentropy",
                    optimizer=tf.keras.optimizers.Adam(),
                    metrics=["accuracy"])
    return model

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

encoder_with_projection_head = add_projection_head(encoder)
encoder_with_projection_head.compile(
    optimizer=keras.optimizers.Adam(learning_rate),
    loss=SupervisedContrastiveLoss(temperature),
)

encoder_with_projection_head.summary()

claim_train_t = claim_train_sentences_df["Label"].to_numpy()
claim_train_t1=[]
for i in claim_train_t:
    claim_train_t1.append(int(i))
train_dataset1 = tf.data.Dataset.from_tensor_slices((claim_train_sentences, claim_train_t1))
train_dataset2 = train_dataset1.batch(32).prefetch(tf.data.AUTOTUNE)

history = encoder_with_projection_head.fit(
    train_dataset2, batch_size=batch_size, epochs=num_epochs
)


classifier = create_classifier(encoder, trainable=False)

history2 = classifier.fit(train_dataset,
                             steps_per_epoch=int(0.1 * len(train_dataset)),
                             epochs = 15,
                             validation_data = valid_dataset,
                             validation_steps=int(0.1 * len(valid_dataset)),)
# Evaluate on whole validation dataset (we only validated on 10% of batches during training)
classifier.evaluate(valid_dataset)

model_1_pred_probs = classifier.predict(valid_dataset)
# Convert pred probs to classes
model_1_preds = tf.argmax(model_1_pred_probs, axis=1)

# Calculate model_1 results
model_1_results = calculate_results(y_true=claim_val_labels_encoded,
                                    y_pred=model_1_preds)
print(model_1_results)
# Make predictions (our model outputs prediction probabilities for each class)
model_1_pred_probs = classifier.predict(test_dataset)
# Convert pred probs to classes
model_1_preds = tf.argmax(model_1_pred_probs, axis=1)

# Calculate model_1 results
model_1_results = calculate_results(y_true=claim_test_labels_encoded,
                                    y_pred=model_1_preds)
print(model_1_results)
