import os
import keras
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
from keras.layers import Activation, Dense, Flatten
# Download pretrained TensorFlow Hub USE
import tensorflow_hub as hub
import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow import keras
from tensorflow.keras import layers
tf_hub_embedding_layer = hub.KerasLayer("https://tfhub.dev/google/universal-sentence-encoder/4",
                                        trainable=False,
                                        name="universal_sentence_encoder")
# One hot encode labels
from sklearn.preprocessing import OneHotEncoder
one_hot_encoder = OneHotEncoder(sparse=False)
# Extract labels ("target" columns) and encode them into integers
from sklearn.preprocessing import LabelEncoder
label_encoder = LabelEncoder()



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

# Split sequence-level data splits into character-level data splits
train_chars = [split_chars(sentence) for sentence in claim_train_sentences]
val_chars = [split_chars(sentence) for sentence in claim_val_sentences]
test_chars = [split_chars(sentence) for sentence in claim_test_sentences]

# Check the Avrage Char Length in the training Sentences
char_lens = [len(sentence) for sentence in claim_train_sentences]
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


def create_encoder():
    # Token_level Model (using Pretrained -- Universal Sentence Encoder)
    token_inputs = layers.Input(shape=[], dtype=tf.string, name="token_input")
    token_embedding = tf_hub_embedding_layer(token_inputs)
    token_dense = layers.Dense(128, activation="relu")(token_embedding)
    token_model = tf.keras.Model(inputs=token_inputs,
                                 outputs=token_dense)
    # char_level Model
    char_inputs = layers.Input(shape=(1,), dtype=tf.string, name="char_input")
    char_vectors = char_vectorizer(char_inputs)
    char_embedding = char_embed(char_vectors)
    char_bi_lstm = layers.Bidirectional(layers.LSTM(25, activation="relu"))(char_embedding)
    char_model = tf.keras.Model(inputs=char_inputs,  # char_dense = layers.Dense(128,activation="relu")(char_bilstm)
                                outputs=char_bi_lstm)

    # Now Concatenate token_model and char_model
    concat_layer = layers.Concatenate(name="token_char_hybrid")([token_model.output,char_model.output])
    model = keras.Model(inputs=[token_inputs,char_inputs], outputs=concat_layer, name="claim-encoder")
    return model


def create_classifier(encoder, trainable=True):

    for layer in encoder.layers:
        layer.trainable = trainable

    #inputs = layers.Input(shape=[], dtype=tf.string)
    embd = encoder(encoder.inputs)
    concat_dropout = layers.Dropout(0.5)(embd)
    concat_dense = layers.Dense(256, activation="relu")(concat_dropout)
    final_dropout = layers.Dropout(0.2)(concat_dense)
    outputs = layers.Dense(2, activation="softmax")(final_dropout)
    model = tf.keras.Model(inputs=encoder.inputs,outputs=outputs)
    # Compile the model
    model.compile(loss="categorical_crossentropy",
                    optimizer=tf.keras.optimizers.Adam(),
                    metrics=["accuracy"])
    return model



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
    #inputs = keras.Input(shape = (1,),dtype = tf.string)
    for layer in encoder.layers:
        layer.trainable = trainable
    features = encoder(encoder.inputs)
    features = Flatten()(features)
    outputs = layers.Dense(projection_units, activation="relu")(features)
    model = keras.Model(
        inputs=encoder.inputs, outputs=outputs, name="cifar-encoder_with_projection-head"
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
    claim_train_t1.append([int(i)])

#labels = tf.data.Dataset.from_tensor_slices(claim_train_t1) # make labels
train_data_pr=tf.data.Dataset.from_tensor_slices((claim_train_sentences, train_chars))
labels = tf.data.Dataset.from_tensor_slices(claim_train_t1) # make labels
train_data_pr=tf.data.Dataset.zip((train_data_pr, labels))
train_dataset2 = train_data_pr.batch(32).prefetch(tf.data.AUTOTUNE)

history = encoder_with_projection_head.fit(
    train_dataset2, batch_size=batch_size, epochs=num_epochs
)


classifier = create_classifier(encoder, trainable=False)

# Fit the model on tokens and chars
model_claim_history = classifier.fit(train_char_token_dataset, # train on dataset of token and characters
                              steps_per_epoch=int(0.1 * len(train_char_token_dataset)),
                              epochs=10,
                              validation_data=val_char_token_dataset,
                              validation_steps=int(0.1 * len(val_char_token_dataset)))


# Evaluate on the whole validation dataset
classifier.evaluate(val_char_token_dataset)
# Make predictions using the token-character model hybrid
claim_model_pred_probs = classifier.predict(val_char_token_dataset)
# Turn prediction probabilities into prediction classes
claim_model_preds = tf.argmax(claim_model_pred_probs, axis=1)
# Get results of token-char-hybrid model
claim_model_results = calculate_results(y_true=claim_val_labels_encoded,y_pred=claim_model_preds)
print(claim_model_results)
# Evaluate on the whole test dataset
classifier.evaluate(test_char_token_dataset)
# Make predictions using the token-character model hybrid
claim_model_pred_probs = classifier.predict(test_char_token_dataset)
# Turn prediction probabilities into prediction classes
claim_model_preds = tf.argmax(claim_model_pred_probs, axis=1)
# Get results of token-char-hybrid model
claim_model_results = calculate_results(y_true=claim_test_labels_encoded,y_pred=claim_model_preds)
print(claim_model_results)
