

!python -m spacy download en_core_web_md

"""# Pre-processing"""

import os
import logging
import pandas as pd
import numpy as np
import tensorflow as tf
import spacy
import re
import nltk
import string
import unicodedata
from sklearn.model_selection import train_test_split
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
import emoji
from bs4 import BeautifulSoup
from tensorflow.keras.layers import Embedding, Dense, Dropout, Input, Conv1D, GlobalMaxPooling1D, Flatten
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import concurrent.futures
# Check GPU availability and set memory growth
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"Using GPU: {gpus}")
    except RuntimeError as e:
        print(e)

"""# Pre-processing"""

logging.basicConfig(level=logging.WARNING, format='%(asctime)s - %(levelname)s - %(message)s')

# Disable XLA optimization
os.environ['XLA_FLAGS'] = '--xla_cpu_multi_thread_eigen=false'

# Load SpaCy English model
nlp = spacy.load('en_core_web_md', disable=['parser', 'ner'])

# Download NLTK resources
nltk.download('stopwords', quiet=True)
nltk.download('punkt', quiet=True)

# Initialize stopwords and stemmer
stop_words = set(stopwords.words('english'))
stemmer = SnowballStemmer("english")

# Contraction mapping
CONTRACTION_MAPPING = {
    "won't": "will not", "can't": "cannot", "don't": "do not", "didn't": "did not",
    "isn't": "is not", "aren't": "are not", "wasn't": "was not", "weren't": "were not",
    "hasn't": "has not", "haven't": "have not", "hadn't": "had not", "he's": "he is",
    "she's": "she is", "it's": "it is", "that's": "that is"
}

# Text cleaning functions
def normalize_elongations(text):
    return re.sub(r'(.)\1{2,}', r'\1\1', text)

def convert_to_ascii(input_str):
    nfkd_form = unicodedata.normalize('NFKD', input_str)
    return ''.join([c for c in nfkd_form if unicodedata.category(c) != 'Mn']).lower()

def clean_contractions(text, mapping):
    for word, replacement in mapping.items():
        text = text.replace(word, replacement)
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub(r'http\S+|www\S+|@\S+|[^a-zA-Z\'\-\s]', '', text)  # keep apostrophes and dashes
    return text

def clean_text(text):
    if text is None:
        return ""
    try:
        text = text.encode('latin1').decode('utf-8')
    except (UnicodeEncodeError, UnicodeDecodeError):
        logging.warning(f"Unicode error during decoding of: {text}")
        return ""
    text = emoji.demojize(text)
    text = BeautifulSoup(text, 'lxml').get_text()
    text = re.sub('<.*?>+', '', text)
    text = re.sub('\n', '', text)
    text = re.sub('\w*\d\w*', '', text)
    text = convert_to_ascii(text)
    text = normalize_elongations(text)
    text = clean_contractions(text, CONTRACTION_MAPPING)
    text = re.sub(r'[" "]+', " ", text)
    doc = nlp(text)
    text = [stemmer.stem(token.text) for token in doc if token.text not in stop_words and not token.is_punct]
    return ' '.join(text).strip()

# Parallel text cleaning
def parallel_clean_text(X_data, num_workers=os.cpu_count()):
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
        return list(executor.map(clean_text, X_data))

# Load and preprocess dataset
train_data = pd.read_csv('/kaggle/input/neural-network-dataset/Forum Discussion Categorization/train.csv')
X = train_data['Discussion'].fillna('').astype(str)
y = train_data['Category']

X_train_cleaned = parallel_clean_text(X)

# Filter out empty texts
non_empty_mask = [bool(text.strip()) for text in X_train_cleaned]
filtered_texts = [text for text, keep in zip(X_train_cleaned, non_empty_mask) if keep]
filtered_labels = y[non_empty_mask]

X_train, X_val, y_train, y_val = train_test_split(filtered_texts, filtered_labels, test_size=0.2, random_state=42)

# Encode labels
label_mapping = {'Politics': 0, 'Sports': 1, 'Media': 2, 'Market & Economy': 3, 'STEM': 4}
y_train_encoded = y_train.map(label_mapping)
y_val_encoded = y_val.map(label_mapping)

# Convert labels to TensorFlow tensors
y_train_encoded_tensor = tf.convert_to_tensor(y_train_encoded, dtype=tf.int32)
y_val_encoded_tensor = tf.convert_to_tensor(y_val_encoded, dtype=tf.int32)



# Tokenizer setup
tokenizer = nltk.word_tokenize

# Build vocabulary from training data
vocab = {}
for text in X_train:
    for word in tokenizer(text):
        if word not in vocab:
            vocab[word] = len(vocab) + 1  # Reserve 0 for padding

# Convert text to sequences
def text_to_sequence(text):
    return [vocab.get(word, 0) for word in tokenizer(text)]

X_train_seq = [text_to_sequence(text) for text in X_train]
X_val_seq = [text_to_sequence(text) for text in X_val]

# Pad sequences
max_len = 100
X_train_padded = pad_sequences(X_train_seq, maxlen=max_len, padding='post', truncating='post')
X_val_padded = pad_sequences(X_val_seq, maxlen=max_len, padding='post', truncating='post')

# One-hot encode labels
y_train_encoded_categorical = to_categorical(y_train_encoded, num_classes=len(label_mapping))
y_val_encoded_categorical = to_categorical(y_val_encoded, num_classes=len(label_mapping))

"""# DNN"""

# Define DNN model
def build_dnn_model(vocab_size, embedding_dim, max_len, label_count):
    input_layer = Input(shape=(max_len,))
    embedding_layer = Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=max_len)(input_layer)
    dropout_layer = Dropout(0.5)(embedding_layer)
    pooled_output = GlobalMaxPooling1D()(dropout_layer)

    dense_layer_1 = Dense(128, activation='relu', kernel_regularizer=l2(0.0001))(pooled_output)
    dropout_layer_2 = Dropout(0.5)(dense_layer_1)
    dense_layer_2 = Dense(64, activation='relu', kernel_regularizer=l2(0.0001))(dropout_layer_2)
    dropout_layer_3 = Dropout(0.5)(dense_layer_2)
    dense_layer_3 = Dense(32, activation='relu', kernel_regularizer=l2(0.0001))(dropout_layer_3)
    dropout_layer_4 = Dropout(0.5)(dense_layer_3)

    output_layer = Dense(label_count, activation='softmax')(dropout_layer_4)

    model = Model(inputs=input_layer, outputs=output_layer)
    model.compile(optimizer=Adam(learning_rate=2e-4), loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# Build and train DNN model
vocab_size = len(vocab) + 1  # Account for padding index
embedding_dim = 100
# model = build_dnn_model(vocab_size, embedding_dim, max_len, len(label_mapping))

# #Train model
# model.fit(X_train_padded, y_train_encoded_categorical, validation_data=(X_val_padded, y_val_encoded_categorical), epochs=1, batch_size=32)

# #Evaluate model
# val_loss, val_accuracy = model.evaluate(X_val_padded, y_val_encoded_categorical)
# print(f"Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}")

"""# TF-IDF Vectorization

"""

import joblib
vectorizer = TfidfVectorizer(max_features=20000)
X_train_tfidf = vectorizer.fit_transform(X_train).toarray()
X_val_tfidf = vectorizer.transform(X_val).toarray()

joblib.dump(vectorizer, '/kaggle/working/tfidf_vectorizer.pkl')

# Convert to TensorFlow tensors
X_train_tensor = tf.convert_to_tensor(X_train_tfidf, dtype=tf.float32)
y_train_tensor = tf.convert_to_tensor(y_train_encoded, dtype=tf.int64)
X_val_tensor = tf.convert_to_tensor(X_val_tfidf, dtype=tf.float32)
y_val_tensor = tf.convert_to_tensor(y_val_encoded, dtype=tf.int64)

"""# FFNN model

"""

def build_ffnn_model(input_dim, label_count):
    model = tf.keras.Sequential([
        tf.keras.layers.InputLayer(input_shape=(input_dim,)),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(128, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(1e-4)),
        tf.keras.layers.Dense(label_count, activation='softmax')
    ])
    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

# # Build and train FFNN model
# ffnn_model = build_ffnn_model(X_train_tensor.shape[1], len(label_mapping))
# ffnn_model.fit(X_train_tensor, y_train_tensor, validation_data=(X_val_tensor, y_val_tensor), epochs=1, batch_size=32)

"""# CNN |"""

model = tf.keras.Sequential([
    # Input shape is inferred here
    tf.keras.layers.Conv1D(filters=128, kernel_size=3, activation='relu', padding='same', input_shape=(X_train_tfidf.shape[1], 1)),
    tf.keras.layers.MaxPooling1D(pool_size=2),

    tf.keras.layers.Conv1D(filters=64, kernel_size=3, activation='relu', padding='same'),
    tf.keras.layers.MaxPooling1D(pool_size=2),

    tf.keras.layers.Conv1D(filters=32, kernel_size=3, activation='relu', padding='same'),
    tf.keras.layers.MaxPooling1D(pool_size=2),

    tf.keras.layers.Flatten(),

    tf.keras.layers.Dense(128, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(1e-4)),
    tf.keras.layers.Dropout(0.5),  # Dropout for regularization

    tf.keras.layers.Dense(len(label_mapping), activation='softmax')  # Output layer with softmax activation
])

# Compile the model
model.compile(
    loss='sparse_categorical_crossentropy',
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005),
    metrics=['accuracy']
)

# Print the model summary (to ensure trainable parameters are not 0)
# model.summary()

# Training the model
epochs = 12
batch_size = 124

# with tf.device('/GPU:0'):  # Forces model training on GPU (Optional, only if GPU is detected)
#     model.fit(
#         X_train_tensor, y_train_tensor,
#         validation_data=(X_val_tensor, y_val_tensor),
#         epochs=epochs,
#         batch_size=batch_size,
#         shuffle=True
#     )

# # Evaluate the model on validation data
# val_loss, val_accuracy = model.evaluate(X_val_tensor, y_val_tensor)
# print(f"Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy*100:.2f}%")

# model.save('/kaggle/working/CNN1.h5')

"""# CNN ||"""

import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# Load and display the image
image_path = '/kaggle/input/cnn-2-archi/xLrP6IM.png'  # Replace with the path to your image
img = mpimg.imread(image_path)

# Display the image
plt.imshow(img)
plt.axis('off')  # Turn off axis labels for better display
plt.show()

# Tokenizer setup
tokenizer = Tokenizer(num_words=10000)
tokenizer.fit_on_texts(X_train)

word_index = tokenizer.word_index
X_train_padded = pad_sequences(tokenizer.texts_to_sequences(X_train), maxlen=1000)
X_val_padded = pad_sequences(tokenizer.texts_to_sequences(X_val), maxlen=1000)


embedding_matrix = np.zeros((len(word_index) + 1, EMBEDDING_DIM))
for word, i in word_index.items():
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        embedding_matrix[i] = embedding_vector

embedding_layer = Embedding(
    input_dim=len(word_index) + 1,
    output_dim=EMBEDDING_DIM,
    weights=[embedding_matrix],
    input_length=MAX_SEQUENCE_LENGTH,
    trainable=True
)

# CNN Model with GloVe Embeddings
def create_cnn_model():
    inputs = tf.keras.layers.Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
    embedding = embedding_layer(inputs)

    reshape = tf.keras.layers.Reshape((MAX_SEQUENCE_LENGTH, EMBEDDING_DIM, 1))(embedding)

    conv_0 = tf.keras.layers.Conv2D(num_filters, kernel_size=(filter_sizes[0], EMBEDDING_DIM), activation='relu')(reshape)
    conv_1 = tf.keras.layers.Conv2D(num_filters, kernel_size=(filter_sizes[1], EMBEDDING_DIM), activation='relu')(reshape)
    conv_2 = tf.keras.layers.Conv2D(num_filters, kernel_size=(filter_sizes[2], EMBEDDING_DIM), activation='relu')(reshape)

    maxpool_0 = tf.keras.layers.MaxPool2D(pool_size=(MAX_SEQUENCE_LENGTH - filter_sizes[0] + 1, 1))(conv_0)
    maxpool_1 = tf.keras.layers.MaxPool2D(pool_size=(MAX_SEQUENCE_LENGTH - filter_sizes[1] + 1, 1))(conv_1)
    maxpool_2 = tf.keras.layers.MaxPool2D(pool_size=(MAX_SEQUENCE_LENGTH - filter_sizes[2] + 1, 1))(conv_2)

    concatenated_tensor = tf.keras.layers.Concatenate(axis=1)([maxpool_0, maxpool_1, maxpool_2])
    flatten = tf.keras.layers.Flatten()(concatenated_tensor)
    dropout = tf.keras.layers.Dropout(drop)(flatten)
    output = tf.keras.layers.Dense(len(label_mapping), activation='softmax')(dropout)

    model = tf.keras.Model(inputs=inputs, outputs=output)
    model.compile(
        loss='sparse_categorical_crossentropy',
        optimizer=Adam(learning_rate=1e-4),
        metrics=['accuracy']
    )
    return model

cnn_model = create_cnn_model()
epochs = 1
batch_size = 64

# cnn_model.fit(
#     X_train_padded, y_train_encoded,
#     validation_data=(X_val_padded, y_val_encoded),
#     epochs=epochs,
#     batch_size=batch_size,
#     shuffle=True
# )

# # Evaluate the model
# val_loss, val_accuracy = cnn_model.evaluate(X_val_padded, y_val_encoded)
# print(f"Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy * 100:.2f}%")

"""# LSTM"""

# LSTM Classifier Model
class ImprovedLSTMClassifier(tf.keras.Model):
    def __init__(self, vocab_size, embedding_dim, output_dim, max_length=100):
        super(ImprovedLSTMClassifier, self).__init__()

        # Embedding layer with spatial dropout
        self.embedding = tf.keras.layers.Embedding(
            vocab_size,
            embedding_dim,
            input_length=max_length,
            embeddings_regularizer=tf.keras.regularizers.l2(1e-5)
        )
        self.spatial_dropout = tf.keras.layers.SpatialDropout1D(0.2)

        # First Bidirectional LSTM layer with higher units
        self.lstm1 = tf.keras.layers.Bidirectional(
            tf.keras.layers.LSTM(
                256,
                return_sequences=True,
                dropout=0.3,
                recurrent_dropout=0.3,
                kernel_regularizer=tf.keras.regularizers.l2(1e-5)
            )
        )

        # Second Bidirectional LSTM layer
        self.lstm2 = tf.keras.layers.Bidirectional(
            tf.keras.layers.LSTM(
                128,
                return_sequences=False,
                dropout=0.3,
                recurrent_dropout=0.3,
                kernel_regularizer=tf.keras.regularizers.l2(1e-5)
            )
        )

        # Dense layers with batch normalization and dropout
        self.batch_norm1 = tf.keras.layers.BatchNormalization()
        self.dropout1 = tf.keras.layers.Dropout(0.4)

        self.dense1 = tf.keras.layers.Dense(
            512,
            activation='relu',
            kernel_regularizer=tf.keras.regularizers.l2(1e-4),
            kernel_initializer='he_normal'
        )

        self.batch_norm2 = tf.keras.layers.BatchNormalization()
        self.dropout2 = tf.keras.layers.Dropout(0.4)

        self.dense2 = tf.keras.layers.Dense(
            256,
            activation='relu',
            kernel_regularizer=tf.keras.regularizers.l2(1e-4),
            kernel_initializer='he_normal'
        )

        # Output layer
        self.output_layer = tf.keras.layers.Dense(
            output_dim,
            activation='softmax',
            kernel_regularizer=tf.keras.regularizers.l2(1e-4)
        )

    def call(self, inputs, training=False):
        # Embedding with spatial dropout
        x = self.embedding(inputs)
        x = self.spatial_dropout(x, training=training)

        # LSTM layers
        x = self.lstm1(x, training=training)
        x = self.lstm2(x, training=training)

        # First dense block
        x = self.batch_norm1(x, training=training)
        x = self.dropout1(x, training=training)
        x = self.dense1(x)

        # Second dense block
        x = self.batch_norm2(x, training=training)
        x = self.dropout2(x, training=training)
        x = self.dense2(x)

        # Output
        return self.output_layer(x)

vocab_size = len(tokenizer.word_index) + 1
embedding_dim = 200
num_classes =5
# Initialize distributed strategy
strategy = tf.distribute.MirroredStrategy()
with strategy.scope():
    model = ImprovedLSTMClassifier(vocab_size, embedding_dim, num_classes)
    model.compile(
        loss='categorical_crossentropy',
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        metrics=['accuracy']
    )

# Train model directly with numpy arrays
# epochs = 1
# batch_size = 256
# model.fit(
#     X_train_padded,
#     y_train_encoded_categorical,
#     batch_size=batch_size,
#     epochs=epochs,
#     validation_data=(X_val_padded, y_val_encoded_categorical),
#     shuffle=True
# )

# # Evaluate
# val_loss, val_accuracy = model.evaluate(
#     X_val_padded,
#     y_val_encoded_categorical,
#     batch_size=batch_size
# )
# print(f"Validation Loss: {val_loss:.4f}")
# print(f"Validation Accuracy: {val_accuracy*100:.2f}%")

"""# RNN"""

# Function to create the RNN model
def create_rnn_model(vocab_size, embedding_dim, output_dim):
    inputs = tf.keras.Input(shape=(None,))  # Input layer for sequences
    embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)(inputs)  # Embedding layer
    # Bidirectional RNN layer
    rnn = tf.keras.layers.Bidirectional(tf.keras.layers.SimpleRNN(32, return_sequences=True, dropout=0.3))(embedding)  # Bidirectional RNN
    # Batch normalization
    batch_norm = tf.keras.layers.BatchNormalization()(rnn)
    # Get the last hidden state
    last_hidden_state = batch_norm[:, -1, :]  # Use the last hidden state
    # Fully connected layer with L2 regularization
    fc1 = tf.keras.layers.Dense(32, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(1e-4))(last_hidden_state)
    # Dropout layer
    dropout = tf.keras.layers.Dropout(0.5)(fc1)
    # Final output layer with softmax activation
    output = tf.keras.layers.Dense(output_dim, activation='softmax')(dropout)
    model = tf.keras.Model(inputs=inputs, outputs=output)
    return model

# Hyperparameters
vocab_size = len(vocab) + 1  # +1 for padding token
embedding_dim = 300  # Increase embedding dimension
output_dim = len(label_mapping)  # Number of categories (5 in this case)

# Create and compile the model inside the strategy scope for multi-GPU usage (optional)
strategy = tf.distribute.MirroredStrategy()  # Use multiple GPUs if available
print(f'Number of devices: {strategy.num_replicas_in_sync}')

with strategy.scope():
    model = create_rnn_model(vocab_size, embedding_dim, output_dim)

    # Compile the model
    model.compile(
        loss='categorical_crossentropy',  # Use categorical crossentropy for one-hot encoded labels
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        metrics=['accuracy']
    )

# Training the model
# epochs = 1

# #Train the model using tf.data API for batched data
# model.fit(
#     X_train_padded,
#     y_train_encoded_categorical,
#     batch_size=batch_size,
#     epochs=epochs,
#     validation_data=(X_val_padded, y_val_encoded_categorical),
# )
# # Evaluate the model on validation data
# val_loss, val_accuracy = model.evaluate(X_val_padded, y_val_encoded_categorical)
# print(f"Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy*100:.2f}%")



"""# Transformer |"""

import tensorflow as tf
import nltk
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import layers, models
import os

# Define the transformer encoder block as a function
def transformer_encoder(inputs, embed_dim, num_heads, ff_dim, dropout_rate=0.1, training=False):
    # Multi-Head Self-Attention
    attention = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)(inputs, inputs)
    attention = layers.Dropout(dropout_rate)(attention, training=training)  # Apply dropout if in training mode
    attention_output = layers.LayerNormalization(epsilon=1e-6)(inputs + attention)  # Add & Normalize
    # Feed-forward network
    ff_output = layers.Dense(ff_dim, activation='relu')(attention_output)
    ff_output = layers.Dense(embed_dim)(ff_output)
    ff_output = layers.Dropout(dropout_rate)(ff_output, training=training)  # Apply dropout if in training mode
    output = layers.LayerNormalization(epsilon=1e-6)(attention_output + ff_output)  # Add & Normalize

    return output

# Transformer Model
def transformer_model(vocab_size, embed_dim, num_heads, ff_dim, num_blocks, max_len, num_classes, dropout_rate=0.1):
    # Input layer: (batch_size, sequence_length)
    inputs = layers.Input(shape=(max_len,))  # Shape: (None, max_len)

    # Embedding layer
    embedding = layers.Embedding(vocab_size, embed_dim)(inputs)  # Shape: (None, max_len, embed_dim)

    # Positional Encoding (it should be of shape (max_len, embed_dim))
    positional_encoding = layers.Embedding(input_dim=max_len, output_dim=embed_dim)(tf.range(max_len))  # Shape: (max_len, embed_dim)

    # Add positional encoding to embedding
    x = embedding + positional_encoding  # Shape: (None, max_len, embed_dim)

    # Dropout after embedding + positional encoding
    x = layers.Dropout(dropout_rate)(x)

    # Transformer blocks
    for _ in range(num_blocks):
        x = transformer_encoder(x, embed_dim, num_heads, ff_dim, dropout_rate)

    # Global Average Pooling
    x = layers.GlobalAveragePooling1D()(x)  # Shape: (None, embed_dim)

    # Output layer
    outputs = layers.Dense(num_classes, activation='softmax')(x)  # Shape: (None, num_classes)

    # Create the model
    model = models.Model(inputs=inputs, outputs=outputs)

    return model

embed_dim = 64  # Embedding dimension
num_heads = 2  # Number of attention heads
ff_dim = 64  # Feed-forward layer dimension
num_blocks = 2  # Number of transformer blocks
max_len = 1000  # Max sequence length (fixed to match input data)
vocab_size = len(vocab) + 1  # Add padding index
num_classes = len(label_mapping)  # Number of output classes
dropout_rate = 0.1  # Dropout rate
batch_size=16
epochs = 1

# Create the model using the transformer function
model = transformer_model(vocab_size, embed_dim, num_heads, ff_dim, num_blocks, max_len, num_classes)

# Compile the model
model.compile(
    loss='categorical_crossentropy',  # For multi-class classification
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
    metrics=['accuracy']
)



# model.fit(
#     X_train_padded,
#     y_train_encoded_categorical,
#     batch_size=batch_size,
#     epochs=epochs,
#     validation_data=(X_val_padded, y_val_encoded_categorical),
#     shuffle=True
# )
# # Evaluate the model on validation data
# val_loss, val_accuracy = model.evaluate(val_dataset)
# print(f"Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy*100:.2f}%")

"""# Transformer ||"""

import tensorflow as tf
from transformers import TFRobertaForSequenceClassification, RobertaTokenizer
from sklearn.model_selection import train_test_split
import pandas as pd
import re
import emoji
import unicodedata

# Ensure GPU usage with memory growth enabled before any TensorFlow operation
physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
    print("Using GPU")
    try:
        # Enabling memory growth on the first GPU device
        tf.config.experimental.set_memory_growth(physical_devices[0], True)
    except RuntimeError as e:
        # Memory growth needs to be set before TensorFlow runtime initialization
        print(f"Error setting memory growth: {e}")
else:
    print("No GPU detected, using CPU")

# Set batch size and max_length variables
batch_size = 32  # You can change this value easily
max_length = 100  # You can change this value easily

# Extract texts and labels (replace this with actual loading code)
train_data = pd.read_csv('/kaggle/input/neural-network-dataset/Forum Discussion Categorization/train.csv')
X = train_data['Discussion'].fillna('').astype(str)
y = train_data['Category']

# Contraction mapping for text preprocessing
contraction_mapping = {
    "won't": "will not", "can't": "cannot", "don't": "do not", "didn't": "did not",
    "isn't": "is not", "aren't": "are not", "wasn't": "was not", "weren't": "were not",
    "hasn't": "has not", "haven't": "have not", "hadn't": "had not", "hasn't": "has not",
    "he's": "he is", "she's": "she is", "it's": "it is", "that's": "that is"
}

# Preprocessing function
def clean_text(text):
    # Convert to ASCII and remove non-alphanumeric characters
    text = unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('utf-8')
    # Expand contractions
    for word, replacement in contraction_mapping.items():
        text = text.replace(word, replacement)
    # Remove URLs, mentions, and special characters
    text = re.sub(r'http\S+|www\S+|@\S+', '', text)
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    text = emoji.demojize(text)  # Handle emojis
    return text.strip()

# Apply preprocessing
X_cleaned = X.apply(clean_text)

# Split the data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X_cleaned, y, test_size=0.2, random_state=42)

# Label mapping
label_mapping = {
    'Politics': 0,
    'Sports': 1,
    'Media': 2,
    'Market & Economy': 3,
    'STEM': 4
}
y_train_encoded = y_train.map(label_mapping)
y_val_encoded = y_val.map(label_mapping)

# Initialize RoBERTa tokenizer
tokenizer = RobertaTokenizer.from_pretrained('roberta-base')

# Tokenize input text for RoBERTa (input IDs and attention masks)
train_inputs = tokenizer(
    X_train.tolist(),
    padding=True,
    truncation=True,
    max_length=max_length,
    return_tensors="tf"
)

val_inputs = tokenizer(
    X_val.tolist(),
    padding=True,
    truncation=True,
    max_length=max_length,
    return_tensors="tf"
)

# Extract input_ids and attention_mask for training and validation
X_train_ids = train_inputs['input_ids']
train_attention_mask = train_inputs['attention_mask']

X_val_ids = val_inputs['input_ids']
val_attention_mask = val_inputs['attention_mask']

# Convert labels to tensors
y_train_encoded_tensor = tf.convert_to_tensor(y_train_encoded.tolist(), dtype=tf.int32)
y_val_encoded_tensor = tf.convert_to_tensor(y_val_encoded.tolist(), dtype=tf.int32)

# Create TensorFlow datasets with the defined batch size
train_dataset = tf.data.Dataset.from_tensor_slices((
    {"input_ids": X_train_ids, "attention_mask": train_attention_mask}, y_train_encoded_tensor
)).batch(batch_size).shuffle(buffer_size=1024)

val_dataset = tf.data.Dataset.from_tensor_slices((
    {"input_ids": X_val_ids, "attention_mask": val_attention_mask}, y_val_encoded_tensor
)).batch(batch_size)

# Load pre-trained RoBERTa model for embeddings (without classification head)
roberta_model = TFRobertaForSequenceClassification.from_pretrained('roberta-base', num_labels=5)

# Build the custom model
input_ids = tf.keras.layers.Input(shape=(max_length,), dtype=tf.int32, name="input_ids")
attention_mask = tf.keras.layers.Input(shape=(max_length,), dtype=tf.int32, name="attention_mask")

# Use the pre-trained RoBERTa model as a layer within a Keras model
roberta_output = roberta_model.roberta(input_ids, attention_mask=attention_mask)
cls_token = roberta_output.last_hidden_state  # Shape: (batch_size, seq_len, hidden_dim)

# Custom Transformer Layer (optional, if you want to add custom layers after RoBERTa)
def transformer_encoder(inputs, head_size, num_heads, ff_dim):
    attention_output = tf.keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=head_size)(inputs, inputs)
    attention_output = tf.keras.layers.LayerNormalization()(attention_output + inputs)

    ff_output = tf.keras.layers.Dense(ff_dim, activation='relu')(attention_output)
    ff_output = tf.keras.layers.Dense(inputs.shape[-1])(ff_output)
    encoder_output = tf.keras.layers.LayerNormalization()(ff_output + attention_output)

    return encoder_output

# Pass through custom transformer layers
transformer_output = transformer_encoder(cls_token, head_size=128, num_heads=5, ff_dim=128)

# Global average pooling over the sequence length (using the first token, cls_token)
pooling_output = tf.keras.layers.GlobalAveragePooling1D()(transformer_output)

# Output layer: classification using softmax
final_output = tf.keras.layers.Dense(len(label_mapping), activation='softmax')(pooling_output)

# Final model
model = tf.keras.Model(inputs=[input_ids, attention_mask], outputs=final_output)

# Compile the model
optimizer = tf.keras.optimizers.Adam(learning_rate=2e-6)
loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False)
metrics = [tf.keras.metrics.SparseCategoricalAccuracy(name="accuracy")]

# model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

# # Train the model
# epochs = 5
# history = model.fit(
#     train_dataset,
#     validation_data=val_dataset,
#     epochs=epochs
# )

# # Evaluate the model
# val_loss, val_accuracy = model.evaluate(val_dataset)
# print(f"Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.2f}%")

# # Save the model
# model.save('/kaggle/working/Transformer2.h5')



# test preprocessing

import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pandas as pd
import spacy
import re
import logging  # Import the logging module
import nltk
import os
import unicodedata
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
import emoji
from bs4 import BeautifulSoup
import string
import concurrent.futures

# Configure logging
logging.basicConfig(level=logging.WARNING, format='%(asctime)s - %(levelname)s - %(message)s')

# Load SpaCy English model with word vectors
nlp = spacy.load('en_core_web_md', disable=['parser', 'ner'])  # Disable parser and ner for speed

# Download NLTK resources
nltk.download('stopwords', quiet=True)
nltk.download('punkt', quiet=True)

# Initialize stopwords
stop_words = set(stopwords.words('english'))
stemmer = SnowballStemmer("english")

contraction_mapping = {
    "won't": "will not", "can't": "cannot", "don't": "do not", "didn't": "did not",
    "isn't": "is not", "aren't": "are not", "wasn't": "was not", "weren't": "were not",
    "hasn't": "has not", "haven't": "have not", "hadn't": "had not", "hasn't": "has not",
    "he's": "he is", "she's": "she is", "it's": "it is", "that's": "that is"
}

def normalize_elongations(text):
    return re.sub(r'(.)\1{2,}', r'\1\1', text)

def convert_to_ascii(input_str):
    nfkd_form = unicodedata.normalize('NFKD', input_str)
    return ''.join([c for c in nfkd_form if unicodedata.category(c) != 'Mn']).lower()

def clean_contractions(text, mapping):
    specials = ["’", "‘", "´", "`"]
    for s in specials:
        text = text.replace(s, "'")
    for word in mapping.keys():
        if word in text:
            text = text.replace(word, mapping[word])
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub(r"([?.!,¿])", r" \1 ", text)
    text = re.sub(r'[" "]+', " ", text)
    return text

def clean_text(text):
    if text is None:
        return ""
    if isinstance(text, str):
        try:
            text = text.encode('latin1').decode('utf-8')
        except (UnicodeEncodeError, UnicodeDecodeError):
            logging.warning(f"Unicode error during decoding of: {text}")
            pass
        text = emoji.demojize(text)
        text = BeautifulSoup(text, 'lxml').get_text()
        text = re.sub('<.*?>+', '', text)
        text = re.sub('\n', '', text)
        text = re.sub('\w*\d\w*', '', text)
        text = convert_to_ascii(text)
        text = normalize_elongations(text)
        text = clean_contractions(text, contraction_mapping)
        text = re.sub(r'http\S+|www\S+|@\S+|[^a-zA-Z\'\-\s]', '', text)  # keep apostrophe and dash
        doc = nlp(text)
        text = [stemmer.stem(token.text) for token in doc if token.text not in stop_words and not token.is_punct]
        text = ' '.join(text).strip()
        text = re.sub(r'\s+', ' ', text)
        return text
    else:
        return ""

# Load the test data
test_data = pd.read_csv('/kaggle/input/disc-dataset/ThursdayTest.csv')
X_test = test_data['Discussion'].fillna('').astype(str)

# Clean the test data
def parallel_clean_text(X_data, num_workers=os.cpu_count()):  # Adjust workers
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
        cleaned_text = list(executor.map(clean_text, X_data))
    return cleaned_text

X_test_cleaned = parallel_clean_text(X_test)

import joblib

vectorizer = joblib.load('/kaggle/working/tfidf_vectorizer.pkl')

X_test_tfidf = vectorizer.transform(X_test_cleaned).toarray()  # Transform the test data


X_test_tensor = tf.convert_to_tensor(X_test_tfidf, dtype=tf.float32)

# If you have a trained model, you can now predict on the test data
# model_FNN = tf.keras.models.load_model('FNN.h5')
# predictions = model_FNN.predict(X_test_tensor)

model_CNN1 = tf.keras.models.load_model('CNN1.h5')
predictions = model_CNN1.predict(X_test_tensor)


predicted_classes = tf.argmax(predictions, axis=1).numpy()

# Prepare the submission file
submission = pd.DataFrame({
    'SampleID': test_data['SampleID'],  # Assuming 'SampleID' is a column in the test data
    'Category': predicted_classes
})

# Save the predictions to a CSV file
submission.to_csv('/kaggle/working/CS_22 submission.csv', index=False)

print("Submission file generated successfully!")

# Load the test data (replace with your actual test data path)
test_data = pd.read_csv('/kaggle/input/disc-dataset/ThursdayTest.csv')
X_test = test_data['Discussion'].fillna('').astype(str)

# Apply preprocessing to the test data
X_test_cleaned = X_test.apply(clean_text)

# Tokenize the test data using the same tokenizer
test_inputs = tokenizer(
    X_test_cleaned.tolist(),
    padding=True,
    truncation=True,
    max_length=max_length,
    return_tensors="tf"
)

# Extract input_ids and attention_mask for the test data
X_test_ids = test_inputs['input_ids']
test_attention_mask = test_inputs['attention_mask']

# Create the test dataset
test_dataset = tf.data.Dataset.from_tensor_slices((
    {"input_ids": X_test_ids, "attention_mask": test_attention_mask}
)).batch(batch_size)

# Make predictions on the test set
model_Trans = tf.keras.models.load_model('/kaggle/working/Transformer2.h5')
predictions = model_Trans.predict(X_test_tensor)
# Convert predictions to class labels
predicted_classes = tf.argmax(predictions.logits, axis=1).numpy()

# Prepare the submission file (assuming 'SampleID' exists in the test data)
submission = pd.DataFrame({
    'SampleID': test_data['SampleID'],  # Replace with the actual column name if it's different
    'Category': predicted_classes
})

# Save the predictions to a CSV file (you can change the path as needed)
submission.to_csv('/kaggle/working/CS_22_submission.csv', index=False)

print("Submission file generated successfully!")









import tensorflow as tf
from transformers import TFBertForSequenceClassification, BertTokenizer
from sklearn.model_selection import train_test_split
import pandas as pd
import re
import emoji
import unicodedata

# Ensure GPU usage with memory growth enabled before any TensorFlow operation
physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
    print("Using GPU")
    try:
        # Enabling memory growth on the first GPU device
        tf.config.experimental.set_memory_growth(physical_devices[0], True)
    except RuntimeError as e:
        # Memory growth needs to be set before TensorFlow runtime initialization
        print(f"Error setting memory growth: {e}")
else:
    print("No GPU detected, using CPU")

# Set batch size and max_length variables
batch_size = 32  # You can change this value easily
max_length = 256  # You can change this value easily


# Extract texts and labels (replace this with actual loading code)
train_data = pd.read_csv('/kaggle/input/neural-network-dataset/Forum Discussion Categorization/train.csv')
X = train_data['Discussion'].fillna('').astype(str)
y = train_data['Category']

# Contraction mapping for text preprocessing
contraction_mapping = {
    "won't": "will not", "can't": "cannot", "don't": "do not", "didn't": "did not",
    "isn't": "is not", "aren't": "are not", "wasn't": "was not", "weren't": "were not",
    "hasn't": "has not", "haven't": "have not", "hadn't": "had not", "hasn't": "has not",
    "he's": "he is", "she's": "she is", "it's": "it is", "that's": "that is"
}

# Preprocessing function
def clean_text(text):
    # Convert to ASCII and remove non-alphanumeric characters
    text = unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('utf-8')
    # Expand contractions
    for word, replacement in contraction_mapping.items():
        text = text.replace(word, replacement)
    # Remove URLs, mentions, and special characters
    text = re.sub(r'http\S+|www\S+|@\S+', '', text)
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    text = emoji.demojize(text)  # Handle emojis
    return text.strip()

# Apply preprocessing
X_cleaned = X.apply(clean_text)


print("done!")
# Split the data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X_cleaned, y, test_size=0.2, random_state=42)

# Label mapping
label_mapping = {
    'Politics': 0,
    'Sports': 1,
    'Media': 2,
    'Market & Economy': 3,
    'STEM': 4
}
y_train_encoded = y_train.map(label_mapping)
y_val_encoded = y_val.map(label_mapping)

# Initialize BERT tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Tokenize input text for BERT (input IDs and attention masks)
train_inputs = tokenizer(
    X_train.tolist(),
    padding=True,
    truncation=True,
    max_length=max_length,
    return_tensors="tf"
)

val_inputs = tokenizer(
    X_val.tolist(),
    padding=True,
    truncation=True,
    max_length=max_length,
    return_tensors="tf"
)

# Extract input_ids and attention_mask for training and validation
X_train_ids = train_inputs['input_ids']
train_attention_mask = train_inputs['attention_mask']

X_val_ids = val_inputs['input_ids']
val_attention_mask = val_inputs['attention_mask']

# Convert labels to tensors
y_train_encoded_tensor = tf.convert_to_tensor(y_train_encoded.tolist(), dtype=tf.int32)
y_val_encoded_tensor = tf.convert_to_tensor(y_val_encoded.tolist(), dtype=tf.int32)

# Create TensorFlow datasets with the defined batch size
train_dataset = tf.data.Dataset.from_tensor_slices((
    {"input_ids": X_train_ids, "attention_mask": train_attention_mask}, y_train_encoded_tensor
)).batch(batch_size).shuffle(buffer_size=1024)

val_dataset = tf.data.Dataset.from_tensor_slices((
    {"input_ids": X_val_ids, "attention_mask": val_attention_mask}, y_val_encoded_tensor
)).batch(batch_size)

# Load pre-trained BERT model for embeddings (without classification head)
bert_model = TFBertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=5)

# Extract the transformer layers (without the classification head)
bert_transformer = bert_model.bert

# Custom Transformer Layer (optional, if you want to add custom layers after BERT)
def transformer_encoder(inputs, head_size, num_heads, ff_dim):
    attention_output = tf.keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=head_size)(inputs, inputs)
    attention_output = tf.keras.layers.LayerNormalization()(attention_output + inputs)

    ff_output = tf.keras.layers.Dense(ff_dim, activation='relu')(attention_output)
    ff_output = tf.keras.layers.Dense(inputs.shape[-1])(ff_output)
    encoder_output = tf.keras.layers.LayerNormalization()(ff_output + attention_output)

    return encoder_output

# Build the custom model
input_ids = tf.keras.layers.Input(shape=(max_length,), dtype=tf.int32, name="input_ids")
attention_mask = tf.keras.layers.Input(shape=(max_length,), dtype=tf.int32, name="attention_mask")

# Use the pre-trained BERT model to get embeddings
bert_output = bert_transformer(input_ids, attention_mask=attention_mask)
cls_token = bert_output.last_hidden_state  # Shape: (batch_size, seq_len, hidden_dim)

# Pass through custom transformer layers
transformer_output = transformer_encoder(cls_token, head_size=128, num_heads=3, ff_dim=128)

# Global average pooling over the sequence length (using the first token, cls_token)
pooling_output = tf.keras.layers.GlobalAveragePooling1D()(transformer_output)

# Output layer: classification using softmax
final_output = tf.keras.layers.Dense(len(label_mapping), activation='softmax')(pooling_output)

# Final model
model = tf.keras.Model(inputs=[input_ids, attention_mask], outputs=final_output)

# Compile the model
optimizer = tf.keras.optimizers.Adam(learning_rate=2e-6)
loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False)
metrics = [tf.keras.metrics.SparseCategoricalAccuracy(name="accuracy")]

model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

# Train the model
epochs = 5
history = model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=epochs
)

# Evaluate the model
val_loss, val_accuracy = model.evaluate(val_dataset)
print(f"Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.2f}%")

model.save('/kaggle/working/Transformer2.h5')

