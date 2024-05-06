import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Load data
data = pd.read_csv('train_cleaned.csv')
data['text'] = data['text'].fillna('')  # Handle NaN values
data['text'] = data['text'].astype(str)  # Ensure all entries are strings

# Parameters
vocab_size = 10000
embedding_dim = 16
max_length = 100
padding_type = 'post'
trunc_type = 'post'
oov_tok = "<OOV>"
training_size = 20000

# Prepare data
sentences = data['text'].tolist()
labels = data['sentiment_score'].tolist()
training_sentences = sentences[:training_size]
testing_sentences = sentences[training_size:]
training_labels = labels[:training_size]
testing_labels = labels[training_size:]

# Tokenization and padding
tokenizer = Tokenizer(num_words=vocab_size, oov_token=oov_tok)
tokenizer.fit_on_texts(training_sentences)
training_sequences = tokenizer.texts_to_sequences(training_sentences)
training_padded = pad_sequences(training_sequences, maxlen=max_length, padding=padding_type, truncating=trunc_type)
testing_sequences = tokenizer.texts_to_sequences(testing_sentences)
testing_padded = pad_sequences(testing_sequences, maxlen=max_length, padding=padding_type, truncating=trunc_type)

# Convert to numpy arrays
training_padded = np.array(training_padded)
training_labels = np.array(training_labels).astype(np.int32)
testing_padded = np.array(testing_padded)
testing_labels = np.array(testing_labels).astype(np.int32)

# Model definition
model = tf.keras.Sequential([
    tf.keras.layers.Embedding(vocab_size, embedding_dim, input_length=max_length),
    tf.keras.layers.GlobalAveragePooling1D(),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(3, activation='softmax')
])

model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Model training
model.fit(training_padded, training_labels, epochs=30, validation_data=(testing_padded, testing_labels), verbose=2)

# Save the trained model
model.save('my_trained_model.h5')
