from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Input, Dense, GlobalMaxPooling1D, LSTM, Embedding
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('spam.csv', encoding='ISO-8859-1')
print(df.head())
# data came in a weird format, so let's alter it a little

df = df.drop(['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'], axis = 1)
df.columns = ['labels', 'data']
print(df.head())

#map 0 for ham (not spam) 1 for spam
df['binary_labels'] = df['labels'].map({'ham': 0, 'spam': 1})
Y = df['binary_labels'].values
df_train, df_test, Ytrain, Ytest = train_test_split(df['data'], Y, test_size=0.33)

MAX_VOCAB_SIZE = 20000
tokenizer = Tokenizer(num_words=MAX_VOCAB_SIZE)
tokenizer.fit_on_texts(df_train)
sequences_train = tokenizer.texts_to_sequences(df_train)
sequences_test = tokenizer.texts_to_sequences(df_test)
#print(sequences_train)

word2idx = tokenizer.word_index
V = len(word2idx)

# Apply padding to training sequence
data_train = pad_sequences(sequences_train)
MAX_SEQUENCE_LENGTH = data_train.shape[1]
data_train = pad_sequences(sequences_train, maxlen=MAX_SEQUENCE_LENGTH, padding='post')
print(data_train)
print(data_train.shape)
# Apply padding to testing sequence
data_test = pad_sequences(sequences_test, maxlen=MAX_SEQUENCE_LENGTH, padding='post')

#Model creation:-
D = 20
M = 15
i = Input(shape=(MAX_SEQUENCE_LENGTH,))
x = Embedding(V+1, D)(i)
x = LSTM(M, return_sequences=True)(x)
x = Dense(1, activation='sigmoid')(x)

model = Model(i, x)

model.compile(
    loss='binary_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
)

Ytrain = Ytrain.reshape(-1, 1)
Ytest = Ytest.reshape(-1, 1)
r = model.fit(data_train, Ytrain, epochs=10, validation_data=(data_test, Ytest))
predictions = model.predict(data_test)
