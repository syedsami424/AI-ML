import tensorflow as tf
from tensorflow.keras.layers import Input, LSTM, Dense, Flatten, Dropout, SimpleRNN, GlobalMaxPool1D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam, SGD
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('https://raw.githubusercontent.com/lazyprogrammer/machine_learning_examples/master/tf2.0/sbux.csv')

#Currently, dimensions are: (1259,)..therefore, we expand dimensions
series = df['close'].values.reshape(-1, 1)
print(series.shape) #now, the shape is N x D (1259, 1)

T = 10 # make model learn from first 10 samples
D = 1 # for now, we only predict 'close' , so only one feature
series = scaler.fit_transform(series)

X = []
Y = []
for t in range(len(series)-T):
    x = series[t:t+T]
    X.append(x)
    y = series[t+T]
    Y.append(y)
#make them into numpy arrays so that we can apply reshape
X = np.array(X)
Y = np.array(Y)
N = len(X)
#currently, the shape is (N x D). so we make it in N X T X D
X = X.reshape(-1, T, 1)
print(X.shape)

i = Input(shape=(T, D))
x = LSTM(10)(i)
x = Dense(1)(x)
model = Model(i, x)

model.compile(
optimizer=Adam(learning_rate=0.1),
loss='mse'
)

r = model.fit(X[:-N//4], Y[:-N//4], epochs=200, validation_data=(X[-N//2:], Y[-N//2:]))

result = model.predict(X)
predictions = result[:,0]
plt.plot(predictions)
plt.show()