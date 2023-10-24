import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

####################### STEP1: LOAD DATA ##############################
df = pd.read_csv("moore.csv", header=None).to_numpy()
print(df)
#separate x and y from above:-
X = df[:,0].reshape(-1, 1)
Y = df[:,1]
#some preprocessing but NO SCALING   (this is just done as an example, normally you always have to scale)
Y = np.log(Y)
X = X - X.mean()
print(X.shape) # (162, 1)

model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(1,)),
    tf.keras.layers.Dense(1)
])

#since there was no scaling done, we use a custom optimizer instead of ADAM and loss = mse (mean square error)
#optimizer = tf.keras.optimizers.SGD(learning rate (lr), momentum)
model.compile(
    optimizer = tf.keras.optimizers.SGD(0.001, 0.9),
    loss = 'mse'
)

def schedule(epoch, lr):
    if epoch >= 50:
        return 0.0001
    return 0.001
scheduler = tf.keras.callbacks.LearningRateScheduler(schedule)

r = model.fit(X, Y, epochs=200, callbacks=[scheduler])