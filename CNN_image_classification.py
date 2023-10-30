'''
-Convolutional neural networks (*). (convolution means complex)
-This is used in image modifying or classifying. (ex: applying a filter on an image)
-Accepted data dimensions: (N, 28, 28, 1)  (Samples, height, width, color)
'''
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Input, Conv2D, Dense, Flatten, Dropout
from tensorflow.keras.models import Model
import tensorflow as tf

fashion_mnist = tf.keras.datasets.fashion_mnist
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

print("Normal shape:",x_train.shape) # (60000, 28, 28)
#but accepted shape is 4D, not 3D. Therefore:-
x_train = np.expand_dims(x_train, -1)
x_test = np.expand_dims(x_test, -1)
print("Expanded shape:",x_train.shape)
K = len(set(y_train))

#print(x_test[0].shape)
#print(test_image.shape)

#here, we built the model a little differently. we use the "functional API"
i = Input(shape=x_train[0].shape)
#below, a 3 x 3 filter learns 32 features within the area of the input image it covers and shifts 2 pixels per iteration.
x = Conv2D(32, (3, 3), strides=2, activation='relu')(i)
x = Conv2D(64, (3, 3), strides=2, activation='relu')(x)
x = Conv2D(128, (3, 3), strides=2, activation='relu')(x)
x = Flatten()(x)
x = Dropout(0.2)(x)
x = Dense(512, activation='relu')(x)
x = Dropout(0.2)(x)
x = Dense(K, activation='softmax')(x)
model = Model(i, x)

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

r = model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=1)
print()
print()
test_img = x_test[23]
test_img = np.expand_dims(test_img, axis = 0)
predictions = model.predict(test_img)
predicted = np.argmax(predictions)

class_labels = {
    0: "T-shirt/top",
    1: "Trouser",
    2: "Pullover",
    3: "Dress",
    4: "Coat",
    5: "Sandal",
    6: "Shirt",
    7: "Sneaker",
    8: "Bag",
    9: "Ankle boot"
}

plt.imshow(test_img.reshape(28,28), cmap='gray')
print(class_labels[predicted])
plt.show()
