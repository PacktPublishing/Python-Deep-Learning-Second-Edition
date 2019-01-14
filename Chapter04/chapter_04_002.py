# for reproducibility
from numpy.random import seed

seed(1)
from tensorflow import set_random_seed

set_random_seed(1)

from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.layers import Convolution2D, MaxPooling2D
from keras.layers import Flatten

from keras.utils import np_utils

(X_train, Y_train), (X_test, Y_test) = mnist.load_data()

X_train = X_train.reshape(60000, 28, 28, 1)
X_test = X_test.reshape(10000, 28, 28, 1)

Y_train = np_utils.to_categorical(Y_train, 10)
Y_test = np_utils.to_categorical(Y_test, 10)

model = Sequential([
    Convolution2D(filters=32,
                  kernel_size=(3, 3),
                  input_shape=(28, 28, 1)),  # first conv layer
    Activation('relu'),
    Convolution2D(filters=32,
                  kernel_size=(3, 3)),  # second conv layer
    Activation('relu'),
    MaxPooling2D(pool_size=(2, 2)),  # max pooling layer
    Flatten(),  # flatten the output tensor
    Dense(64),  # fully-connected hidden layer
    Activation('relu'),
    Dense(10),  # output layer
    Activation('softmax')])

print(model.summary())

model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer='adadelta')

model.fit(X_train, Y_train, batch_size=100, epochs=5, validation_split=0.1, verbose=1)

score = model.evaluate(X_test, Y_test, verbose=1)
print('Test accuracy:', score[1])
