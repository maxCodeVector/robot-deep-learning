from __future__ import print_function
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
from tensorflow.python.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.python.keras.optimizers import SGD
import numpy as np


def create_model(input_shape, hid_size, num_outputs):
    model = Sequential()
    model.add(Dense(hid_size, input_shape=input_shape, activation='relu'))
    model.add(Dense(num_outputs, activation='sigmoid'))
    return model


def train(model:Sequential,
          x_train, y_train, steps, epochs,
          x_test, y_test, model_name):
    model.compile(
        optimizer=SGD(lr=0.7, momentum=0.3),
        loss='mean_squared_error',
        metrics=['accuracy'])
    save_callback = ModelCheckpoint(filepath=model_name)

    early_stop = EarlyStopping(monitor='loss', min_delta=0.01, patience=20)

    model.fit(x=x_train, y=y_train,
              steps_per_epoch=steps, epochs=epochs, shuffle=True,
              callbacks=[save_callback, early_stop])

    print(model.evaluate(x=x_test, y=y_test))


def predict(model:Sequential, x):
    print(model.predict(x))


def main():
    x = [(0., 0,), (0., 1.), (1., 0.), (1, 1)]
    y = [0., 1., 1., 0.]
    x_train = np.asarray(x)
    y_train = np.asarray(y)

    print(x_train)
    print(y_train)
    model = create_model((2,), 8, 1)
    train(model, x_train, y_train, 16, 1000, x_train, y_train, 'xor.hd5')
    predict(model, x_train)


if __name__ == '__main__':
    print("hello")
    main()
