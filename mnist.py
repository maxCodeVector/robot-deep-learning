from __future__ import print_function
from tensorflow.python.keras.models import Sequential, load_model
from tensorflow.python.keras.layers import Dense, Dropout
from tensorflow.python.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.python.keras.datasets import mnist
from tensorflow.python.keras.utils import to_categorical
import os
import time


def build_model(model_name):
    if os.path.exists(model_name):

        print("loading existing model.")
        model = load_model(model_name)
    else:
        print("making new model")
        model = Sequential()
        model.add(Dense(1024, input_shape=(784,), activation='relu'))
        model.add(Dropout(0.3))
        model.add(Dense(256, activation='relu'))
        model.add(Dropout(0.3))
        model.add(Dense(10, activation='softmax'))

    return model


def train(model: Sequential, train_x, train_y,
          epoches, test_x, test_y, model_file):
    model.compile(loss='categorical_crossentropy',
                  optimizer='sgd',
                  metrics=['accuracy'])
    print('running for %d epoches.' % epoches)
    save_model = ModelCheckpoint(model_file)
    stop_model = EarlyStopping(min_delta=0.001, patience=10)

    model.fit(x=train_x, y=train_y,
              shuffle=True,
              batch_size=60,
              epochs=epoches,
              validation_data=(test_x, test_y),
              callbacks=[save_model, stop_model])
    print("Done training, Now evaluating.")
    loss, acc = model.evaluate(x=test_x, y=test_y)

    print("Final loss: %3.2f Final accuracy: %3.2f" % (loss, acc))


def load_mnist():
    (train_x, train_y), (test_x, test_y) = mnist.load_data()

    train_x = train_x.reshape(train_x.shape[0], 784).astype('float32') / 255.0
    test_x = test_x.reshape(test_x.shape[0], 784).astype('float32') / 255.0

    train_y = to_categorical(train_y, 10)
    test_y = to_categorical(test_y, 10)

    return (train_x, train_y), (test_x, test_y)


def main():
    model_path = 'minist.hd5'
    model = build_model(model_path)
    (train_x, train_y), (test_x, test_y) = load_mnist()
    train(model, train_x, train_y, 50, test_x, test_y, model_path)


if __name__ == '__main__':
    start = time.time()
    main()
    print("cost time: ", time.time() - start)
