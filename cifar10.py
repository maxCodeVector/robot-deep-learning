import os
from tensorflow.python.keras import losses
from tensorflow.python.keras.models import Sequential, load_model
from tensorflow.python.keras.layers import Dense, Dropout, Flatten
from tensorflow.python.keras.layers import Conv2D, MaxPooling2D
from tensorflow.python.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.python.keras.optimizer_v2.rmsprop import RMSProp
from tensorflow.python.keras.utils import to_categorical
from tensorflow.python.keras.datasets import cifar10


MODEL_NAME = 'cifar10.hd5'


def build_model(model_name):

    if os.path.exists(model_name):

        print("loading existing model.")
        model = load_model(model_name)
    else:
        print("making new model")
        model = Sequential()
        model.add(Conv2D(32, kernel_size=(3, 3),
                         input_shape=(32, 32, 3),
                         padding='same',
                         activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=2))
        model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=2))
        model.add(Flatten())
        model.add(Dense(1024, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(10, activation='softmax'))
    return model


def load_cifar10():
    # Load the CIFAR10 data set.
    # Convert it to a Numpy array
    # Normalize inputs to [0,1]
    # Convert labels to one-hot array

    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    train_x = x_train.astype('float32') / 255.0
    test_x = x_test.astype('float32') / 255.0

    train_y = to_categorical(y_train, 10)
    test_y = to_categorical(y_test, 10)

    return (train_x, train_y), (test_x, test_y)


def train(model:Sequential, epochs,
          train_x, train_y, test_x, test_y):
    model.compile(optimizer=RMSProp(), loss=losses.mean_squared_error, metrics=['accuracy'])

    print('running for %d epoches.' % epochs)
    save_model = ModelCheckpoint(MODEL_NAME)
    stop_model = EarlyStopping(min_delta=0.0002, patience=10)
    print("start training")
    model.fit(x=train_x, y=train_y,
              shuffle=True,
              batch_size=32,
              epochs=epochs,
              validation_data=(test_x, test_y),
              callbacks=[save_model, stop_model])
    print("Done training, Now evaluating.")

    loss, acc = model.evaluate(x=test_x, y=test_y)
    print("Final loss: %3.2f Final accuracy: %3.2f" % (loss, acc))


def main():
    (train_x, train_y), (test_x, test_y) = load_cifar10()
    model = build_model(MODEL_NAME)
    train(model, 1, train_x, train_y, test_x, test_y)


if __name__ == '__main__':
    main()
