import os
import time

from tensorflow.python.keras.callbacks import ModelCheckpoint
from tensorflow.python.keras.layers import Conv2D, MaxPooling2D
from tensorflow.python.keras.layers import Dense, Flatten
from tensorflow.python.keras.models import Sequential, load_model
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator

MODEL_NAME = 'hand_gesture.hd5'


def build_model(model_name, num_class=5):
    # turn left , right, rotation, forward, backward

    if os.path.exists(model_name):

        print("loading existing model.")
        model = load_model(model_name)
    else:
        print("\n\n****making new model****\n\n")
        model = Sequential()
        model.add(Conv2D(32, kernel_size=(5, 5),
                         input_shape=(256, 256, 6),
                         padding='same',
                         activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=2))

        model.add(Conv2D(32, kernel_size=(5, 5), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=2))

        model.add(Conv2D(32, kernel_size=(3, 3), activation='relu'))
        # model.add(MaxPooling2D(pool_size=(2, 2), strides=2))
        model.add(Flatten())

        model.add(Dense(300, activation='relu'))
        # model.add(Dropout(0.3))
        model.add(Dense(num_class, activation='softmax'))

    return model


def train(model_name, train_path, validation_path, epochs=20, steps=200):
    model = build_model(model_name)

    check_point = ModelCheckpoint(model_name, period=1)
    model.summary()
    model.compile(optimizer='rmsprop', loss='mean_squared_error', metrics=['accuracy'])
    train_datagen = ImageDataGenerator(
        rescale=1. / 255,
        shear_range=0.3,
        zoom_range=0.3,
        # horizontal_flip=True,

        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        brightness_range=(0.8, 1.2)
    )
    test_datagen = ImageDataGenerator(rescale=1. / 255)
    train_generator = train_datagen.flow_from_directory(
        train_path,
        target_size=(256, 256),
        batch_size=32,
        class_mode='categorical'
    )
    validation_generator = test_datagen.flow_from_directory(
        validation_path,
        target_size=(256, 256),
        batch_size=32,
        class_mode='categorical'
    )
    model.fit_generator(
        train_generator,
        steps_per_epoch=steps,
        epochs=epochs,
        callbacks=[check_point, ],
        validation_data=validation_generator,
        validation_steps=50
    )

    print("Done training, Now evaluating.")


if __name__ == '__main__':
    start = time.time()
    train(MODEL_NAME, "", "")
    print("cost time: ", time.time()-start)
