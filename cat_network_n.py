import os.path
import sys

from tensorflow.python.keras.callbacks import ModelCheckpoint
from tensorflow.python.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout
from tensorflow.python.keras.models import load_model, Sequential
from tensorflow.python.keras.optimizers import SGD
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator

if len(sys.argv) > 1:
    MODEL_FILE = sys.argv[1]
else:
    MODEL_FILE = "cat_net.hd5"
print("model file will save in", MODEL_FILE)


def create_model():

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


def load_existing(model_file):
    model = load_model(model_file)

    num_layers = len(model.layers)

    for layer in model.layers[:num_layers-3]:
        layer.trainable = False

    for layer in model.layers[num_layers-3:]:
        layer.trainable = True

    return model


def train(model_file, train_path, validation_path,
           steps=32, num_epochs=28, save_period=1):
    if os.path.exists(model_file):
        print('\n*** existing model found at {}. Loading. ***\n\n'.format(model_file))
        model = load_existing(model_file)
    else:
        print("\n*** Creating new model ***\n\n")
        model = create_model()

    check_point = ModelCheckpoint(model_file, period=save_period)
    model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True
    )
    test_datagen = ImageDataGenerator(rescale=1./255)
    train_generator = train_datagen.flow_from_directory(
        train_path,
        target_size=(249, 249),
        batch_size=2,
        class_mode='categorical'
    )
    validation_generator = test_datagen.flow_from_directory(
        validation_path,
        target_size=(249, 249),
        batch_size=2,
        class_mode='categorical'
    )
    model.fit_generator(
        train_generator,
        steps_per_epoch=steps,
        epochs=num_epochs,
        callbacks=[check_point, ],
        validation_data=validation_generator,
        validation_steps=50
    )
    for layer in model.layers[:249]:
        layer.trainable = False

    for layer in model.layers[249:]:
        layer.trainable = True

    model.compile(optimizer=SGD(lr=0.00001, momentum=0.9),
                  loss='categorical_crossentropy', metrics=['accuracy'])
    model.fit_generator(
        train_generator,
        steps_per_epoch=steps,
        epochs=num_epochs,
        callbacks=[check_point],
        validation_data=validation_generator,
        validation_steps=50
    )
    print("Done training, Now evaluating.")


def main():
    train(MODEL_FILE, train_path='cat_dataset', validation_path='cat_dataset_test',
          steps=12, num_epochs=10)


if __name__ == '__main__':
    main()
