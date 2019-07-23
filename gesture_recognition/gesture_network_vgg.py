import os.path

from tensorflow.python.keras.applications.vgg19 import VGG19
from tensorflow.python.keras.callbacks import ModelCheckpoint
from tensorflow.python.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.python.keras.models import Model, load_model
from tensorflow.python.keras.optimizers import SGD
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator

MODEL_FILE = 'gesture_network_vgg.hd5'


def create_model(num_classes):
    # base_model = InceptionV3(include_top=False, weights='imagenet')
    base_model = VGG19(include_top=False, weights='imagenet')

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(256, activation='relu')(x)
    predictions = Dense(num_classes, activation='softmax')(x)

    for layer in base_model.layers:
        layer.trainable = False

    model = Model(base_model.input, predictions)

    return model


def load_existing(model_file):
    model = load_model(model_file)

    num_layers = len(model.layers)

    for layer in model.layers[:num_layers-3]:
        layer.trainable = False

    for layer in model.layers[num_layers-3:]:
        layer.trainable = True

    return model


def train(model_file, train_path, validation_path, target_size=(256, 256),
         num_classes=5, steps=32, num_epochs=28):
    if os.path.exists(model_file):
        print('\n*** existing model found at {}. Loading. ***\n\n'.format(model_file))
        model = load_existing(model_file)
    else:
        print("\n*** Creating new model ***\n\n")
        model = create_model(num_classes=num_classes)

    check_point = ModelCheckpoint(model_file, period=1)
    model.compile(optimizer='rmsprop', loss='categorical_crossentropy')
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
    test_datagen = ImageDataGenerator(rescale=1./255)
    train_generator = train_datagen.flow_from_directory(
        train_path,
        target_size=target_size,
        batch_size=32,
        class_mode='categorical'
    )
    validation_generator = test_datagen.flow_from_directory(
        validation_path,
        target_size=target_size,
        batch_size=32,
        class_mode='categorical'
    )
    model.fit_generator(
        train_generator,
        steps_per_epoch=steps,
        epochs=num_epochs,
        callbacks=[check_point, ],
        validation_data=validation_generator,
        validation_steps=50,
        shuffle=True,
    )
    for layer in model.layers[:249]:
        layer.trainable = False

    for layer in model.layers[249:]:
        layer.trainable = True

    model.compile(optimizer=SGD(lr=0.00001, momentum=0.9),
                  loss='categorical_crossentropy')
    model.fit_generator(
        train_generator,
        steps_per_epoch=steps,
        epochs=num_epochs,
        callbacks=[check_point],
        validation_data=validation_generator,
        validation_steps=50
    )


def main():
    train(MODEL_FILE, train_path='flower_photos', validation_path='flower_photos',
          steps=120, num_epochs=10)


if __name__ == '__main__':
    main()
