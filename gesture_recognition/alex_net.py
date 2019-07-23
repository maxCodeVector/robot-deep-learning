import os.path

from tensorflow.python.keras.callbacks import ModelCheckpoint
from tensorflow.python.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout
from tensorflow.python.keras.models import load_model, Sequential
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator

MODEL_FILE = 'alex_gesture_network.hd5'


def get_alnext_net(num_classes, target_size=(227, 227, 3)):

    """
    使用AlexNet结构
    输入
    images      输入的图像
    batch_size  每个批次的大小
    n_classes   n分类
    keep_prob   droupout保存的比例（ 设置神经元被选中的概率）
    返回
    softmax_linear 还差一个softmax
    """

    model = Sequential()

    # 1st Convolutional Layer
    model.add(Conv2D(filters=96, input_shape=target_size,
                     kernel_size=(11, 11),
                     strides=(4, 4),
                     padding='valid',
                     activation='relu'))
    # Max Pooling
    model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='valid'))

    # 2nd Convolutional Layer
    model.add(Conv2D(filters=256, kernel_size=(5, 5), strides=(1, 1), padding='same', activation='relu'))
    # Max Pooling
    model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='valid'))

    # 3rd Convolutional Layer
    model.add(Conv2D(filters=384, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu'))

    # 4th Convolutional Layer
    model.add(Conv2D(filters=384, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu'))

    # 5th Convolutional Layer
    model.add(Conv2D(filters=256, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu'))
    # Max Pooling
    model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='valid'))

    # Passing it to a Fully Connected layer
    model.add(Flatten())
    # 1st Fully Connected Layer
    # model.add(Dense(9126,  activation='relu'))
    # # Add Dropout to prevent overfitting
    # model.add(Dropout(0.4))

    # 2nd Fully Connected Layer
    model.add(Dense(1024, activation='relu'))
    model.add(Dropout(0.3))

    # 3rd Fully Connected Layer
    model.add(Dense(1024, activation='relu'))
    model.add(Dropout(0.3))
    # Output Layer
    model.add(Dense(num_classes, activation='softmax'))
    model.summary()
    return model


def load_existing(model_file):
    model = load_model(model_file)
    return model


def train(model_file, train_path, validation_path, target_size=(227, 227),
         num_classes=5, steps=32, num_epochs=28):
    if os.path.exists(model_file):
        print('\n*** existing model found at {}. Loading. ***\n\n'.format(model_file))
        model = load_existing(model_file)
    else:
        print("\n*** Creating new model ***\n\n")
        model = get_alnext_net(num_classes=num_classes)

    check_point = ModelCheckpoint(model_file, period=1)
    model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
    train_datagen = ImageDataGenerator(
        rescale=1. / 255,
        shear_range=0.3,
        zoom_range=0.3,
        horizontal_flip=True,
        vertical_flip=True,

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


def main():
    train(MODEL_FILE, train_path='../../hand_1to5', validation_path='../../hand_1to5_test',
          steps=150, num_epochs=50, num_classes=6)


if __name__ == '__main__':
    main()
