import os
import sys

import numpy as np
from PIL import Image
from tensorflow.python.keras import Input
from tensorflow.python.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.python.keras.layers import BatchNormalization, MaxPool2D, Flatten
from tensorflow.python.keras.layers import Dense, ConvLSTM2D, MaxPool3D, Activation, Convolution3D, concatenate
from tensorflow.python.keras.models import load_model, Model
from tensorflow.python.keras.utils import to_categorical

if len(sys.argv) > 1:
    MODEL_FILE = sys.argv[1]
else:
    MODEL_FILE = "cnn_lstm.hd5"
print("model file will save in {}".format(MODEL_FILE))


def create_model(num_frames=30, num_classes=27, batch_size=10):
    inputs = Input(shape=(num_frames, 112, 112, 3), batch_size=batch_size)  # Not quite sure this line
    # conv layer1
    conv1 = Convolution3D(64, kernel_size=(3, 3, 3), strides=(1, 1, 1), padding="same")(inputs)
    norm1 = BatchNormalization()(conv1)
    act1 = Activation('relu')(norm1)
    pol1 = MaxPool3D(pool_size=(1, 2, 2), strides=(1, 2, 2))(act1)
    # conv layer2
    conv2 = Convolution3D(128, kernel_size=(3, 3, 3), strides=(1, 1, 1), padding='same')(pol1)
    norm2 = BatchNormalization()(conv2)
    act2 = Activation('relu')(norm2)
    pol2 = MaxPool3D(pool_size=(2, 2, 2), strides=(2, 2, 2))(act2)
    # conv layer3
    conv3 = Convolution3D(256, kernel_size=(3, 3, 3), strides=(1, 1, 1), padding='same')(pol2)
    # conv layer4
    conv4 = Convolution3D(256, kernel_size=(3, 3, 3), strides=(1, 1, 1), padding='same')(conv3)
    norm3 = BatchNormalization()(conv4)
    act3 = Activation('relu')(norm3)
    pre_output_temp = ConvLSTM2D(256, kernel_size=3,
                                 strides=(1, 1),
                                 padding='same',
                                 return_sequences=True,
                                 stateful=True)(act3)
    pre_output = ConvLSTM2D(256, kernel_size=3,
                            strides=(1, 1),
                            padding='same',
                            batch_input_shape=(None, 18, 1),
                            stateful=True)(pre_output_temp)
    # SPP Layer
    spp1 = MaxPool2D(pool_size=(28, 28), strides=(28, 28))(pre_output)
    spp1 = Flatten()(spp1)

    spp2 = MaxPool2D(pool_size=(14, 14), strides=(14, 14))(pre_output)
    spp2 = Flatten()(spp2)

    spp4 = MaxPool2D(pool_size=(7, 7), strides=(7, 4))(pre_output)
    spp4 = Flatten()(spp4)

    spp7 = MaxPool2D(pool_size=(4, 4), strides=(4, 4))(pre_output)
    spp7 = Flatten()(spp7)

    merge = concatenate([spp1, spp2, spp4, spp7], name="Concat")

    # final_model = Sequential()
    # final_model.add(merge)
    # FC Layer
    classes = Dense(num_classes, activation='softmax')(merge)
    final_model = Model(inputs=[inputs, ], outputs=classes)
    # final_model.add(Dense(classes, activation='softmax'))

    return final_model


def load_existing(model_file):
    model = load_model((model_file))
    return model


class VideoDataset:

    def __init__(self, train_path, train_y, test_path, test_y,
                 target_size=(112, 112), num_frames=30, num_class=27):
        self.data_num = 0
        self.train_path = train_path
        self.train_y = train_y

        self.test_path = test_path
        self.test_y = test_y

        self.target_size = target_size
        self.num_frame = num_frames
        self.num_class = num_class

        self.label_list = None  # a list of hand gesture
        self.label_dict = dict()
        self.ground_truth_list = self.__load_ground_truth()

    def __load_ground_truth(self):
        with open("dataset/jester-v1-labels.csv", "r") as flabel:
            self.label_list = flabel.readlines()
            for i, label in enumerate(self.label_list):
                self.label_dict[label] = i
        ground_truth_list = np.zeros(150000, dtype=np.int)
        with open(self.train_y, "r") as fdata:
            info = fdata.readlines()
            self.data_num += len(info)
            for i, datalabel in enumerate(info):
                data = datalabel.split(";")
                ground_truth_list[int(data[0])] = self.label_dict[data[1]]
        with open(self.test_y, "r") as fdata:
            info = fdata.readlines()
            self.data_num = +len(info)
            for i, datalabel in enumerate(info):
                data = datalabel.split(";")
                ground_truth_list[int(data[0])] = self.label_dict[data[1]]
        return ground_truth_list

    def get_trainset(self, batch_size):
        return self.get_data_Iter(self.train_path, batch_size)

    def get_testset(self, batch_size):
        return self.get_data_Iter(self.test_path, batch_size)

    def get_data_Iter(self, dataset_path, batch_size):
        file_list = os.listdir(dataset_path)
        while True:
            file_list = list(set(file_list))
            for video_index in range(0, len(file_list), batch_size):
                if video_index + batch_size > len(file_list):
                    break
                X, Y = [], []
                for i in range(batch_size):
                    path = os.path.join(dataset_path, file_list[video_index+i])
                    X.append(self.get_video_set(path))
                    Ylabel = self.ground_truth_list[int(file_list[video_index+i])]
                    Y.append(to_categorical(Ylabel, self.num_class))
                yield np.array(X), np.array(Y)

    def get_video_set(self, path):
        img_list = []
        last = None
        img_paths = os.listdir(path)
        img_paths.sort()
        for img_path in os.listdir(path):
            img_abs_path = os.path.join(path, img_path)
            last = np.array(Image.open(img_abs_path).resize(self.target_size))
            img_list.append(last)
        while len(img_list) < self.num_frame:
            img_list.append(last)
        if len(img_list) > self.num_frame:
            img_list = img_list[0:self.num_frame]
        return np.array(img_list)


def train(model_file, num_frames=30, num_class=27, batch_size=1, num_epochs=28, save_period=1):
    if os.path.exists(model_file):
        print('\n*** existing model found at {}. Loading. ***\n\n'.format(model_file))
        model = load_existing(model_file)
    else:
        print("\n*** Creating new model ***\n\n")
        model = create_model(num_frames=num_frames, num_classes=num_class, batch_size=batch_size)

    model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

    save_model = ModelCheckpoint(model_file, period=save_period)
    stop_model = EarlyStopping(min_delta=0.001, patience=10)

    dataset = VideoDataset("~/20bn-jester-v1", "dataset/jester-v1-train.csv",
                           "~/20bn-jester-v1_val", 'dataset/jester-v1-validation.csv',
                           num_frames=num_frames, num_class=num_class)

    train_generator = dataset.get_trainset(batch_size=batch_size)
    test_generator = dataset.get_testset(batch_size=batch_size)

    # for train_x, train_y in test_generator:
    #     pass
    model.fit_generator(
        train_generator,
        steps_per_epoch=500,
        epochs=num_epochs,
        callbacks=[save_model, stop_model],
        validation_data=test_generator,
        validation_steps=10
    )
    print("Done training, Now evaluating.")


if __name__ == '__main__':
    train(MODEL_FILE, num_class=27, num_frames=35, batch_size=15, num_epochs=20)
