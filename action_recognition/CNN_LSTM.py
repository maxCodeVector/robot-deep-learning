import os
import sys

import numpy as np
from tensorflow.python.keras import Input
from tensorflow.python.keras.layers import Dense, ConvLSTM2D, MaxPool3D, Activation, Convolution3D
from tensorflow.python.keras.layers import BatchNormalization, MaxPool2D, Concatenate, Flatten
from tensorflow.python.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.python.keras.models import load_model, Sequential, Model
from PIL import Image


if len(sys.argv) > 1:
    MODEL_FILE = sys.argv[1]
else:
    MODEL_FILE = "cnn_lstm.hd5"
print("model file will save in {}".format(MODEL_FILE))


def create_model(batch_size=1):
    inputs = Input(shape=(36, 112, 112, 3), batch_size=batch_size)  # Not quite sure this line
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

    merge = Concatenate([spp1, spp2, spp4, spp7], name="Concat")

    final_model = Sequential()
    final_model.add(merge)
    # FC Layer
    # classes = Dense(27, activation='softmax')(merge)
    # final_model = Model(inputs=[input_video, ], outputs=classes)
    final_model.add(Dense(27, activation='softmax'))

    return final_model


def load_existing(model_file):
    model = load_model((model_file))
    return model


class VideoDataset:

    def __init__(self, path, ground_path):
        self.data_num = 0
        self.path = path
        self.ground_path = ground_path
        self.label_dict = dict()
        self.label_list = []  # a list of hand gesture
        self.ground_truth_list = self.__load_ground_truth()

    def __load_ground_truth(self):
        with open("dataset/jester-v1-labels.csv", "r") as flabel:
            self.label_list = flabel.readlines()
            for i, label in enumerate(self.label_list):
                self.label_dict[label] = i
        ground_truth_list = np.zeros(150000, dtype=np.int)
        with open(self.ground_path, "r") as fdata:
            info = fdata.readlines()
            self.data_num = len(info)
            for i, datalabel in enumerate(info):
                data = datalabel.split(";")
                ground_truth_list[int(data[0])] = self.label_dict[data[1]]
        return ground_truth_list

    def get_trainset(self):
        for video_folder in os.listdir(self.path):
            path = os.path.join(self.path, video_folder)
            yield VideoDataset.get_video_set(path), self.ground_truth_list[int(video_folder)]

    @staticmethod
    def get_video_set(path):
        img_list = []
        for img_path in os.listdir(path):
            img_abs_path = os.path.join(path, img_path)
            img_list.append(np.array(Image.open(img_abs_path)))
        return np.array(img_list)

    def get_testset(self):
        return None, None


def train(model_file, dataset, step=32, num_epochs=28, save_period=1):
    if os.path.exists(model_file):
        print('\n*** existing model found at {}. Loading. ***\n\n'.format(model_file))
        model = load_existing(model_file)
    else:
        print("\n*** Creating new model ***\n\n")
        model = create_model()

    model.compile(optimizer='rmsprop', loss='categorical_crossentropy')

    save_model = ModelCheckpoint(model_file, period=save_period)
    stop_model = EarlyStopping(min_delta=0.001, patience=10)

    exit(1)
    train_gene = dataset.get_trainset()
    test_x, test_y = dataset.get_testset()

    for i in range(num_epochs):
        print("Iter: %d of %d" % (i, num_epochs))
        for train_x, train_y in train_gene:
            model.fit(train_x, train_y,
                      epochs=1, batch_size=20,
                      validation_data=(test_x, test_y),
                      callbacks=[save_model, stop_model],
                      verbose=2, shuffle=False)
        model.reset_states()

    # model.fit(x=train_x, y=train_y,
    #           shuffle=True,
    #           batch_size=60,
    #           epochs=epochs,
    #           validation_data=(test_x, test_y),
    #           callbacks=[save_model, stop_model])
    print("Done training, Now evaluating.")
    loss, acc = model.evaluate(x=test_x, y=test_y)

    print("Final loss: %3.2f Final accuracy: %3.2f" % (loss, acc))


if __name__ == '__main__':
    dataset = VideoDataset("/home/hya/Downloads/20bn-jester-v1", "dataset/jester-v1-train.csv")
    # for x, y in dataset.get_trainset():
    #     print(x, y)

    train(MODEL_FILE, dataset)
