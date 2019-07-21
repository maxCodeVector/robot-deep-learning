import os
import sys

from tensorflow.python.keras import Input
from tensorflow.python.keras.layers import Dense, ConvLSTM2D, MaxPool3D, Activation, Convolution3D
from tensorflow.python.keras.layers import BatchNormalization, MaxPool2D, Concatenate, Flatten
from tensorflow.python.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.python.keras.models import load_model, Sequential, Model

if len(sys.argv) > 1:
    MODEL_FILE = sys.argv[1]
else:
    MODEL_FILE = "cnn_lstm.hd5"
print("model file will save in {}".format(MODEL_FILE))


def create_model():
    input_video = Input(shape=(36, 112, 112, 3))  # batch_size,image_num,weight,height,channel

    pre_model = Sequential()
    # conv layer1
    pre_model.add(Convolution3D(64,
                                kernel_size=(3, 3, 3),
                                input_shape=(36, 112, 112, 3),  # batch_size,image_num,weight,height,channel
                                strides=(1, 1, 1),
                                padding='same'))

    pre_model.add(BatchNormalization())
    pre_model.add(Activation('relu'))
    pre_model.add(MaxPool3D(pool_size=(1, 2, 2),
                            strides=(1, 2, 2)))
    # conv layer2
    pre_model.add(Convolution3D(128,
                                kernel_size=(3, 3, 3),
                                strides=(1, 1, 1),
                                padding='same'))
    pre_model.add(BatchNormalization())
    pre_model.add(Activation('relu'))
    pre_model.add(MaxPool3D(pool_size=(2, 2, 2),
                            strides=(2, 2, 2)))
    # conv layer3
    pre_model.add(Convolution3D(256, kernel_size=(3, 3, 3),
                                strides=(1, 1, 1),
                                padding='same'))
    # conv layer4
    pre_model.add(Convolution3D(256, kernel_size=(3, 3, 3),
                                strides=(1, 1, 1),
                                padding='same'))
    pre_model.add(BatchNormalization())
    pre_model.add(Activation('relu'))
    pre_model.summary()

    pre_output_test = pre_model(input_video)
    pre_output_test = ConvLSTM2D(256, kernel_size=3,
                             strides=(1, 1),
                             padding='same',
                             return_sequences=True,
                             stateful=True)(pre_output_test)
    pre_output = ConvLSTM2D(256, kernel_size=3,
                                 strides=(1, 1),
                                 padding='same',
                                 return_sequences=True,
                                 batch_input_shape=(None, 18, 1),
                                 stateful=True)(pre_output_test)

    # ConvLSTM
    # pre_model.add(ConvLSTM2D(256, kernel_size=3,
    #                          strides=(1, 1),
    #                          padding='same',
    #                          return_sequences=True,
    #                          batch_input_shape=(None, 18, 1),
    #                          stateful=True
    #                          ))
    # exit(-9)
    # pre_model.add(ConvLSTM2D(384,
    #                          kernel_size=(3, 3),
    #                          strides=(1, 1),
    #                          padding='same',
    #                          stateful=True
    #                          ))

    # pre_output = pre_model(input_video)
    # SPP Layer
    spp1 = MaxPool2D(pool_size=(28, 28), strides=(28, 28))(pre_output)
    spp1 = Flatten(spp1)

    spp2 = MaxPool2D(pool_size=(14, 14), strides=(14, 14))(pre_output)
    spp2 = Flatten(spp2)

    spp4 = MaxPool2D(pool_size=(7, 7), strides=(7, 4))(pre_output)
    spp4 = Flatten(spp4)

    spp7 = MaxPool2D(pool_size=(4, 4), strides=(4, 4))(pre_output)
    spp7 = Flatten(spp7)

    merge = Concatenate([spp1, spp2, spp4, spp7])

    # FC Layer
    classes = Dense(27, activation='softmax')(merge)
    final_model = Model(inputs=[input_video, ], outputs=classes)
    return final_model


def load_existing(model_file):
    model = load_model((model_file))
    return model


class VideoDataset:

    def __init__(self, path):
        self.path = path

    def get_trainset(self):

        return None, None

    def get_testset(self):
        return None, None


def train(model_file, dataset, step=32, num_epochs=28, save_period=1):
    if os.path.exists(model_file):
        print('\n*** existing model found at {}. Loading. ***\n\n'.format(model_file))
        model = load_existing(model_file)
    else:
        print("\n*** Creating new model ***\n\n")
        model = create_model()

    model.summary()
    model.compile(optimizer='rmsprop', loss='categorical_crossentropy')

    save_model = ModelCheckpoint(model_file, period=save_period)
    stop_model = EarlyStopping(min_delta=0.001, patience=10)

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
    dataset = VideoDataset("www")
    train(MODEL_FILE, dataset)
