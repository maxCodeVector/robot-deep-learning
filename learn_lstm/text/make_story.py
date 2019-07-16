from keras.models import load_model
from os.path import isfile
from read_corpus import ReadCorpus
import numpy as np

batch_size = 30
look_back = 20
skip = 1
hidden_size = 250
num_epochs = 100
demo_file = "./sherlock.demo.hd5"

if isfile(demo_file):
    model = load_model(demo_file)

    rc = ReadCorpus()
    rc.set_params(batch_size = batch_size, look_back = look_back, skip = skip)
    train_gen = rc.generate_train()

    starter = ''
    numwords = 1000

    wc = 0
    for in_array, _ in train_gen:


        if starter == '':
            # Generate the words that are in in_array as a starting point
            for tok in in_array[0]:
                starter += rc.token_to_words([int(tok)]) + ' '

            starter = starter.upper() + ": "


        if wc >= numwords:
            break;

        wc += batch_size

        print("Generating word %d of %d." % (wc, numwords))

        predict = model.predict(in_array)
        print("Predict shape: ", predict.shape)

        for i in range(batch_size):
            predict_word = np.argmax(predict[i, look_back - 1, :])
            starter = starter + ' ' + rc.token_to_words([predict_word])

    print("Final statement: %s. " % starter)

else:
    print("Model file %s missing. Please run embedding.py to create." % demo_file)

