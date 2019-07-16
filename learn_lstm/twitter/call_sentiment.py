import numpy as np
from tensorflow.python.keras.models import load_model
import read_tc
from constants import *

print("Loading model from %s." % filename)
model = load_model(filename)

tc = read_tc.ReadTC('train.csv', input_length, vocab_size, 1.0)

read_batch = np.zeros((batch_size, input_length))

quitFlag = False
while not quitFlag:
    print("Enter tweet: ", end="")
    mytweet = input().lower()

    quitFlag = (mytweet == 'quit')
    if not quitFlag:
        read_batch[0] = tc.tokenize(mytweet)
        pred = model.predict(read_batch)

        ndx = np.argmax(pred[0])
        print(pred[0], ndx)

        if ndx == 1:
            print("%s is Positive" % mytweet)
        else:
            print("%s is Negative" % mytweet)
