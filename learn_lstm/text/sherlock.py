
from read_corpus import ReadCorpus
from numpy import asarray
from numpy import array
import numpy as np
from tensorflow.python.keras.models import Sequential, load_model
from tensorflow.python.keras.layers import Dense
from tensorflow.python.keras.layers import Flatten
from tensorflow.python.keras.layers import Embedding
from tensorflow.python.keras.layers import CuDNNLSTM
from tensorflow.python.keras.layers import Dropout
from tensorflow.python.keras.layers import Activation
from tensorflow.python.keras.layers import TimeDistributed
from tensorflow.python.keras.callbacks import ModelCheckpoint
from os.path import isfile

batch_size = 30
look_back = 20
skip = 1
hidden_size = 250
num_epochs = 100
save_file = "./sherlock.hd5"

TEST_CYCLES = 2     # Number of cycles between test prints
TEST_WORDS = 1000   # Number of words to generate for testing

# Read in the corpus
rc = ReadCorpus()
print("Number of words: %d." % (rc.vocab_size()))

vocab_size = rc.vocab_size()

# Get generators for counting

rc.set_params(batch_size = batch_size, look_back = look_back, skip = skip, loop = False)
tr_gen = rc.generate_train()
te_gen = rc.generate_test()

print("Counting training samples")
for tr_count, __ in enumerate(tr_gen):
    pass

print("Counting testing samples")
for te_count, _ in enumerate(te_gen):
    pass

print("DONE!")

### WRITE THE GIVEN CODE BELOW ###

save_cp = ModelCheckpoint(save_file)

if isfile(save_file):
    print("Loading existing model.")
    model = load_model(save_file)
else:
    print("Creating new model.")
    model = Sequential()
    model.add(Embedding(vocab_size, hidden_size, batch_input_shape = (batch_size, look_back)))
    model.add(CuDNNLSTM(hidden_size, return_sequences = True))
    model.add(CuDNNLSTM(hidden_size, return_sequences = True))
    model.add(Dropout(0.5))
    model.add(TimeDistributed(Dense(vocab_size)))
    model.add(Activation('softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics = ['categorical_accuracy'])

### WRITE THE GIVEN CODE ABOVE

for i in range(num_epochs):

    # Output story every 5 epochs
    if (i % TEST_CYCLES) == 0:
        rc.set_params(batch_size = batch_size, look_back = look_back, skip = skip, loop = False)

        # Get the first string
        test_gen = rc.generate_train()

        start, _ = next(test_gen)

        str = ''
        for tok in start[0]:
            str += rc.token_to_words([int(tok)]) + ' '

        str = str.upper() + ": "
        # Recreate the generator
        test_gen = rc.generate_train()

        wc = 0

        for in_array, _ in test_gen:
            if wc >= TEST_WORDS:
                break

            wc += batch_size

            predict = model.predict(in_array)
            for j in range(batch_size):
                predict_word = np.argmax(predict[j, look_back - 1, :])
                str += " " + rc.token_to_words([predict_word])
                
        print("Epoch %d. Story: %s." % (i, str)) 

    print("EPOCH %d of %d." % (i, num_epochs))
    rc.set_params(batch_size = batch_size, look_back = look_back, skip = skip, loop = True)
    model.fit_generator(rc.generate_train(), tr_count, 1, validation_data = rc.generate_test(), callbacks = [save_cp], 
    validation_steps = tr_count)
    model.reset_states()

        
        
