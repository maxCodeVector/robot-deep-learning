import numpy as np
import pandas
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from constants import batch_size

class ReadTC:
    def __init__(self, corpus_file, input_len, vocab_size, train_percent):

        print("Reading data")
        dataframe = pandas.read_csv(corpus_file, engine = 'python')
        print("DONE")

        total_len = len(dataframe.values)
        train_len = int(total_len * train_percent)
        print(train_len)

        self._train_corpus_text_ = dataframe.values[:train_len]
        self._test_corpus_text_ = dataframe.values[train_len:]

        print("Using %3.2f%% of %d samples for training. This gives us %d training samples, %d testing samples."
        %(train_percent * 100.0, total_len, len(self._train_corpus_text_), len(self._test_corpus_text_)))

        fit_text = []

        for txt in self._train_corpus_text_:
            fit_text.append(txt[2].strip())

        # Fit the tokenizer
        self.tok = Tokenizer(num_words = vocab_size)
        self.tok.fit_on_texts(fit_text)
        self._train_corpus_tokens_ = pad_sequences(self.tok.texts_to_sequences(fit_text), maxlen = input_len,
        truncating = 'post')

        fit_text = []

        for txt in self._test_corpus_text_:
            fit_text.append(txt[2].strip())

        self._test_corpus_tokens_ = pad_sequences(self.tok.texts_to_sequences(fit_text), maxlen = input_len,
        truncating = 'post')


    def get_data(self, batch_size, loop, main_corpus, token_corpus):
        ret = []
        
        index = 0
            
        input_len = len(token_corpus[0])
        exitFlag = False

        while not exitFlag:
            retX=np.zeros((batch_size, input_len))
            retY=np.zeros((batch_size, 2))

            for i in range(batch_size):

                # Wrap around if we hit the end
                if index >= len(main_corpus):
                    if loop:
                        index = 0
                    else:
                        exitFlag = True
                        break;
            
                retX[i] = np.asarray(token_corpus[index])
                retY[i] = np.asarray([[1 - main_corpus[index][1], main_corpus[index][1]]])
                index += 1

            if not exitFlag:
                yield retX, retY 

    def get_train_data(self, batch_size = batch_size, loop = True):
        return self.get_data(batch_size, loop, self._train_corpus_text_, self._train_corpus_tokens_)

    def get_train_len(self):
        return len(self._train_corpus_tokens_)

    def get_test_len(self):
        return len(self._test_corpus_tokens_)

    def get_test_data(self, batch_size = batch_size, loop = True):
        return self.get_data(batch_size, loop, self._test_corpus_text_, self._test_corpus_tokens_)

    def tokenize(self, sent):
        if len(self._train_corpus_tokens_) > 0:
            input_len = len(self._train_corpus_tokens_[0]) 
        else:
            input_len = len(self._test_corpus_tokens_[0])

        output =  np.asarray(pad_sequences(self.tok.texts_to_sequences([sent]), 
        maxlen = input_len, truncating='post')[0])
        print(output)
        return output
