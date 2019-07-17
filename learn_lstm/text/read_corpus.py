from tensorflow.python.keras.preprocessing.text import Tokenizer
from tensorflow.python.keras.utils import to_categorical
from math import log
import numpy as np
import scipy
from os import listdir
from os.path import isfile, join

EPS = 1e-10

class ReadCorpus:
    def __init__(self, trainDir='corpus/Training', testDir='corpus/Testing', max_vocab_size=10000):
        trainList = [join(trainDir, f) for f in listdir(trainDir) if isfile(join(trainDir, f))]
        testList = [join(testDir, f) for f in listdir(testDir) if isfile(join(testDir, f))]

        self.trainCorpus = []
        self.testCorpus = []

        for fname in trainList:
            with open(fname) as f:
                x = f.read()
                self.trainCorpus.append(x)

        for fname in testList:
            with open(fname) as f:
                x = f.read()
                self.testCorpus.append(x)

        self.t = Tokenizer(num_words=max_vocab_size, lower = True, oov_token='OOV')
        self.t.fit_on_texts(self.trainCorpus)
        self.output_size = len(self.t.word_counts) + 2
        self.revDict = []

        self.P = None
        self.T = None
        self.back = None
        self.ndx = 0

        for w in self.t.word_index:
            self.revDict.append(w)

        self._bigrams_ = None

    def tokenize_one(self, text):
        return self.t.texts_to_sequences([text])

    def tokenize_texts(self, texts):
        return self.t.texts_to_sequences(texts)

    def get_one_count(self, text):
        return self.t.texts_to_matrix([text])

    def get_texts_counts(self, texts):
        return self.t.texts_to_matrix(texts)

    def token_to_words(self, tokens):
        ret = ''

        for tok in tokens:
            if (tok - 1) >= 0 and tok < len(self.revDict):
                if ret == '':
                    ret = ret + self.revDict[tok - 1]
                else:
                    ret = ret + ' ' + self.revDict[tok - 1]

        return ret

    def get_train_corpus(self, index = -1):
        if index < 0:
            return self.trainCorpus
        else:
            return self.trainCorpus[index]

    def get_test_corpus(self, index = -1):
        if index < 0:
            return self.testCorpus
        else:
            return self.testCorpus[index]

    def set_params(self, batch_size = 1, look_back = 1, skip = 1, loop = False):
        self.batch_size = batch_size
        self.look_back = look_back
        self.skip = skip
        self.loop = loop


    def generate(self, batch_size, look_back, skip, numDocs, output_size, toks, loop):
        currDoc = 0
        currNdx = 0

        vocab_size = output_size
        retX = np.zeros((batch_size, look_back))
        retY = np.zeros((batch_size, look_back, vocab_size))

        # Length of current document
        currLen = len(toks[currDoc])

        exitFlag = False

        while currDoc < numDocs and not exitFlag:

            for i in range(batch_size):

                if currNdx + look_back >= currLen:
                    currDoc += 1
                    
                    if currDoc >= numDocs:
                        if loop:
                            currDoc = 0
                        else:
                            exitFlag = True
                            break

                    currNdx = 0
                    currLen = len(toks[currDoc])

                if not exitFlag:

                    retX[i][:] = toks[currDoc][currNdx:currNdx + look_back]

                    thetoks = toks[currDoc][currNdx + 1:currNdx + look_back + 1]

                    if vocab_size > 1:
                        retY[i][:][:] = to_categorical(thetoks, vocab_size)

                    currNdx += skip

            if not exitFlag:
                yield retX, retY


    def generate_train(self):
        toks = self.tokenize_texts(self.trainCorpus)
        return self.generate(self.batch_size, self.look_back, self.skip, self.t.document_count, self.output_size, toks, self.loop)


    def generate_test(self):
        toks = self.tokenize_texts(self.testCorpus)
        return self.generate(self.batch_size, self.look_back, self.skip, len(self.testCorpus), self.output_size, toks, self.loop)

    def vocab_size(self):
        return self.output_size

    def make_bigrams(self):

        print("Generating %d x %d bigram table. May take a LONG TIME." % (self.output_size, self.output_size))
        bigrams = np.zeros((self.output_size, self.output_size))
        
        toks = self.tokenize_texts(self.trainCorpus)
        gen = self.generate(batch_size = 1, look_back = 2, skip = 1, numDocs = self.t.document_count, output_size = self.output_size, toks = toks, loop = False)
        tcount = 0

        while True:
            try:
                v,_ = next(gen)
                w = int(v[0][0])
                wp = int(v[0][1])

                bigrams[w][wp] += 1
                tcount += 1

            except StopIteration:
                bigrams /= tcount
                break

        #self._bigrams_ = scipy.sparse.coo_matrix(bigrams)
        self._bigrams_ = bigrams + EPS
        print("Done. Found %d tokens." % tcount)

    def _backtrack_(self, path):
        outstr = self.token_to_words(path)
        return path, outstr

    def reset_viterbi(self):
        self.ndx = 0

    def viterbi(self, scores, start_token, seq_len = 5000):

        if self.ndx > seq_len:
            return backtrack_viterbi()

        # Generate bigrams if not yet done.
        if self._bigrams_ is None:
            self.make_bigrams()

        if self.P is None:
            self.P = np.zeros((self.output_size, seq_len))
            self.T = np.zeros((self.output_size, seq_len))
            self.back = np.zeros((self.output_size, seq_len))

        # Prevent domain errors
        scores += EPS


        if self.ndx == 0:
            for w in range(self.output_size):
                self.P[w][0] = -log(self._bigrams_[start_token][w]) * -log(scores[w])
                self.back[w][0] = start_token

            self.ndx += 1
        else:

            for w in range(self.output_size):
                self.P[w][self.ndx] = np.min(np.asarray([self.P[wp, self.ndx - 1] * -log(self._bigrams_[wp, w]) * -log(scores[w]) for wp in range(self.output_size)]))
                self.back[w][self.ndx] = np.argmin(np.asarray([self.P[wp, self.ndx - 1] * -log(self._bigrams_[wp, w]) for wp in range(self.output_size)]))

            self.ndx += 1
            print("Word %d" % self.ndx)

    def backtrack_viterbi(self):

        winner = np.argmax(self.P[:, self.ndx - 1])
        return self._backtrack_(self.back[winner])

