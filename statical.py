import sklearn
import pandas as pd
import numpy as np



import scipy.stats as stats
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import SGDClassifier, LinearRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.datasets import fetch_20newsgroups

from sklearn.feature_extraction.text import TfidfTransformer

_20_train = fetch_20newsgroups(subset="train", shuffle=True)
_20_test = fetch_20newsgroups(subset='test', shuffle=True)
tfidf_transformer = TfidfTransformer()

from sklearn.pipeline import Pipeline


def pipeLine_NB():
    text_clf = Pipeline([
        ('vect', CountVectorizer(stop_words='english')),
        ('tfidf', TfidfTransformer()),
        ('clf', MultinomialNB()),
    ])
    text_clf.fit(_20_train.data, _20_train.target)
    predicted = text_clf.predict(_20_test.data)
    print("Accuracy (Pipelined NB): {}."
          .format(np.mean(predicted == _20_test.target)))


def pipline_SGD():
    text_svm = Pipeline([
        ('vect', CountVectorizer(stop_words='english')),
        ('tfidf', TfidfTransformer()),
        ('svm', SGDClassifier(loss='hinge', penalty='l2', alpha=1e-3,
                              max_iter=10, random_state=42)), ])
    text_svm.fit(_20_train.data, _20_train.target)
    predicted = text_svm.predict(_20_test.data)
    print("Accuracy(Pipelined SVM): {}."
          .format(np.mean(predicted == _20_test.target)))


def tfidf():
    _20_train = fetch_20newsgroups(subset="train", shuffle=True)
    _20_test = fetch_20newsgroups(subset='test', shuffle=True)

    count_vect = CountVectorizer()
    X_train_counts = count_vect.fit_transform(_20_train.data)
    X_test_counts = count_vect.transform(_20_test.data)

    X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)

    tfidf_clf = MultinomialNB().fit(X_train_tfidf, _20_train.target)

    X_test_tfidf = tfidf_transformer.transform(X_test_counts)

    predicted = tfidf_clf.predict(X_test_tfidf)
    print('Accuracy (tfidf) is {}'
          .format(np.mean(predicted == _20_test.target)))


def del_stop():
    sw_count_vect = CountVectorizer(stop_words='english')

    X_train_counts = sw_count_vect.fit_transform(_20_train.data)
    X_test_counts = sw_count_vect.transform(_20_test.data)
    X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
    X_test_tfidf = tfidf_transformer.transform(X_test_counts)
    tfidf_clf = MultinomialNB().fit(X_train_tfidf, _20_train.target)
    predicted = tfidf_clf.predict(X_test_tfidf)
    print('Accuracy(tfidf – stopwords): {}'
          .format(np.mean(predicted == _20_test.target)))


def using_padas_file():
    sales = pd.read_csv('sales.csv', sep=',', engine='python')
    print("Correlation: {}".
          format(sales['GDPChange'].corr(sales['Sales'])))
    X = sales['Change in GDP'].values.reshape(-1, 1)
    Y = sales['Sales'].values.reshape(-1, 1)
    lm = LinearRegression()
    lm.fit(X, Y)
    print("Coeff: {} Intercept: {}".format(lm.coef_, lm.intercept_))
    print("If we have 2.5% growth, sales will be {} units.”"
          .format(lm.predict(2.5)))


if __name__ == '__main__':
    # tfidf()
    # del_stop()
    # pipeLine_NB()
    # pipline_SGD()
    using_padas_file()
