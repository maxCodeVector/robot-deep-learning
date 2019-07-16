import numpy
import pandas
import math
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, LSTM, CuDNNLSTM

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

batch_size = 6
look_back = 5
skip = 1
hidden_size = 128
num_epochs = 750

numpy.random.seed(7)

dataframe = pandas.read_csv(
        "airline-passengers.csv", usecols=[1], engine='python')
dataset = dataframe.values
dataset = dataset.astype("float32")

scaler = MinMaxScaler(feature_range=(0, 1))
dataset = scaler.fit_transform(dataset)

train_size = int(len(dataset)*0.67)
test_size = len(dataset) - train_size
train, test = dataset[0:train_size], dataset[train_size:]


def create_dataset(dataset, look_back=1):
    dataX, dataY = [], []

    for i in range(len(dataset)-look_back-1):
        a = dataset[i:(i+look_back), 0]
        dataX.append(a)
        dataY.append(dataset[i+look_back, 0])
    return numpy.array(dataX), numpy.array(dataY)


trainX, trainY = create_dataset(train, look_back)
testX, testY = create_dataset(test, look_back)

trainX = numpy.reshape(trainX, (trainX.shape[0], trainX.shape[1], 1))
testX = numpy.reshape(testX, (testX.shape[0], testX.shape[1], 1))


num_hidden = 8
model = Sequential()
model.add(CuDNNLSTM(hidden_size,batch_input_shape=(batch_size, look_back, 1), stateful=True, return_sequences=True))
model.add(CuDNNLSTM(hidden_size,batch_input_shape=(batch_size, look_back, 1), stateful=True))

model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')

for i in range(num_epochs):
    print("Iter: %d of %d" % (i, num_epochs))
    model.fit(trainX, trainY, epochs=1, batch_size=batch_size, verbose=2, shuffle=False)
    model.reset_states()

trainPredict = model.predict(trainX)
testPredict = model.predict(testX)

trainPredict = scaler.inverse_transform(trainPredict)
trainY = scaler.inverse_transform([trainY])
testPredict = scaler.inverse_transform(testPredict)
testY = scaler.inverse_transform([testY])

trainScore = math.sqrt(mean_squared_error(trainY[0], trainPredict[:, 0]))
print('Train Score: %.2f RMSE' % (trainScore))

testScore = math.sqrt(mean_squared_error(testY[0], testPredict[:, 0]))
print('Test Score: %.2f RMSE' % (testScore))





