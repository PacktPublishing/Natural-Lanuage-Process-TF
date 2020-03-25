import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, SimpleRNN
from tensorflow.keras.utils import to_categorical
import sys
sys.path.append("../Chapter_03")
from Chapter_03_utils import IntEncoder, terms2ints, ints2terms

# convert into dataset matrix
def convertToMatrix(X_in, y_in, rnn_step):
    X_out, y_out =[], []
    for i in range(len(X_in)-rnn_step):
        dim=i+step
        X_out.append(X_in[i:dim,])
        y_out.append(y_in[dim,])
    return np.array(X_out), np.array(y_out)

#Generating sample dataset

#For this tutorial, we'll generate simple sequence data.
step = 2
train_in = [1, 1, 1, 1, 2, 1, 1, 2, 1, 1, 2, 1, 1, 1, 2, 1]
train_in.extend(train_in[-1 - step:])
train_in = np.reshape(train_in, (len(train_in), 1))
train_out = ['a', 'a', 'a', 'a', 'a', 'b', 'a', 'a', 'b', 'a', 'a', 'b', 'a', 'a', 'a', 'b']
train_out.extend(train_out[-1-step:])
from nltk import FreqDist
fd = FreqDist(train_out)
d = terms2ints(fd)
NUM_LABELS = len(list(d.keys()))
d2 = ints2terms(d)
enc = IntEncoder(d, d2)
train_out_ints = enc.encode(train_out)
train_out_enc = to_categorical(train_out_ints)

test_in = [1, 2, 1, 1]
test_in.extend(test_in[-1-step:])
test_in = np.reshape(test_in, (len(test_in), 1))
test_out = ['a', 'a', 'b', 'a']
test_out.extend(test_out[-1-step:])
test_out_ints = enc.encode(test_out)
test_out_enc = to_categorical(test_out_ints)


trainX,trainY =convertToMatrix(train_in, train_out_enc, step)
testX,testY =convertToMatrix(test_in, test_out_enc, step)

#Finally, we'll reshape trainX and testX to fit with the Keras model. RNN model requires three-dimensional input data. You can see the shape of testX below.

trainX = np.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
testX = np.reshape(testX, (testX.shape[0], 1, testX.shape[1]))

trainX.shape

# SimpleRNN model
model = Sequential()
model.add(SimpleRNN(units=8, input_shape=(1,step), activation="relu"))
model.add(Dense(8, activation="relu"))
model.add(Dense(NUM_LABELS, activation="softmax"))
model.compile(loss='categorical_crossentropy', optimizer='rmsprop')
model.summary()

model.fit(trainX,trainY, epochs=50, batch_size=4, verbose=2)

testPredict= model.predict_classes(testX)

#Next, we'll check the loss

trainScore = model.evaluate(trainX, trainY, verbose=0)
print(trainScore)

#Finally, we check the result in a plot. A vertical line in a plot identifies a splitting point between the training and the test part.

index = df.index.values
plt.plot(index,df)
plt.plot(index,predicted)
plt.axvline(df.index[Tp], c="r")
plt.show()
