#Learn from a text
#Predict next char
from os import path
import csv
import numpy as np
import random
import sys
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation
from tensorflow.keras.layers import LSTM, RNN, SimpleRNNCell, SimpleRNN
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.callbacks import Callback

'''
Data downloaded from here:
https://www.usgs.gov/core-science-systems/ngp/board-on-geographic-names/download-gnis-data
'''

def loadData(f, fc):
    '''
    open and read GNIS file, filter features according to fc, return as list of names
    '''
    dataDir = '.\\data'
    filepath = path.normpath(dataDir + '\\'+ f)

    results = []

    with open (filepath, 'rt') as fh:
        reader = csv.DictReader(fh, delimiter='|')

        for row in reader:
            if (row['FEATURE_CLASS'] == fc):
                results.append(row['FEATURE_NAME'])

    return results



def sample(preds, temperature=1.0):
    # helper function to sample an index from a probability array
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)

class SampleResult(Callback):

    def on_epoch_end(self, epoch, logs={}):

        start_index = random.randint(0, len(mashedUpSchools) - maxlen - 1)

        for diversity in [0.2, 0.5, 1.0, 1.2]:
            generated = ''
            sentence = mashedUpSchools[start_index: start_index + maxlen]
            generated += sentence
            print()
            print('----- Generating with diversity',
                  diversity, 'seed: "' + sentence + '"')
            sys.stdout.write(generated)

            for i in range(100):
                x = np.zeros((1, maxlen, len(chars)))
                for t, char in enumerate(sentence):
                    x[0, t, char_indices[char]] = 1.

                preds = self.model.predict(x, verbose=0)[0]
                next_index = sample(preds, diversity)
                next_char = indices_char[next_index]

                generated += next_char
                sentence = sentence[1:] + next_char

                sys.stdout.write(next_char)
                sys.stdout.flush()
        print('\n\n')


if __name__ == '__main__':
    #Load an array of place names
    filename = 'MD_Features_20200301.txt'
    schools = loadData(filename, 'School')
    mashedUpSchools = '|'.join(schools)

    #How many chars?
    chars = sorted(list(set(mashedUpSchools)))
    print('total chars: {}'.format(len(chars)))
    char_indices = dict((c, i) for i, c in enumerate(chars))
    indices_char = dict((i, c) for i, c in enumerate(chars))

    maxlen = 10
    step = 3

    trainingSamples = []
    next_chars = []
    for i in range(0, len(mashedUpSchools) - maxlen, step):
        trainingSamples.append(mashedUpSchools[i: i + maxlen])
        next_chars.append(mashedUpSchools[i + maxlen])

    X = np.zeros((len(trainingSamples), maxlen, len(chars)), dtype=np.bool)
    y = np.zeros((len(trainingSamples), len(chars)), dtype=np.bool)
    for i, school in enumerate(trainingSamples):

        for t, char in enumerate(school):
            X[i, t, char_indices[char]] = True
        y[i, char_indices[next_chars[i]]] = True

    print('Size of X: {:.2f} MB'.format(X.nbytes/1024/1024))
    print('Size of y: {:.2f} MB'.format(y.nbytes/1024/1024))

    nb_units = 64

    model = Sequential()

    model.add(SimpleRNN(nb_units, activation='tanh', input_shape=(maxlen, len(chars))))

    model.add(Dense(units=len(chars)))
    model.add(Activation('softmax'))

    optimizer = RMSprop(lr=0.01)
    model.compile(loss='categorical_crossentropy', optimizer=optimizer)

    print(model.summary())

    sample_callback = SampleResult()

    history = model.fit(X, y,
                        epochs=10,
                        batch_size=512,
                        verbose=2,
                        callbacks=[sample_callback])
