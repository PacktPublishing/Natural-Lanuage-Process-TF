#Part of speech tagging with RNNs

from nltk.corpus import brown
import gensim
import numpy as np
import sys
sys.path.append('../Chapter_03')
from itertools import islice
from nltk import FreqDist
from Chapter_03_utils import IntEncoder, terms2ints, ints2terms
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Embedding, InputLayer, Dense, SimpleRNN, TimeDistributed
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import to_categorical

EMBEDDING_DIM = 300
MAX_SEQUENCE_LENGTH = 180
TRAINING_PROPORTION = .6
VALIDATION_PROPORTION = .2
VOCAB_LENGTH = 1000
RNN_STEP = 5

def tripleSplitList(inputList, firstCut, secondCut):
    '''
    Split a list into three sublists, using the first and second proportion cutpoints
    '''
    if (firstCut + secondCut) > 1:
        sys.stderr.write("Arguments to tripleSplitList must be between 0 and 1")
        return(None, None, None)
    elif (firstCut + secondCut) < 0:
        sys.stderr.write("Arguments to tripleSplitList must be between 0 and 1")
        return(None, None, None)
    else:
        inputLen = len(inputList)
        a = inputList[:(int(inputLen * firstCut))]
        b = inputList[(int(inputLen * firstCut)):(int(inputLen * (firstCut + secondCut)))]
        c = inputList[(int(inputLen * (firstCut + secondCut))):]

        if (len(a) + len(b) + len(c)) != inputLen:
            sys.stderr.write("Warning: lengths of splits lists don't sum to original list.\n")
        return (a, b, c)

# convert into dataset matrix
def sentencesToSlices(X_in, y_in, rnn_step):
    '''
    Convert lists of sentences and labels to list of slices of consistent length, with single label
    '''
    X_out, y_out =[], []
    for sentence, tag_list in zip(X_in, y_in):
        for i in range(len(sentence)-rnn_step):
            dim=i+rnn_step
            X_out.append(sentence[i:dim])
            y_out.append(tag_list[dim])
    return np.array(X_out), np.array(y_out)

if __name__ == '__main__':

    sentences = [] #List of lists of terms
    tag_lists = [] #List of lists of tags
    for (i, tagged_sentence) in islice(enumerate(brown.tagged_sents(tagset='universal')), 10):
        words = [w.lower() for (w, t) in tagged_sentence]
        tags = [t for (w, t) in tagged_sentence]
        sentences.append(words)
        tag_lists.append(tags)
        if (i%1000 == 0):
            sys.stderr.write("Processing sentence {}\n".format(i))
    #To retrieve simpler tags, use (brown.tagged_sents(tagset='universal')))
    sys.stderr.write("Collecting unique tags...\n")
    tag_set = set([tag for taglist in tag_lists for tag in taglist])
    num_tags = len(tag_set)


    sys.stderr.write("Counting terms in tagged corpus...\n")
    brownTermCounts = FreqDist(w.lower() for s in sentences for w in s)
    #num_words = {lambda vfor k, v in brownTermCounts}

    '''
    print("Tagged corpus contains {} words in {} sentences with {} distinct tags.".format(num_words, len(sentences), num_tags))
    print("Tagset:")
    print("\n".join(tag_set))
    maxlen = 0
    longest = []
    for s in sentences:
        if len(s) > maxlen:
            maxlen = len(s)
            longest = s

    longest_sentence = maxlen
    print("Longest sentence length = {}, content = {}".format(maxlen, " ".join(longest)))
    '''

    #https://towardsdatascience.com/deep-learning-for-arabic-part-of-speech-tagging-810be7278353 arabic
    #https://easychair.org/publications/preprint_open/Lczp Nepali, rnn


    #Now we'll make a vocabulary dictionary with integer values from the Brown corpus, and a reverse dictionary too
    brownTerms = terms2ints([t for (t, i) in brownTermCounts.most_common(VOCAB_LENGTH)])
    brownReverseTerms = ints2terms(brownTerms)
    #We use these to build an integer encoder for terms in the Brown corpus
    wordEncoder = IntEncoder(brownTerms, brownReverseTerms)

    #Now we make a similar dict of tags and a reverse dict
    tagCodes = terms2ints(list(tag_set))
    tagInts = ints2terms(tagCodes)
    tagEncoder = IntEncoder(tagCodes, tagInts)

    del brownTermCounts #Free some memory

    encodedSentences = [wordEncoder.encode(s) for s in sentences]
    encodedTags = [tagEncoder.encode(t) for t in tag_lists]

    #Split into train and test and validation
    (encodedSentencesTrain, encodedSentencesTest, encodedSentencesVal) = tripleSplitList(encodedSentences, TRAINING_PROPORTION, VALIDATION_PROPORTION)
    (encodedTagsTrain, encodedTagsTest, encodedTagsVal) = tripleSplitList(encodedTags, TRAINING_PROPORTION, VALIDATION_PROPORTION)

    #Transform these into strings of RNN_STEP length plus a label
    (X_raw, y_raw) = sentencesToSlices(encodedSentencesTrain, encodedTagsTrain, RNN_STEP)
    print(X_raw.shape)
    print(y_raw.shape)

    del encodedTags #Free up some memory
    del encodedSentences

    #Encode tags
    y_train = np.zeros((y_raw.shape[0], (num_tags+1)))
    for i in range(y_train.shape[0]):
        y_train[i] = to_categorical(y_raw[i], num_classes = (num_tags + 1))

    #Load the file into a gensim model object
    #This step takes a few minutes to load...
    sys.stderr.write("Loading pre-trained embedding vectors from disk...\n")
    #word_vectors = gensim.models.KeyedVectors.load_word2vec_format('../Chapter_03/data/GoogleNews-vectors-negative300.bin.gz', binary=True)

    #Build a dictionary of words and integer codes
    sys.stderr.write("Building embedding terms dictionary and reverse dictionary")
    #embedTerms = {w:(i+1) for (i, w) in enumerate(word_vectors.wv.vocab) }
    #Build a reverse dictionary of integer codes and associated terms
    #embedInts = {(i+i):w for (i,w) in enumerate(word_vectors.wv.vocab)}
    # prepare embedding matrix
    sys.stderr.write("Preparing embeddings matrix...\n")
    embedding_matrix = np.zeros(((VOCAB_LENGTH + 1), EMBEDDING_DIM))
    '''
    i = 0
    for w in brownTerms.keys():
        if w in embedTerms.keys():
            embedding_vector = word_vectors.wv.get_vector(w)
            # words not found in embedding index will be all-zeros.
            embedding_matrix[i] = embedding_vector
        i += 1
    '''
    weights = [embedding_matrix]

    #del word_vectors
    #del embedTerms #Free some more memory

    sys.stderr.write("Building model...\n")

    model = Sequential()

    model.add(SimpleRNN(nb_units, activation='tanh', input_shape=(RNN_STEP, len(chars))))

    model.add(Dense(units=len(chars)))
    model.add(Activation('softmax'))

    optimizer = RMSprop(lr=0.01)
    model.compile(loss='categorical_crossentropy', optimizer=optimizer)
