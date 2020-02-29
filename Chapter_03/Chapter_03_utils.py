'''
Utility Functions for Chapter_03.ipynb
Make sure this file is stored in the same folder where your Chapter_03.ipynb is located
'''
#-----------------------------------------------------------------------------------#
# Imports                                                                           #
#-----------------------------------------------------------------------------------#
import re
import collections
import numpy as np

#-----------------------------------------------------------------------------------#
# Globals                                                                           #
#-----------------------------------------------------------------------------------#

punctPat = re.compile(r'^[\'\".,;:()!?-]+$')

def isPunctuation(token):
    '''
    True if input string is only punctuation
    '''
    if punctPat.match(token):
        return True
    return False

def terms2ints(termsList):
    '''
    Return dictionary of keys = terms, values = integer codes
    Include '<UNKNOWN>' at beginning as first value
    '''
    maxWords = len(list(termsList))
    result =  {t:(i+1) for (i,t) in enumerate(termsList)}
    result['<UNKNOWN>'] = 0
    return result

def ints2terms(termsDict):
    '''
    Flip the term:int key value pairs, return flipped result
    '''
    return {v:k for (k, v) in termsDict.items()}

class IntEncoder():
    '''
    Class to encode lists of terms as integers.
    We pass the terms-integers codes dictionary, and the reverse (integer codes to terms dictionary)
    as arguments to the init function.
    '''

    def __init__(self, termsDict, reverseTermsDict):
        '''
        Instantiate the encoder object
        '''
        assert isinstance(termsDict, dict)
        assert isinstance(reverseTermsDict, dict)

        self.termsDict = termsDict
        self.reverseTermsDict =  reverseTermsDict
        self.vocabLen = len(list(self.termsDict.keys()))

    def lookupCode(self, term):
        '''
        Lookup the integer code for a term in the dict, or return Unknown
        '''
        if (term in self.termsDict.keys()):
            return self.termsDict[term]
        else:
            return self.termsDict['<UNKNOWN>']

    def lookupTerm(self, code):
        '''
        Lookup the term corresponding to an integer code
        '''

        if (code in self.reverseTermsDict.keys()):
            return self.reverseTermsDict[code]
        else:
            return('<UNKNOWN>')

    def encode(self, termsList):
        '''
        Return a list of integer codes corresponding to the words in the input termsList
        '''

        return [self.lookupCode(t) for t in termsList]

def build_dataset(words, n_words):
    """
    Process raw inputs into a dataset.
    Borrowed from https://adventuresinmachinelearning.com/word2vec-keras-tutorial/
    """
    count = [['UNK', -1]]
    count.extend(collections.Counter(words).most_common(n_words - 1))
    dictionary = dict()
    for word, _ in count:
        dictionary[word] = len(dictionary)
    data = list()
    unk_count = 0
    for word in words:
        if word in dictionary:
            index = dictionary[word]
        else:
            index = 0  # dictionary['UNK']
            unk_count += 1
        data.append(index)
    count[0][1] = unk_count
    reversed_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
    return data, count, dictionary, reversed_dictionary

class SimilarityCallback:
    '''
    Examine similarity of a set of frequent words during training
    Modified from https://adventuresinmachinelearning.com/word2vec-keras-tutorial/
    '''
    def run_sim(self, evalSetSize, evalExamples, reverse_dictionary, vocabSize, similarityModel):
        for i in range(evalSetSize):
            evalWord = reverse_dictionary[evalExamples[i]]
            top_k = 8  # number of nearest neighbors
            sim = self._get_sim(evalExamples[i], vocabSize, similarityModel)
            nearest = (-sim).argsort()[1:top_k + 1]
            log_str = 'Nearest to %s:' % evalWord
            for k in range(top_k):
                closeWord = reverse_dictionary[nearest[k]]
                log_str = '%s %s,' % (log_str, closeWord)
            print(log_str)

    def probe_word(self, testWord, term_dictionary, reverse_dictionary, vocabSize, similarityModel):
        '''
        Find the most similar words to the probe word at a point in the training cycle
        '''
        idx = term_dictionary[testWord]
        top_k = 8
        sim = self._get_sim(idx, vocabSize, similarityModel)
        nearest = (-sim).argsort()[1:top_k + 1]
        log_str = 'Nearest to %s:' % testWord
        for k in range(top_k):
            close_word = reverse_dictionary[nearest[k]]
            log_str = '%s %s,' % (log_str, close_word)
        print(log_str)

    @staticmethod
    def _get_sim(eval_word_idx, vocabSize, similarityModel):
        sim = np.zeros((vocabSize,))
        in_arr1 = np.zeros((1,))
        in_arr2 = np.zeros((1,))
        in_arr1[0,] = eval_word_idx
        for i in range(vocabSize):
            in_arr2[0,] = i
            out = similarityModel.predict_on_batch([in_arr1, in_arr2])
            sim[i] = out
        return sim

def is_Sublist(l, s):
    sub_set = False
    if s == []:
        sub_set = True
    elif s == l:
        sub_set = True
    elif len(s) > len(l):
        sub_set = False
    else:
        for i in range(len(l)):
            if l[i] == s[0]:
                n = 1
                while (n < len(s)) and (l[i+n] == s[n]):
                    n += 1

                if n == len(s):
                    sub_set = True

    return sub_set
