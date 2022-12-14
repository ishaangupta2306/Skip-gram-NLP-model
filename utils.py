import re
import numpy as np

class Utils:
    def __init__(self):
        self.matrix = self.construct_table()

    # Function to read the file and clean the corpus
    def read_file(self, file):
        file = open(file, 'r')
        return self.clean_text(file.read())

    # Function to clean the corpus and extract sentences
    def clean_text(self, text_corpus):
        ls = []
        par = re.sub(r"\([^)]*\)", "", text_corpus).split('\n')
        # Iterate over each paragraph
        for p in par:
            # If para does not exists, ignore
            if not p:
                pass
            else:
                # Vocabulary for each sentence
                sen = p.split('.')
                for i in range(len(sen)):
                    # Ignore if sentence does not exist
                    if not sen[i]:
                        pass
                    # Otherwise, pick the words
                    else:
                        # all words are in lower case for the vocabulary used in the model
                        tokens = re.sub(r"[!\"#$%&'()*+,-./:;<=>?@[\]^_`{|}~]+\ *", " ", sen[i]).lower().split()
                        ls.append(tokens)
        # List of cleaned corpus text words
        return ls

    def seperate_words_digits(self, array, vocabulary, words):
        size = 0
        for tokens in array:
            x = 0
            size += len(tokens)
            while x in range(len(tokens)):
                word = tokens[x]
                x += 1
                flag = word.isdigit()
                if word in vocabulary or flag:
                    if not flag:
                        vocabulary[word]['ocrnc'] += 1
                    else:
                        vocabulary['num']['ocrnc'] += 1
                else:
                    vocabulary[word]['ocrnc'] = 1
                    words.append(word)
        return vocabulary, words, size

    def find_words_with_given_max_occurence(self, words, occurence):
        ls = []
        for w in words:
            if not words[w]['ocrnc'] >= occurence: ls.append(w)
        return ls

    def sort_words(self, words, by):
        ls = []
        for w in words:
            ls.append((w, words[w][by]))
        ls.sort(key=lambda tup: tup[1], reverse=True)
        return ls

    def construct_table(self):
        matrix = []
        for i in range(1001):
            t = i - 500
            t *= 6
            a = np.exp(t / 500)
            matrix.append(a)
        return matrix
    
    # Activation function for the model
    def sigmoid_function(self, x, lm):
        t = 1
        if x > -6:
            pass
        else:
            x = -6
        if x < 6:
            pass
        else:
            x = 6
        x *= - (1/6)
        ind = int(x * lm + 500)
        t += self.matrix[ind]
        return np.reciprocal(t)

    # Calculates distance between two vectors
    def dist(self, a1, b1):
        dot_product12 = np.dot(a1, b1)
        dot_product11 = np.dot(a1, a1)
        dot_product22 = np.dot(b1, b1)
        num = dot_product12
        den = np.sqrt(dot_product11 * dot_product22)
        return num / den