import argparse
import math
from collections import defaultdict
import numpy as np
from utils import Utils
util = Utils()

class SkipGramModel:
    def __init__(self, data, window, size, sample_constant, minimumAlpha, alpha_param, extension1):
        self.alpha = alpha_param
        self.size = size
        self.window = window
        self.sample = sample_constant
        self.min_alpha = minimumAlpha
        self.highest_occurence = 0
        self.corpus = data
        self.lowest_occurence = 0
        self.avg_occurence = 0
        self.vocabulary = self.create_vocabulary()
        self.iWeights, self.iEmbedding = self.prepare_embedding_vectors("input")
        self.oWeights, self.oEmbedding = self.prepare_embedding_vectors("output")
        self.enable_extension1 = extension1

    # Function that creates input and output embeding vectors
    def prepare_embedding_vectors(self, st):
        length = len(self.vocabulary)
        shape = (length, self.size)
        y = np.full(shape, float(0))
        if st == 'input':
            d = pow(self.size, 0.75)
            l = -0.5 / d
            h = 0.5 / d
            i = np.random.uniform(l, h, shape)
            return i, y
        else:
            w = np.full(shape, float(0))
            return w, y

    # Function that creates vocabulary through tokenized corpus
    def create_vocabulary(self):
        np.random.seed(1)
        corpus = self.corpus
        occurence = 5
        vocabulary = defaultdict(dict)
        vocabulary['num']['ocrnc'] = 0
        vocabulary, words, size = util.seperate_words_digits(corpus, vocabulary, ['num'])
        # Create a set of words with less than a particular occurence (e.g 5)
        word_set = util.find_words_with_given_max_occurence(vocabulary, occurence)
        # Remove the set of words from the vocabulary as they don't occur enough
        for i in range(len(word_set)):
            w = word_set[i]
            size -= vocabulary[w]['ocrnc']
            del vocabulary[w]
            words.remove(w)
        # Sort the modified vocabulary
        by = 'ocrnc'
        sorted_vocabulary = util.sort_words(vocabulary, by)
        # Structure voabulary with words, their frequencies and indices
        vocabulary = self.structure_vocabulary(sorted_vocabulary, vocabulary, size)
        return vocabulary

    # Function used to structure the vocabulary
    def structure_vocabulary(self, ar, vocabulary, size):
        # Track highest and lowest occurrences for the extension1
        high_f = 0
        low_f = math.inf
        # Iterate over the sorted vocabulary
        for i in range(len(ar)):
            w = ar[i]
            t = w[0]
            # For each word determine its index & frequency
            f = vocabulary[t]['ocrnc'] / size
            vocabulary[t]['index'] = i
            vocabulary[t]['frequency'] = f
            # Fetch the highest and lowest occurrences of any word in the vocabulary
            if vocabulary[t]['ocrnc']  < low_f:
                low_f = vocabulary[t]['ocrnc']
            if vocabulary[t]['ocrnc']  > high_f:
                high_f = vocabulary[t]['ocrnc']
        # Highest occurrences of a word in the vocabulary
        self.highest_occurence = high_f
        # Lowest occurrences of a word in the vocabulary
        self.lowest_occurence = low_f
        # Average occurrences of a word in the vocabulary
        self.avg_occurence = (self.highest_occurence + self.lowest_occurence) / 2
        return vocabulary

    # Forward Propagation
    def move_forward(self):
        # Negative sampled list
        ls = self.negative_sampling(self.vocabulary)
        for t in self.corpus:
            # Determine subsampled words
            if self.enable_extension1:
                ar = self.remove_words_with_high_ocrnc(t)
            else:
                ar = self.contruct_subsampled_array(t)
            # Iterate over subsampled words
            for i in range(len(ar)):
                s, t = self.refine_slice(ar, i)
                for c in range(len(s)):
                    print("...")
                    # if position of context words same as index, ignore
                    if t + c == i:
                        pass
                    # Otherwise perform back propagation using negative sampled list
                    else:
                        self.move_backwards(ls, s[c], (ar[i], 15))

    def refine_slice(self, arr, i):
        w = self.window - np.random.randint(self.window)
        l = max(0, i - w)
        r = i + w + 1
        return arr[l:r], l

    # Function to create an array of subsampled words
    def contruct_subsampled_array(self, t):
        # List to store subsampled words
        ls = []
        # Iterate over the words in the given corpus
        for i in range(len(t)):
            # If word is not str stype
            w = t[i]
            if w.isdigit():
                w = 'num'
            # Subsampling of High-Frequency Word
            if w in self.vocabulary and self.subsample(w):
                ls.append(w)
            else:
                continue
        return ls

    # Research Extension
    # Determine the highest and average frequency,
    # Discard the words with frequency >= avg (highest, average frequencies)
    def remove_words_with_high_ocrnc(self, t):
        mid_high_frequency = (self.avg_occurence + self.highest_occurence) / 2
        index = int(mid_high_frequency * 1000)
        ls = []
        # Iterate over the words in the given corpus
        for i in range(len(t)):
            # If word is not str stype
            w = t[i]
            if w.isdigit():
                w = 'num'
            # Subsampling of High-Frequency Word
            if w in self.vocabulary and vocabulary[w]['ocrnc'] >= mid_high_frequency:
                ls.append(w)
            else:
                continue
        return ls

    def subsampling_probability(self, word):
        # 1e-4
        # Fraction of the total words in the corpus that are 'word'
        z = vocabulary[word]['frequency']
        # first term
        first = np.sqrt(z / self.sample) + 1
        second = self.sample / z
        probability = first * second
        return min(probability, 1)

    # Function to perform subsampling for a particular word
    def subsample(self, word):
        # Find the subsampling probability of the word
        probability = self.subsampling_probability(word)
        # Pick its position corresponding to propability
        index = int(probability * 1000)
        # Construct array with 1s upto that position
        ls0 = np.ones(index, dtype=int)
        # Construct array with 0s after that position
        ls1 = np.zeros((1000 - index), dtype=int)
        # Concatenate the arrays
        ls = np.concatenate((ls0, ls1), axis=None)
        # If randomly picked position in the concatenated array has 1, return the word
        i = np.random.randint(len(ls))
        return word if ls[i] else None

    # Back Propagation
    # Takes context words, negative sampled words, center word and no. of negatives
    def move_backwards(self, negative_sampled_words, ctx, word):
        # If number of negatives is not > 0
        if word[1] <= 0:
            pass
        # Perform back propagation
        else:
            # Create the negative sample
            s = [(word[0], word[1]-14)]
            # self.iWeights=  self.prepare_input_embedding_vectors()
            # p = np.zeros_like(self.iWeights)
            iw = self.iWeights[self.vocabulary[ctx]['index']]
            # Pick negatively sampled words randomly
            s = self.pick_negative_sampled_words_randomly(word[1], negative_sampled_words, s, word[0])
            # Create a zero array of window size length
            ar = np.zeros(self.size)
            # Iterate over negative samples
            for i in range(len(s)):
                # Pick the sample
                x = s[i]
                # Pick the index of the negative word from the vocabulary
                ind = self.vocabulary[x[0]]['index']
                # Find the weight for the word
                w = self.oWeights[ind]
                # Apply Sigmoid function on the weights of context and negative sampled word
                uw = util.sigmoid_function(np.dot(iw, w), 500)
                uw -= x[1]
                # Update the output weights
                ar = self.update_weights(ind, uw * iw, self.oEmbedding, "output", (ar, uw, w))
            # Update the input embedding matrix
            self.update_weights(self.vocabulary[ctx]['index'], ar, self.iEmbedding, "input", None)

    # Function to modify the weights in backward propagation
    def update_weights(self, i, arr, array, st, tp):
        x = array[i]
        x += np.power(arr, 2)
        xsqrt = np.sqrt(x)
        arr /= xsqrt + 1e-6
        if st == "input":
            self.iWeights[i] -= self.alpha * arr
        else:
            a = tp[0] + (tp[1] * tp[2])
            self.oWeights[i] -= self.alpha * arr
            return a

    # Function to pick negatively sampled words randomly
    def pick_negative_sampled_words_randomly(self, num, ls, s, word):
        while num != 0:
            # Random Generation
            x = np.random.randint(len(ls))
            if ls[x] != word and (ls[x], 0) not in s:
                s.append((ls[x], 0))
            num -= 1
        return s

    # Negative Sampling
    # As skip-gram mode has a large number of weights, all of which would be updated slightly by every one of our billions of training samples!
    # Optimize it by having each training sample only modify a small percentage of the weights, rather than all
    def negative_sampling(self, v):
        ls = []
        it = []
        by = 'frequency'
        # Sorts the vocabulary with 'frequency' of words
        vocabulary = util.sort_words(v, by)
        # Iterate over the v
        for word in v:
            x = v[word]['frequency']
            y = pow(pow(x, 3), 0.25)
            it.append(y)
        # Iterate over the sorted vocabulary
        for i in range(len(vocabulary)):
            # Pick the word
            w = vocabulary[i]
            # Mathematical formula for negative sampling
            s = pow(pow(w[1], 3), 0.25)
            s /= sum(it)
            s = s * pow(10,6)
            # Construct 1-d array filled with the word
            l = np.full(int(s), w[0])
            # Append the array with list
            ls.extend(l)
        # Return the list
        return ls

    def similarity(self, word1, word2):
        ls = []
        # Index of the input word
        ind1 = self.vocabulary[word1]['index']
        # Index of the input word
        ind2 = self.vocabulary[word2]['index']
        # Determine the distance between the context word and the center word
        closeness = util.dist(self.iWeights[ind1], self.iWeights[ind2])
        return closeness

    # Function that picks similar words to a particular word
    def similar10(self, word):
        # List to store the top 10 similar words to given word
        ls = []
        # Index of the input word
        ind = self.vocabulary[word]['index']
        # Iterate over the vocabulary
        for w in self.vocabulary:
            # If picked word is the same vocab word
            if self.vocabulary[w]['index'] == ind:
                pass
            else:
                # Determine the distance between the context word and the center word
                closeness = util.dist(self.iWeights[ind], self.iWeights[self.vocabulary[w]['index']])
                # Create a tuple of the context word and cosine distance
                word_closeness_tuple = (w, closeness)
                ls.append(word_closeness_tuple)
        # Sort to pick the most similar
        ls.sort(key=lambda tup: tup[1], reverse=True)
        k = []
        i = 0
        while i < 10:
            k.append(ls[i][0])
            i+=1
        # Return top 10
        return k

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--window_size', nargs=1, type=int,  default=5,
                        required=False, help='Size of window for generating training examples (default 5)')
    parser.add_argument('--windv_size', nargs=1, type=int, default=90,
                        required=False, help='Size of windv for training (default 90)')
    parser.add_argument('--sample_constant', nargs=1, type=str, default=1e-4,
                        required=False, help='Sample constant (default 1e-4)')
    parser.add_argument('--minimum_alpha', nargs=1, type=int, default=0.0001,
                        required=False, help='Minimum value of alpha parameter (default 0.0001)')
    parser.add_argument('--alpha_param', nargs=1, type=int, default=0.37,
                        required=False, help='Alpha parameter (default 0.37)')
    parser.add_argument('--corpus', nargs=1, type=str, default="ml.txt",
                        required=False, help='Corpus file to use (default ml.txt)')
    parser.add_argument('--extension1', nargs=1, type=bool, default=True,
                        required=False, help='yes / no to enable research extension1')

    args = parser.parse_args()

    # Preprocessing of corpus
    data = util.read_file(args.corpus)
    # Parsing main function arguments
    window_size = args.window_size
    windv_size = args.windv_size
    sample_constant = args.sample_constant
    minimumAlpha = args.minimum_alpha
    alpha_param = args.alpha_param
    extension1 = args.extension1
    # Initialization of the Skip-gram model
    model = SkipGramModel(data, window_size, windv_size, sample_constant, minimumAlpha, alpha_param, extension1)
    # Vocabulary of the model
    vocabulary = model.vocabulary
    # Training of the model
    model.move_forward()
    # Results by the model
    if args.corpus == "brown.txt":
        # Find top 10 similar words for the given central word
        a = model.similar10('shade')
        b = model.similar10('landscape')
        c = model.similar10('compass')
        d = model.similar10('countless')
        e = model.similar10('large')
        f = model.similar10('day')

        # Find the cosine distance between two similar words
        print("Similar words' similarity comparison-----")
        print("Man-Woman", model.similarity("man", "woman"))
        print("City-State",  model.similarity("city", "state"))
        print("Big-Little",  model.similarity("big", "little"))
        print("Big-Large",  model.similarity("big", "large"))
        print("Money-Dollar",  model.similarity("money", "dollar"))
        print("Day-Night", model.similarity("day", "night"))

        print("\n")
        # Find the cosine distance between two dissimilar words
        print("Dissimilar words' similarity comparison-----")
        print("Man-Tree", model.similarity("man", "tree"))
        print("City-Blue",  model.similarity("city", "blue"))
        print("Big-Cheese",  model.similarity("big", "cheese"))
        print("Night-Large",  model.similarity("night", "large"))
        print("Money-Dog",  model.similarity("money", "dog"))
        print("Day-Little", model.similarity("day", "little"))
        print("\n")

    else:
        a = model.similar10('algorithms')
        b = model.similar10('data')
        c = model.similar10('artificial')
        d = model.similar10('supervised')
        e = model.similar10('machine')
        f = None

    print("Top 10 similar words for the given central word-----")
    print(str(a))
    print(str(b))
    print(str(c))
    print(str(d))
    print(str(e))
    if f:
        print(str(f))