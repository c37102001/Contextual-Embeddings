import logging
import ipdb
import spacy
import nltk

from multiprocessing import Pool


class Preprocessor:

    def __init__(self, embedding):
        self.embedding = embedding
        self.logging = logging.getLogger(name=__name__)

    def tokenize(self, sentence):
        """ Tokenize a sentence.
        Args:
            sentence (str): One string.
        Return:
            indices (list of str): List of tokens in a sentence.
        """

        # nlp = spacy.load('en', max_length=2000000)
        # doc = nlp(sentence)
        # return [token.text for token in doc]

        return nltk.word_tokenize(sentence)

    def collect_words(self, corpus_dir, n_workers=4):

        with open(corpus_dir) as f:
            corpus = [next(f).rstrip('\n') for _ in range(100000)]

        chunks = [
            ' '.join(corpus[i: i+len(corpus)//n_workers]) for i in range(0, len(corpus), len(corpus)//n_workers)
        ]

        with Pool(n_workers) as pool:
            chunks = pool.map_async(self.tokenize, chunks)
            words = set(sum(chunks.get(), []))

        return words


