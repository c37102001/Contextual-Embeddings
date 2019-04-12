import logging
from collections import Counter
from tqdm import tqdm
import ipdb

from dataset import CorpusDataset


class Preprocessor:

    def __init__(self, embedding):
        self.embedding = embedding
        self.logging = logging.getLogger(name=__name__)

    def sentence_to_indices(self, sentence):
        tokens = sentence.split()
        return [self.embedding.to_index(token) for token in tokens]

    def collect_words(self, corpus_dir):
        # words = set()
        # with open(corpus_dir) as f:
        #     bar = tqdm(f, desc='[*] Collecting words from {}'.format(corpus_dir), dynamic_ncols=True)
        #     for l in bar:
        #         words |= {*map(lambda x: x.lower(), l.split())}
        #     bar.close()
        # return words

        words = Counter()
        with open(corpus_dir) as f:
            bar = tqdm(f, desc='[*] Collecting words from {}'.format(corpus_dir), dynamic_ncols=True)
            count = 0
            for line in bar:
                words.update([w.lower() for w in line.split()])
                count += 1
                if count >= 2000000:
                    break
            bar.close()
        tokens = [w for w, _ in words.most_common(50000)]
        return tokens

    def get_dataset(self, corpus_dir):

        data = []
        self.logging.info('loading corpus...')
        with open(corpus_dir) as f:
            bar = tqdm(f, desc='[*] Making dataset from {}'.format(corpus_dir), dynamic_ncols=True)
            count = 0
            for line in bar:
                data.append(self.sentence_to_indices('<bos> ' + line.lower() + ' <eos>'))
                count += 1
                if count >= 100000:
                    break

        training_size = len(data) - 1
        train_dataset = data[:training_size]
        valid_dataset = data[training_size:]

        padding = self.embedding.to_index('<pad>')
        return (CorpusDataset(train_dataset, padding=padding),
                CorpusDataset(valid_dataset, padding=padding, shuffle=False))
