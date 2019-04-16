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
        words = Counter()
        with open(corpus_dir) as f:
            bar = tqdm(f, desc='[*] Collecting words from {}'.format(corpus_dir), dynamic_ncols=True)
            count = 0
            for line in bar:
                line = '<bos> ' + line + ' <eos>'
                words.update([w for w in line.split()])
                count += 1
                if count >= 1000000:
                    break
            bar.close()
        tokens = [word for word, time in words.most_common() if time >= 3]

        return tokens

    def get_dataset(self, corpus_dir, max_pad_len, num_of_sentence):

        data = []
        self.logging.info('loading corpus...')
        with open(corpus_dir) as f:
            bar = tqdm(f, desc='[*] Making dataset from {}'.format(corpus_dir), dynamic_ncols=True)
            count = 0
            for line in bar:
                line = self.sentence_to_indices('<bos> ' + line + ' <eos>')
                for i in range(len(line) // max_pad_len + 1):
                    if i == len(line) // max_pad_len:
                        data.append(line[max_pad_len * i:])
                    else:
                        data.append(line[max_pad_len*i: max_pad_len*(i+1)])
                count += 1
                if count >= num_of_sentence:
                    break

        training_size = len(data)//100 * 95
        train_dataset = data[:training_size]
        valid_dataset = data[training_size:]

        padding = self.embedding.to_index('<pad>')
        return (CorpusDataset(train_dataset, padding=padding, shuffle=True, max_pad_len=max_pad_len),
                CorpusDataset(valid_dataset, padding=padding, max_pad_len=max_pad_len))
