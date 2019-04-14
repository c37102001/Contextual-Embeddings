import re
import torch


class Embedding:

    def __init__(self, embedding_path, words=None, lower=False, rand_seed=524):
        self.word_dict = {}
        self.vectors = None
        self.lower = lower
        self.load_embedding(embedding_path, words)
        torch.manual_seed(rand_seed)

        if '<bos>' not in self.word_dict:
            self.add('<bos>')
        if '<eos>' not in self.word_dict:
            self.add('<eos>')
        if '<pad>' not in self.word_dict:
            self.add('<pad>')
        if '<unk>' not in self.word_dict:
            self.add('<unk>')

    def to_index(self, word):
        if self.lower:
            word = word.lower()
        if word not in self.word_dict:
            return self.word_dict['<unk>']
        else:
            return self.word_dict[word]

    def get_dim(self):
        return self.vectors.shape[1]

    def get_vocabulary_size(self):
        return self.vectors.shape[0]

    def add(self, word, vector=None):
        if self.lower:
            word = word.lower()

        if vector is not None:
            vector = vector.view(1, -1)
        else:
            vector = torch.empty(1, self.get_dim())
            torch.nn.init.uniform_(vector)
        self.vectors = torch.cat([self.vectors, vector], 0)
        self.word_dict[word] = len(self.word_dict)

    def load_embedding(self, embedding_path, words):
        if words is not None:
            words = set(words)

        vectors = []

        with open(embedding_path) as fp:

            row1 = fp.readline()
            # if the first row is not header
            if not re.match('^[0-9]+ [0-9]+$', row1):
                # seek to 0
                fp.seek(0)
            # otherwise ignore the header

            for i, line in enumerate(fp):
                cols = line.rstrip().split(' ')
                word = cols[0]

                # skip word not in words if words are provided
                if words is not None and word not in words:
                    continue
                elif word not in self.word_dict:
                    self.word_dict[word] = len(self.word_dict)
                    vectors.append([float(v) for v in cols[1:]])

        vectors = torch.tensor(vectors)
        if self.vectors is not None:
            self.vectors = torch.cat([self.vectors, vectors], dim=0)
        else:
            self.vectors = vectors
