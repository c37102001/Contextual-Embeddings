from collections import namedtuple

import numpy as np
from tqdm import tqdm
import ipdb


SpecialToken = namedtuple('SpecialToken', ['sym', 'idx'])


class SpecialVocab:
    def __init__(self, special_tokens):
        self._special_tokens = special_tokens       # special_tokens: ['pad', 'unk']
        for i, tok in enumerate(special_tokens):
            setattr(self, tok, SpecialToken(sym='<{}>'.format(tok), idx=i))
        # self.pad = SpecialToken(sym='<pad>', idx='0')
        # self.unk = SpecialToken(sym='<unk>', idx='1')

    def __len__(self):
        return len(self._special_tokens)

    def __iter__(self):
        self._iter_idx = 0
        return self

    def __next__(self):
        if self._iter_idx < len(self):
            self._iter_idx += 1
            return getattr(self, self._special_tokens[self._iter_idx - 1])
        raise StopIteration


def load_embedding(embedding_path):
    with open(embedding_path) as f:
        lines = f.readlines()
    if len(lines[0].strip().split()) == 2:
        lines = lines[1:]
    emb = {}
    bar = tqdm(
        lines, desc='[*] Loading embedding from {}'.format(embedding_path),
        dynamic_ncols=True)
    for l in bar:
        if '\xa0' in l or '\x85' in l:
            continue
        v, *e = l.strip().split(' ')       # type(e):list of str ['-0.082752', '0.67204', '-0.064983', ...]
        emb[v.lower()] = list(map(float, e))
    bar.close()

    return emb   # {',': [-0.082752, 0.67204, -0.14987], 'the': [123, 123, 123]}


class Vocab:
    def __init__(self, tokens, special_tokens, embedding_path=None,
                 freeze_embedding=None, embedding_dimension=None, **kwargs):

        self._special = SpecialVocab(special_tokens)
        self._iv = [v.sym for v in self._special] + tokens
        # word _iv = ['<pad>', '<unk>', '.', 'the', ',', 'a', 'and', ...]
        # char _iv = ['<pad>', '<unk>', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'a', 'b', 'c', ...]
        self._vi = {v: i for i, v in enumerate(self._iv)}
        # _vi = {'<pad>':1, '<unk>':2, '.':3, 'the':4, ...}

        if embedding_path:
            if freeze_embedding is None:
                raise ValueError('Vocab: Please specify whether the embedding should be'
                                 'freezed or not')
            self.freeze_emb = freeze_embedding
            emb = load_embedding(embedding_path)
            self._emb_dim = len(emb['the'])
            self._ie = np.random.normal(
                size=(len(self._special) + len(tokens), self._emb_dim))
            self._ie[self._special.pad.idx] = np.zeros(self._emb_dim)    # _ie[unk] = np.random.normal
            for i, t in enumerate(tokens):
                idx = len(self._special) + i  # from 2
                if t in emb:
                    self._ie[idx] = np.array(emb[t])
                #   self._ie:[[pad's emb], [unk's emb], [other word's emb]]
        else:
            if freeze_embedding is not None:
                raise ValueError('Vocab: No need to specify freeze_embedding when '
                                 'embedding_path is not provided')
            self._emb_dim = embedding_dimension
            self._ie = None

    def vtoi(self, v):
        return self._vi.get(v, self._special.unk.idx)

    def itov(self, i):
        return self._iv[i]

    @property
    def emb_dim(self):
        return self._emb_dim

    @property
    def emb(self):
        return self._ie

    @property
    def sp(self):
        return self._special

    @property
    def n_sp(self):
        return len(self._special)

    def __len__(self):
        return len(self._vi)
