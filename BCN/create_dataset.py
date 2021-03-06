import argparse
import csv
import pickle
import re
import string
import sys
from collections import Counter
from pathlib import Path

import ipdb
import spacy
from box import Box
from tqdm import tqdm

from .dataset import Part1Dataset
from common.vocab import Vocab


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset_dir', type=Path, help='Target dataset directory')  # dataset/classification
    args = parser.parse_args()

    return vars(args)


def load_data(mode, data_path, nlp):
    print('[*] Loading {} data from {}'.format(mode, data_path))
    with data_path.open() as f:
        reader = csv.DictReader(f)
        data = [r for r in reader]
        # r = data[0] format: OrderedDict([('Id', '10001'), ('text', "The Rock is...."), ('label', '4')])

    for d in tqdm(data, desc='[*] Tokenizing', dynamic_ncols=True):
        text = re.sub('-+', ' ', d['text'])
        text = re.sub('\s+', ' ', text)
        doc = nlp(text)  # text = doc = 'The Rock is destined to be the 21st ....'
        d['text'] = [token.text for token in doc]  # = ['The', 'Rock', 'is']
        # r = data[0] format: OrderedDict([('Id', '10001'), ('text', ['The', 'Rock', 'is']), ('label', '4')])
    print('[-] {} data loaded\n'.format(mode.capitalize()))

    return data


def create_vocab(data, cfg, dataset_dir):
    print('[*] Creating word vocab')
    words = Counter()
    for m, d in data.items():    # m:'train'
        bar = tqdm(
            d, desc='[*] Collecting word tokens form {} data'.format(m),
            dynamic_ncols=True)
        for dd in bar:
            # dd:OrderedDict([('Id', '10001'), ('text', ['The', 'Rock', 'is', ...]), ('label', '4')])
            words.update([w.lower() for w in dd['text']])
            # words: Counter({'the': 2, 'to': 2, "'s": 2, '`': 2, 'rock': 1, 'is': 1, '.': 1})
        bar.close()
    tokens = [w for w, _ in words.most_common(cfg.word.size)]
    word_vocab = Vocab(tokens, **cfg.word)
    word_vocab_path = (dataset_dir / 'word.pkl')
    with word_vocab_path.open(mode='wb') as f:
        pickle.dump(word_vocab, f)
    print('[-] Word vocab saved at {}\n'.format(word_vocab_path))

    print('[*] Creating char vocab')
    char_vocab = Vocab(list(string.printable), **cfg.char)
    # ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'a', 'b', 'c', 'd', 'e', ...]
    char_vocab_path = (dataset_dir / 'char.pkl')
    with char_vocab_path.open(mode='wb') as f:
        pickle.dump(char_vocab, f)
    print('[-] Char vocab saved to {}\n'.format(char_vocab_path))

    return word_vocab, char_vocab


def create_dataset(data, word_vocab, char_vocab, dataset_dir):
    for m, d in data.items():  # m(key)=mode, d(value)=data
        print('[*] Creating {} dataset'.format(m))
        dataset = Part1Dataset(d, word_vocab, char_vocab)
        # d = [OrderedDict([('Id', '10001'), ('text', ['The', 'Rock', 'is', 'destined']), ('label', '4')]),
        #     [OrderedDict([('Id1', '10002'), ('text', ['The', 'Rock', 'is', 'destined']), ('label', '1')]),
        #     [OrderedDict([('Id2', '10003'), ('text', ['The', 'Rock', 'is', 'destined']), ('label', '3')])...

        dataset_path = (dataset_dir / '{}.pkl'.format(m))
        with dataset_path.open(mode='wb') as f:
            pickle.dump(dataset, f)
        print('[-] {} dataset saved to {}\n'.format(m.capitalize(), dataset_path))


def main(dataset_dir):
    try:
        cfg = Box.from_yaml(filename=dataset_dir / 'config.yaml')
    except FileNotFoundError:
        print('[!] Dataset directory({}) must contain config.yaml'.format(dataset_dir))
        exit(1)
    print('[-] Vocabs and datasets will be saved to {}\n'.format(dataset_dir))

    output_files = ['word.pkl', 'char.pkl', 'train.pkl', 'dev.pkl', 'test.pkl']
    if any([(dataset_dir / p).exists() for p in output_files]):
        print('[!] Directory already contains saved vocab/dataset')
        exit(1)

    nlp = spacy.load('en')
    nlp.disable_pipes(*nlp.pipe_names)

    data_dir = Path(cfg.data_dir)
    data = {m: load_data(m, data_dir / '{}.csv'.format(m), nlp) for m in ['train', 'dev', 'test']}
    # len(data)=3
    # data['train'][0]=OrderedDict([('Id', '10001'), ('text', ['The', 'Rock', 'is', 'destined']), ('label', '4')])

    word_vocab, char_vocab = create_vocab(data, cfg.vocab, dataset_dir)
    create_dataset(data, word_vocab, char_vocab, dataset_dir)


if __name__ == "__main__":
    with ipdb.launch_ipdb_on_exception():
        sys.breakpointhook = ipdb.set_trace
        kwargs = parse_args()
        main(**kwargs)
