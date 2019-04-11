import ipdb
import sys
import argparse
import logging
import json
import os
import pickle

from .preprocessor import Preprocessor
from .embedding import Embedding


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('dest_dir', type=str, help='corpus directory')
    args = parser.parse_args()

    return args


def main(args):
    config_path = os.path.join(args.dest_dir, 'config.json')
    logging.info('loading configuration from {}'.format(config_path))
    with open(config_path) as f:
        config = json.load(f)
    corpus_dir = config['corpus_path']

    preprocessor = Preprocessor(None)

    words = set()
    logging.info('collecting words from {}'.format(corpus_dir))
    words |= preprocessor.collect_words(corpus_dir)

    logging.info('loading embedding from {}'.format(config['embedding_vec_path']))
    embedding = Embedding(config['embedding_vec_path'], words)
    embedding_pkl_path = os.path.join(args.dest_dir, 'embedding.pkl')
    logging.info('Saving embedding to {}'.format(embedding_pkl_path))
    with open(embedding_pkl_path, 'wb') as f:
        pickle.dump(embedding, f)


if __name__ == '__main__':

    logging.basicConfig(
        format='%(asctime)s | %(levelname)s | %(name)s: %(message)s',
        level=logging.INFO, datefmt='%m-%d %H:%M:%S'
    )

    with ipdb.launch_ipdb_on_exception():
        sys.breakpointhood = ipdb.set_trace
        args = parse_args()
        main(args)

