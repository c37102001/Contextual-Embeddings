import ipdb
import sys
import argparse
import logging
import json
import os
import pickle
from embedding import Embedding
from preprocessor import Preprocessor


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dest_dir', type=str, default='./ELMo/elmo_data/', help='corpus directory')
    parser.add_argument('--max_pad_len', type=int, default=64, help='data max padding length')
    args = parser.parse_args()
    return args


def main(args):
    config_path = os.path.join(args.dest_dir, 'config.json')
    embedding_pkl_path = os.path.join(args.dest_dir, 'embedding.pkl')
    train_pkl_path = os.path.join(args.dest_dir, 'train.pkl')
    valid_pkl_path = os.path.join(args.dest_dir, 'valid.pkl')

    with open(config_path) as f:
        config = json.load(f)

    preprocessor = Preprocessor(None)

    # process embedding
    words = preprocessor.collect_words(config['corpus_path'])
    logging.info('loading embedding.')
    embedding = Embedding(config['embedding_vec_path'], words)
    preprocessor.embedding = embedding
    logging.info('Saving embedding.')
    with open(embedding_pkl_path, 'wb') as f:
        pickle.dump(embedding, f)

    # process corpus into dataset
    logging.info('Processing dataset.')
    train_dataset, valid_dataset = preprocessor.get_dataset(config['corpus_path'], **config['dataset_config'])
    logging.info('Saving dataset.')
    with open(train_pkl_path, 'wb') as f:
        pickle.dump(train_dataset, f)
    with open(valid_pkl_path, 'wb') as f:
        pickle.dump(valid_dataset, f)


if __name__ == '__main__':
    logging.basicConfig(
        format='%(asctime)s | %(levelname)s | %(name)s: %(message)s',
        level=logging.INFO, datefmt='%m-%d %H:%M:%S'
    )
    with ipdb.launch_ipdb_on_exception():
        sys.breakpointhood = ipdb.set_trace
        args = parse_args()
        main(args)

