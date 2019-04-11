import ipdb
import sys
import argparse
import logging
import json
import os
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('dest_dir', type=str, help='corpus directory')
    args = parser.parse_args()
    return args


def collect_words(corpus_dir):
    words = set()
    with open(corpus_dir) as f:
        bar = tqdm(f, desc='[*] Collecting words from {}'.format(corpus_dir), dynamic_ncols=True)
        for l in bar:
            words |= {*map(lambda x: x.lower(), l.split())}
        bar.close()
    return words


def main(args):
    config_path = os.path.join(args.dest_dir, 'config.json')
    logging.info('loading configuration from {}'.format(config_path))
    with open(config_path) as f:
        config = json.load(f)
    corpus_dir = config['test_corpus_path']

    words = list(collect_words(corpus_dir))
    word_to_index = {w: i for i, w in enumerate(words)}

    ipdb.set_trace()




if __name__ == '__main__':
    logging.basicConfig(
        format='%(asctime)s | %(levelname)s | %(name)s: %(message)s',
        level=logging.INFO, datefmt='%m-%d %H:%M:%S'
    )
    with ipdb.launch_ipdb_on_exception():
        sys.breakpointhood = ipdb.set_trace
        args = parse_args()
        main(args)

