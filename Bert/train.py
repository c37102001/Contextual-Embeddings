import argparse
import csv
import sys
from pathlib import Path
import torch

import ipdb


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('csv_dir', type=Path, help='csv_dir')  # data/classification
    args = parser.parse_args()

    return vars(args)


def load_csv_data(csv_path):
    data = {'text': [], 'label': []}
    with csv_path.open() as f:
        reader = csv.DictReader(f)
        for r in reader:
            data['text'].append(r['text'])
            data['label'].append(r['label'])
    return data


class dataset()


def main(csv_dir):

    BATCH_SIZE = 32

    csv_dir = Path(csv_dir)
    train_csv_path = csv_dir / 'train.csv'
    dev_csv_path = csv_dir / 'dev.csv'

    train_data = load_csv_data(train_csv_path)
    valid_data = load_csv_data(dev_csv_path)

    train_loader = torch.utils.data.DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True)
    valid_loader = torch.utils.data.DataLoader(dataset=valid_data, batch_size=BATCH_SIZE, shuffle=True)


if __name__ == "__main__":
    with ipdb.launch_ipdb_on_exception():
        sys.breakpointhook = ipdb.set_trace
        kwargs = parse_args()
        main(**kwargs)
