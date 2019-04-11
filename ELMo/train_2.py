import argparse
import sys
import random
from pathlib import Path
import pickle
import ipdb
import numpy as np
import torch
import torch.nn as nn
from box import Box
from torch.utils.data import Dataset, DataLoader
from predictor import Predictor


def load_pkl(pkl_path):
    with open(pkl_path, mode='rb') as f:
        obj = pickle.load(f)

    return obj


def main(model_dir):
    cfg = Box.from_yaml(filename=model_dir / 'config.yaml')

    device = torch.device('{}:{}'.format(cfg.device.type, cfg.device.ordinal))  # 'cuda:0'
    random.seed(cfg.random_seed)
    np.random.seed(cfg.random_seed)
    torch.manual_seed(cfg.random_seed)

    log_path = model_dir / 'log.csv'
    ckpt_dir = model_dir / 'ckpts'
    ckpt_dir.mkdir()

    print('[*] Loading datasets from {}'.format(cfg.dataset_dir))
    dataset_dir = Path(cfg.dataset_dir)
    train_dataset = load_pkl(dataset_dir / 'train.pkl')
    valid_dataset = load_pkl(dataset_dir / 'valid.pkl')
    embedding = load_pkl(dataset_dir / 'embedding.pkl')
    embedding = embedding.vectors

    cfg.data_loader.batch_size //= cfg.train.n_gradient_accumulation_steps
    train_dataloader = DataLoader(
        train_dataset, collate_fn=train_dataset.collate_fn, **cfg.data_loader)
    valid_dataloader = DataLoader(
        valid_dataset, collate_fn=valid_dataset.collate_fn, **cfg.data_loader)

    predictor = Predictor(embedding)
    predictor.fit_dataset(train_dataloader, valid_dataloader)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('model_dir', type=Path, help='Target model directory')  # ./model/
    args = parser.parse_args()

    return vars(args)


if __name__ == "__main__":
    with ipdb.launch_ipdb_on_exception():
        sys.breakpointhook = ipdb.set_trace
        kwargs = parse_args()
        main(**kwargs)
